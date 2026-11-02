import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from stepopd.common import read_jsonl, write_json
from stepopd.eval.chat_template import (
    NON_THINKING_CONDITION,
    PATCHED_NON_THINKING_CONDITION,
)
from stepopd.eval.checkpoint import validate_hf_directory
from stepopd.eval.final_eval import (
    CODE_ROOT,
    checkpoint_identity,
    stop_vllm,
    wait_vllm_ready,
)
from stepopd.eval.final_eval import (
    load_config as load_opd_config,
)

from .evaluate import load_config
from .prepare import prepare, tokenizer_fingerprint
from .rollout import collect_trajectories, collection_state


def rollout_settings(config):
    opd = load_opd_config(Path(__file__).resolve().parent.parent / "opd" / config.collection.opd_config)
    rollout = opd.actor_rollout_ref.rollout
    if (
        opd.stepopd.environment != config.sft.environment
        or Path(opd.actor_rollout_ref.model.path).resolve() != Path(config.model.path).resolve()
        or not opd.data.continuous_token.enable
        or opd.data.apply_chat_template_kwargs.enable_thinking
        or rollout.full_determinism
    ):
        raise ValueError("collection must use non-thinking OPD stochastic rollouts for the same environment and student")
    return {
        "sampling": {
            "temperature": config.collection.temperature,
            "top_p": rollout.top_p,
            "top_k": rollout.top_k,
            "repetition_penalty": 1.0,
            "ignore_eos": rollout.ignore_eos,
            "min_p": 0.0,
        },
        "prompt_length": rollout.prompt_length,
        "response_length": rollout.response_length,
        "context_limit": rollout.max_model_len,
        "max_tokens": opd.stepopd.max_tokens_per_turn,
        "retrieval_url": opd.stepopd.retrieval_url,
    }


def plan_collection(configs):
    from transformers import AutoTokenizer

    groups, tokenizers = {}, {}
    common_keys = (
        "teacher_path",
        "work_dir",
        "port",
        "gpu_memory_utilization",
        "api_server_count",
        "max_num_seqs",
        "max_num_batched_tokens",
    )
    common = {key: configs[0].collection[key] for key in common_keys}
    teacher_name = configs[0].sft.teacher_model
    for config in configs:
        settings = config.collection
        if common != {key: settings[key] for key in common_keys} or config.sft.teacher_model != teacher_name:
            raise ValueError("all configs in a suite must share the same local teacher and serving parameters")
        if (
            settings.concurrency < 1
            or settings.data_reuse_epochs < 1
            or settings.request_timeout <= 0
            or settings.request_retries < 0
            or settings.api_server_count < 1
            or settings.max_num_seqs < 1
            or settings.max_num_batched_tokens < settings.max_num_seqs
            or type(settings.max_attempts) is not int
            or settings.max_attempts < 1
            or settings.temperature < 0
            or not 0 < settings.gpu_memory_utilization < 1
            or not 0 < settings.port < 65536
        ):
            raise ValueError("teacher collection parameters must be within valid ranges")
        model = str(Path(config.model.path).expanduser().resolve())
        if model not in tokenizers:
            tokenizers[model] = AutoTokenizer.from_pretrained(model, local_files_only=True)
        tokenizer = tokenizers[model]
        environment = config.sft.environment
        rollout = rollout_settings(config)
        if config.data.max_length < rollout["prompt_length"] + rollout["response_length"]:
            raise ValueError("SFT max_length must cover the shared OPD trajectory budget")
        resolved = OmegaConf.to_container(settings, resolve=True)
        del resolved["opd_config"]
        del resolved["data_reuse_epochs"]
        signature = {
            "settings": resolved,
            "rollout": rollout,
            "output": str(Path(config.sft.data_dir).expanduser().resolve()),
            "max_length": config.data.max_length,
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        }
        if environment in groups:
            if groups[environment][3] != signature:
                raise ValueError("two students in one environment must share collection settings, data dir, length, and tokenizer")
            continue
        train_file = Path(settings.data_dir).expanduser().resolve() / "train.jsonl"
        tasks = read_jsonl(train_file)
        if not tasks or any(row["split"] != "train" or row["environment"] != environment for row in tasks):
            raise ValueError(f"{environment}: train.jsonl must be non-empty with correct split rows")
        if len({row["id"] for row in tasks}) != len(tasks):
            raise ValueError(f"{environment}: duplicate train IDs")
        groups[environment] = (config, tokenizer, tasks, signature)
    if len({item[3]["output"] for item in groups.values()}) != len(groups):
        raise ValueError("each environment must use a separate SFT data directory")
    if len({item[0].collection.rollout_dir for item in groups.values()}) != len(groups):
        raise ValueError("each environment must use a separate teacher rollout directory")
    return groups


def start_teacher(settings, template_path, context_limit, served_model):
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise ValueError("teacher collection needs at least one visible CUDA GPU")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", settings.port))
    vllm = shutil.which("vllm")
    if not vllm:
        raise FileNotFoundError("local teacher collection requires vllm in the container")
    command = [
        vllm,
        "serve",
        str(Path(settings.teacher_path).expanduser().resolve()),
        "--served-model-name",
        served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.port),
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        str(gpu_count),
        "--data-parallel-backend",
        "mp",
        "--api-server-count",
        str(settings.api_server_count),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--max-model-len",
        str(context_limit),
        "--max-num-seqs",
        str(settings.max_num_seqs),
        "--max-num-batched-tokens",
        str(settings.max_num_batched_tokens),
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--chat-template",
        str(template_path),
        "--generation-config",
        "vllm",
    ]
    print(
        f"Teacher server started: TP=1, DP={gpu_count}, API servers={settings.api_server_count}, "
        f"sample concurrency={settings.concurrency}, max_num_seqs={settings.max_num_seqs} per replica, "
        f"max_num_batched_tokens={settings.max_num_batched_tokens}, "
        f"read timeout={settings.request_timeout}s, timeout retries={settings.request_retries}",
        flush=True,
    )
    with (template_path.parent / "vllm.log").open("a") as log:
        return subprocess.Popen(
            command,
            cwd=CODE_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session="STEPOPD_MANAGED_STAGE" not in os.environ,
        )


def collect(configs):
    from transformers import AutoTokenizer

    groups = plan_collection(configs)
    targets = {
        environment: max(
            math.ceil(
                config.data.train_batch_size
                * config.trainer.total_training_steps
                * 1.2
                / config.collection.data_reuse_epochs
            )
            for config in configs
            if config.sft.environment == environment
        )
        for environment in groups
    }
    settings = configs[0].collection
    teacher_path = Path(settings.teacher_path).expanduser().resolve()
    validate_hf_directory(teacher_path)
    teacher = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
    template = teacher.chat_template
    if not isinstance(template, str) or NON_THINKING_CONDITION not in template:
        raise ValueError("teacher must use a Qwen3 chat template containing the non-thinking condition")
    if PATCHED_NON_THINKING_CONDITION not in template:
        template = template.replace(NON_THINKING_CONDITION, PATCHED_NON_THINKING_CONDITION)
    teacher_identity = checkpoint_identity(teacher_path)
    teacher_identity["template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
    work_dir = Path(settings.work_dir).expanduser().resolve()
    pending = {}
    for environment, (config, _, tasks, signature) in groups.items():
        rollout = Path(config.collection.rollout_dir).expanduser().resolve()
        identity = {
            **signature,
            "teacher_model": config.sft.teacher_model,
            "teacher": teacher_identity,
            "train_sha256": hashlib.sha256(
                (Path(config.collection.data_dir).expanduser() / "train.jsonl").read_bytes()
            ).hexdigest(),
        }
        identity_file = rollout / "collection.json"
        if identity_file.exists():
            stored = json.loads(identity_file.read_text())
            if stored != identity:
                raise ValueError(f"{environment}: collection inputs or settings changed; use new rollout and data directories")
        write_json(identity_file, identity)
        state = collection_state(rollout, tasks, config.collection.max_attempts)
        successes = sum(row["success"] for row in state.values())
        unfinished = sum(
            not row["success"] and len(row["runs"]) < config.collection.max_attempts for row in state.values()
        )
        pending[environment] = unfinished if successes < targets[environment] else 0
        print(
            f"{environment}: {successes}/{targets[environment]} successes so far, "
            f"{len(tasks)} train tasks, {pending[environment]} candidates to collect.",
            flush=True,
        )
        if targets[environment] > len(tasks):
            print(
                f"{environment}: target exceeds the number of tasks; only the first success "
                "per task is kept; collection runs until candidates are exhausted, and "
                "existing successes are used for training.",
                flush=True,
            )
    print(
        f"Teacher collection pending: {pending}; at most {settings.max_attempts} attempts per task; "
        "new attempts stop per environment once the success target is reached; "
        "ETA is printed per task as episode length varies.",
        flush=True,
    )
    served_model = f"teacher-{uuid.uuid4().hex}"
    resources = ExitStack()
    summaries = {}
    handlers = {sig: signal.signal(sig, signal.default_int_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if any(pending.values()):
            work_dir.mkdir(parents=True, exist_ok=True)
            template_path = work_dir / "non_thinking.jinja"
            temporary = template_path.with_suffix(".tmp")
            temporary.write_text(template)
            temporary.replace(template_path)
            context_limit = max(item[3]["rollout"]["context_limit"] for item in groups.values())
            process = start_teacher(settings, template_path, context_limit, served_model)
            resources.callback(stop_vllm, process)
            wait_vllm_ready(settings.port, process, served_model)
        for environment, (config, tokenizer, _, signature) in groups.items():
            item = config.collection
            args = SimpleNamespace(
                environment=environment,
                data_dir=Path(item.data_dir).expanduser().resolve(),
                data_file=Path(item.data_dir).expanduser().resolve() / "train.jsonl",
                output_dir=Path(item.rollout_dir).expanduser().resolve(),
                base_url=f"http://127.0.0.1:{item.port}/v1",
                model=config.sft.teacher_model,
                **signature["rollout"],
                concurrency=item.concurrency,
                max_attempts=item.max_attempts,
                request_timeout=item.request_timeout,
                request_retries=item.request_retries,
            )
            with ExitStack() as environment_resources:
                if (
                    pending[environment]
                    and environment == "search_r1"
                    and args.retrieval_url
                    in (
                        "http://127.0.0.1:8000/retrieve",
                        "http://localhost:8000/retrieve",
                    )
                ):
                    from stepopd.opd.search_r1.service import bm25_service

                    environment_resources.enter_context(bm25_service(CODE_ROOT, args.output_dir, time.time() + 1800))
                asyncio.run(
                    collect_trajectories(
                        args,
                        tokenizer=teacher,
                        checkpoint_identity=teacher_identity,
                        request_model=served_model,
                        target_successes=targets[environment],
                    )
                )
            summaries[environment] = prepare(
                SimpleNamespace(
                    environment=environment,
                    data_dir=Path(item.data_dir).expanduser().resolve(),
                    rollout_dir=args.output_dir,
                    output_dir=Path(config.sft.data_dir).expanduser().resolve(),
                    teacher_model=config.sft.teacher_model,
                    max_length=config.data.max_length,
                ),
                tokenizer,
            )
            print(
                f"{environment}: SFT export stats {summaries[environment]['counts']}, "
                f"success target {targets[environment]}.",
                flush=True,
            )
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            resources.close()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            collect([load_config(path, args.overrides) for path in args.configs]),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
