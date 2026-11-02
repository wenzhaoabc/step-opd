import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf

from stepopd.common import read_jsonl, write_json
from stepopd.eval.checkpoint import validate_hf_directory

from .prepare import digest, tokenizer_fingerprint


class SFTConfigPaths(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        source = Path(__file__).resolve()
        search_path.append(provider="stepopd-sft", path=str(source.parent))
        search_path.append(
            provider="verl-sft",
            path=str(source.parents[3] / "verl/verl/trainer/config"),
        )


Plugins.instance().register(SFTConfigPaths)


def run_training(command):
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    process = subprocess.Popen(command, start_new_session=True)
    handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        code = process.wait()
        if code:
            raise SystemExit(code if code > 0 else 128 - code)
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def validate_config(config, samples, visible_gpus):
    gpus = config.trainer.n_gpus_per_node
    sp = config.engine.ulysses_sequence_parallel_size
    batch = config.data.train_batch_size
    if (
        config.data.custom_cls.path != "pkg://stepopd.sft.dataset"
        or config.data.custom_cls.name != "AgentSFTDataset"
        or config.data.train_max_samples != -1
        or config.data.apply_chat_template_kwargs
    ):
        raise ValueError("SFT requires AgentSFTDataset, the full success-only dataset, and non-thinking encoding")
    if str(Path(config.data.train_files).expanduser().resolve()) != str(
        Path(config.sft.data_dir).expanduser().resolve() / "train/*.parquet"
    ):
        raise ValueError(
            "specify the exported dataset via sft.data_dir; do not override data.train_files separately"
        )
    if config.trainer.nnodes != 1 or gpus < 1 or gpus > visible_gpus:
        raise ValueError("SFT requires a single node with at least as many visible GPUs as configured")
    if sp < 1 or gpus % sp or batch < 1 or batch % (gpus // sp):
        raise ValueError("GPUs must be divisible by SP; global batch must be divisible by DP")
    if samples < batch:
        raise ValueError(f"success trajectories {samples} < batch={batch}; no updates after drop_last")
    if (
        config.data.micro_batch_size_per_gpu < 1
        or config.data.max_token_len_per_gpu < 1
    ):
        raise ValueError("micro batch and token budget must be positive")
    if not config.data.use_dynamic_bsz:
        if batch % ((gpus // sp) * config.data.micro_batch_size_per_gpu):
            raise ValueError("static batch must be divisible by DP x micro batch")
    if (
        config.engine.strategy != "fsdp2"
        or config.data.pad_mode != "no_padding"
        or config.data.truncation != "error"
        or not config.model.use_remove_padding
    ):
        raise ValueError("SFT requires fsdp2 / no_padding / remove_padding; truncating trajectories is forbidden")
    if config.data.val_files is not None or config.trainer.test_freq != -1:
        raise ValueError("SFT uses no val/test success trajectories; run a separate environment evaluation after training")
    if config.trainer.save_freq < 1 or config.trainer.total_training_steps < 1:
        raise ValueError("total_training_steps and checkpoint save_freq must be positive")
    if any(
        not {"model", "optimizer", "extra"} <= set(contents)
        for contents in (
            config.checkpoint.save_contents,
            config.checkpoint.load_contents,
        )
    ):
        raise ValueError("checkpoint must save model/optimizer/extra for resumability")


def validate_resume(config, output, identity):
    mode = config.trainer.resume_mode
    requested = config.trainer.resume_from_path
    if mode not in ("auto", "disable", "resume_path"):
        raise ValueError("unknown resume_mode")
    tracker = output / "latest_checkpointed_iteration.txt"
    checkpoint = None
    if mode == "disable":
        if requested is not None or list(output.glob("global_step_*")):
            raise ValueError("resume_mode=disable requires a fresh checkpoint output directory")
    elif requested is not None:
        checkpoint = Path(requested).expanduser().resolve()
    elif mode == "resume_path":
        raise ValueError("resume_path requires resume_from_path")
    elif tracker.is_file():
        checkpoint = output / f"global_step_{int(tracker.read_text().strip())}"
    elif list(output.glob("global_step_*")):
        raise ValueError("checkpoints exist without a completion marker; inspect them and specify a full resume path explicitly")
    identity_path = output / "training.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("training data, model, or parallel config changed; use a new training output directory")
    if checkpoint is not None:
        if json.loads((checkpoint.parent / "training.json").read_text()) != identity:
            raise ValueError("resume checkpoint training identity mismatch")
        world_size = config.trainer.n_gpus_per_node
        files = [checkpoint / "fsdp_config.json"]
        for rank in range(world_size):
            for kind in ("model", "optim", "extra_state"):
                files.append(
                    checkpoint / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
                )
        files.extend(
            checkpoint / f"data_{rank}.pt"
            for rank in range(
                world_size // config.engine.ulysses_sequence_parallel_size
            )
        )
        for path in files:
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"incomplete resume state: {path}")
        config.trainer.resume_mode = "resume_path"
        config.trainer.resume_from_path = str(checkpoint)
    return identity_path


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    data_dir = Path(config.sft.data_dir).expanduser().resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if (
        manifest["environment"] != config.sft.environment
        or manifest["teacher_model"] != config.sft.teacher_model
        or manifest["max_length"] != config.data.max_length
        or manifest["encoding"] != "qwen3_continuous_text_sft_v1"
    ):
        raise ValueError("SFT data source, environment, or length mismatch with experiment config")
    records = read_jsonl(data_dir / "records.jsonl")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("duplicate SFT export record IDs")
    kept = [row for row in records if row["status"] == "kept"]
    shards = sorted((data_dir / "train").glob("*.parquet"))
    if {data_dir / row["shard"] for row in kept} != set(shards):
        raise ValueError("SFT shards mismatch incremental export records; restore prepare first")
    if any(pq.ParquetFile(path).metadata.num_rows != 1 for path in shards):
        raise ValueError("each SFT shard must be one full success trajectory")
    validate_config(config, len(shards), torch.cuda.device_count())
    steps_per_epoch = len(shards) // config.data.train_batch_size
    target_steps = config.trainer.total_training_steps
    config.trainer.total_epochs = (
        target_steps + steps_per_epoch - 1
    ) // steps_per_epoch
    model_path = Path(config.model.path).expanduser().resolve()
    validate_hf_directory(model_path)
    if (
        config.model.tokenizer_path is not None
        or config.model.custom_chat_template is not None
    ):
        raise ValueError("SFT uses the student model's own tokenizer and chat template")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer_fingerprint(tokenizer) != manifest["tokenizer_sha256"]:
        raise ValueError(
            "student tokenizer mismatch with the one used by prepare; re-run prepare"
        )
    output = Path(config.trainer.default_local_dir).expanduser().resolve()
    identity = {
        "manifest": manifest,
        "records_sha256": digest(records),
        "shards": [str(path) for path in shards],
        "model_path": str(model_path),
        "model": OmegaConf.to_container(config.model, resolve=True),
        "data": OmegaConf.to_container(config.data, resolve=True),
        "engine": OmegaConf.to_container(config.engine, resolve=True),
        "optim": OmegaConf.to_container(config.optim, resolve=True),
        "gpus": config.trainer.n_gpus_per_node,
        "seed": config.trainer.seed,
    }
    identity_path = validate_resume(config, output, identity)
    write_json(identity_path, identity)
    launch_dir = output / "launches" / uuid4().hex
    write_json(launch_dir / "config.json", OmegaConf.to_container(config, resolve=True))
    native = OmegaConf.to_container(config, resolve=True)
    del native["sft"]
    del native["collection"]
    native["data"]["train_files"] = [str(path) for path in shards]
    native["model"]["path"] = str(model_path)
    native["trainer"]["default_local_dir"] = str(output)
    write_json(launch_dir / "native.yaml", native)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={config.trainer.n_gpus_per_node}",
        "-m",
        "verl.trainer.sft_trainer",
        "--config-path",
        str(launch_dir),
        "--config-name",
        "native",
    ]
    record = {
        "algorithm": "sft",
        "environment": config.sft.environment,
        "config": str(launch_dir / "config.json"),
        "command": command,
        "samples": len(shards),
        "data_manifest": manifest,
    }
    write_json(launch_dir / "launch.json", record)
    write_json(output / "launch.json", record)
    if config.trainer.resume_mode == "resume_path":
        completed_steps = int(
            Path(config.trainer.resume_from_path).name.removeprefix("global_step_")
        )
        if completed_steps >= target_steps:
            print(
                f"SFT already complete: checkpoint step={completed_steps}, "
                f"target={target_steps}; skipping training.",
                flush=True,
            )
            return
    print(
        f"SFT: {len(shards)} trajectories, batch={config.data.train_batch_size}, "
        f"up to {config.data.max_length} tokens each, {target_steps} target updates, "
        f"{config.trainer.total_epochs} epochs, "
        f"checkpoint every {config.trainer.save_freq} updates; native tqdm shows progress and ETA.",
        flush=True,
    )
    run_training(command)


if __name__ == "__main__":
    main()
