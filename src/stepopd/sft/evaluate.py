import argparse
import asyncio
import hashlib
import json
import signal
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stepopd.common import write_json
from stepopd.eval.chat_template import patch_non_thinking
from stepopd.eval.checkpoint import merge_to_hf
from stepopd.eval.final_eval import (
    CODE_ROOT,
    CONCURRENCY,
    EVALS_PER_SAMPLE,
    run_evaluation,
    start_vllm,
    stop_vllm,
    wait_vllm_ready,
)

from .train import run_training


def load_config(path, overrides):
    path = path.expanduser().resolve()
    directory = Path(__file__).resolve().parent
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(
            config_name=str(path.relative_to(directory).with_suffix("")),
            overrides=overrides,
        )


def latest_checkpoint(output, config):
    step = int((output / "latest_checkpointed_iteration.txt").read_text().strip())
    if step < 1:
        raise ValueError("SFT checkpoint step must be a positive integer")
    checkpoint = output / f"global_step_{step}"
    world_size = json.loads((checkpoint / "fsdp_config.json").read_text())["world_size"]
    if type(world_size) is not int or world_size != config.trainer.n_gpus_per_node:
        raise ValueError("checkpoint world_size mismatch with training record")
    files = [
        checkpoint / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        for kind in ("model", "optim", "extra_state")
    ]
    files.extend(
        checkpoint / f"data_{rank}.pt" for rank in range(world_size // config.engine.ulysses_sequence_parallel_size)
    )
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"incomplete SFT checkpoint: {path}")
    return checkpoint


def evaluate(config, port=8100, temperature=0.4, evals_per_sample=EVALS_PER_SAMPLE):
    output = Path(config.trainer.default_local_dir).expanduser().resolve()
    launch = json.loads((output / "launch.json").read_text())
    saved = OmegaConf.load(launch["config"])
    for key in ("sft", "model.path", "engine", "trainer.n_gpus_per_node"):
        if OmegaConf.select(config, key) != OmegaConf.select(saved, key):
            raise ValueError(f"eval config {key} mismatch with actual training record")
    manifest = launch["data_manifest"]
    if manifest != json.loads((Path(config.sft.data_dir).expanduser() / "manifest.json").read_text()):
        raise ValueError("SFT data manifest changed")
    environment = manifest["environment"]
    data_dir = Path(manifest["source"]["data_dir"]).expanduser().resolve()
    retrieval_url = manifest["source"]["retrieval_url"]
    checkpoint = latest_checkpoint(output, saved)
    evaluation_root = checkpoint / "final_eval"
    identity = {
        "checkpoint": str(checkpoint),
        "environment": environment,
        "data_dir": str(data_dir),
        "manifest_sha256": hashlib.sha256((data_dir / "test.jsonl").read_bytes()).hexdigest(),
        "retrieval_url": retrieval_url,
        "port": port,
        "temperature": temperature,
        "evals_per_sample": evals_per_sample,
    }
    identity_file = evaluation_root / "identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("this checkpoint already has an evaluation with different settings; inspect the final_eval directory")
    write_json(identity_file, identity)
    complete = evaluation_root / "complete.json"
    if complete.exists():
        print(f"SFT: skipping completed evaluation: {evaluation_root}", flush=True)
        return json.loads(complete.read_text())
    hf_dir = merge_to_hf(checkpoint, "fsdp", run_training)
    patch_non_thinking(hf_dir)
    served_model = f"student-{uuid.uuid4().hex}"
    resources = ExitStack()
    handlers = {sig: signal.signal(sig, signal.default_int_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        process = start_vllm(hf_dir, port, evaluation_root / "evaluation" / "vllm.log", served_model)
        resources.callback(stop_vllm, process)
        if environment == "search_r1" and retrieval_url in (
            "http://127.0.0.1:8000/retrieve",
            "http://localhost:8000/retrieve",
        ):
            from stepopd.opd.search_r1.service import bm25_service

            resources.enter_context(bm25_service(CODE_ROOT, evaluation_root, time.time() + 86400))
        wait_vllm_ready(port, process, served_model)
        summary = asyncio.run(
            run_evaluation(
                environment,
                evaluation_root / "evaluation",
                port,
                temperature,
                evals_per_sample,
                CONCURRENCY[environment],
                data_dir=data_dir,
                retrieval_url=retrieval_url,
                hf_dir=hf_dir,
                served_model=served_model,
            )
        )
        write_json(complete, summary)
        return summary
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            resources.close()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--evals-per-sample", type=int, default=EVALS_PER_SAMPLE)
    args = parser.parse_args()
    if args.evals_per_sample < 1 or args.temperature < 0 or not 0 < args.port < 65536:
        parser.error("evals-per-sample must be positive, temperature non-negative, and port in 1..65535")
    summary = evaluate(
        load_config(args.config, args.overrides),
        args.port,
        args.temperature,
        args.evals_per_sample,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
