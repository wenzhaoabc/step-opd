import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from .chat_template import patch_non_thinking
from .checkpoint import prepare_hf_checkpoint

CODE_ROOT = Path(__file__).resolve().parents[3]
SERVED_MODEL = "student"
CONCURRENCY = {"alfworld": 64, "scienceworld": 32, "search_r1": 128}
EVALS_PER_SAMPLE = 5
CONTEXT_LIMIT = {"alfworld": 16384, "scienceworld": 30000, "search_r1": 16384}


def load_config(config: Path, overrides: list[str] | None = None):
    from hydra import compose, initialize_config_dir

    from stepopd.opd import train  # noqa: F401  import registers SearchPathPlugin

    config = config.expanduser().resolve()
    for package in ("opd", "grpo", "tcod_f2b", "opsd"):
        root = CODE_ROOT / "src" / "stepopd" / package
        if config.is_relative_to(root):
            directory = root
            break
    else:
        raise ValueError("experiment config must live under src/stepopd/{opd,grpo,tcod_f2b,opsd}/")
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(
            config_name=str(config.relative_to(directory).with_suffix("")),
            overrides=overrides,
        )


def checkpoint_identity(hf_dir: Path) -> dict:
    hf_dir = hf_dir.resolve()
    files = []
    for path in sorted(hf_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append([str(path.relative_to(hf_dir)), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    if not files:
        raise ValueError(f"checkpoint directory is empty: {hf_dir}")
    return {"path": str(hf_dir), "files": files}


def start_vllm(
    hf_dir: Path,
    port: int,
    log_path: Path,
    served_model: str,
    gpu_util: float = 0.85,
    max_model_len: int = 32768,
) -> subprocess.Popen:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"vLLM port {port} is unavailable; refusing to reuse an existing server") from exc
    vllm = shutil.which("vllm")
    if not vllm:
        raise FileNotFoundError("vllm not found in PATH; final_eval must run in the same container as training")
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("evaluation requires at least one visible CUDA GPU; check device allocation and CUDA_VISIBLE_DEVICES")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        vllm,
        "serve",
        str(hf_dir),
        "--served-model-name",
        served_model,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        str(gpu_count),
        "--data-parallel-backend",
        "mp",
        "--gpu-memory-utilization",
        str(gpu_util),
        "--max-model-len",
        str(max_model_len),
    ]
    visible = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else "unset"
    print(
        f"final_eval: starting vllm serve (TP=1, DP={gpu_count}, CUDA_VISIBLE_DEVICES={visible})",
        flush=True,
    )
    with log_path.open("a") as log:
        return subprocess.Popen(
            command,
            cwd=CODE_ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session="STEPOPD_MANAGED_STAGE" not in os.environ,
        )


def wait_vllm_ready(port: int, process: subprocess.Popen, served_model: str, timeout: float = 1200) -> None:
    started = time.monotonic()
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited early (code={process.returncode}); see vllm.log")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    models = [model["id"] for model in json.load(response)["data"]]
                    if models != [served_model]:
                        raise RuntimeError(f"model on vLLM port {port} does not belong to this launch: {models!r}")
                    print(f"final_eval: vLLM ready ({time.monotonic() - started:.0f}s)", flush=True)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready within {timeout:.0f}s; see vllm.log")


def stop_vllm(process: subprocess.Popen) -> None:
    managed = "STEPOPD_MANAGED_STAGE" in os.environ
    if process.poll() is None:
        if managed:
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if not managed:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print("final_eval: vLLM stopped", flush=True)


async def run_evaluation(
    environment: str,
    output_dir: Path,
    port: int,
    temperature: float,
    evals_per_sample: int,
    concurrency: int,
    *,
    data_dir: Path,
    retrieval_url: str,
    hf_dir: Path,
    served_model: str,
) -> dict:
    from stepopd.opd.evaluate import evaluate

    manifest = data_dir / "test.jsonl"
    eval_args = SimpleNamespace(
        environment=environment,
        data_file=manifest,
        data_dir=data_dir,
        output_dir=output_dir,
        base_url=f"http://127.0.0.1:{port}/v1",
        model=SERVED_MODEL,
        max_episodes=None,
        max_tokens=1024,
        temperature=temperature,
        context_limit=CONTEXT_LIMIT[environment],
        retrieval_url=retrieval_url,
        concurrency=concurrency,
        evals_per_sample=evals_per_sample,
    )
    return await evaluate(eval_args, checkpoint_identity=checkpoint_identity(hf_dir), request_model=served_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="experiment YAML path (environment/experiment_name/output are derived from it)"
    )
    parser.add_argument(
        "--output-root", type=Path, default=None, help="override experiment output root (default: trainer.default_local_dir from the YAML)"
    )
    parser.add_argument("--overrides", nargs="*", default=[], help="Hydra overrides, same as training")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--temperature", type=float, default=0.4, help="evaluation sampling temperature (matches training-time val; 0 for greedy)")
    parser.add_argument(
        "--evals-per-sample",
        type=int,
        default=EVALS_PER_SAMPLE,
        help="evaluations per sample to estimate per-sample variance (1 disables)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None, help="override default concurrency (alfworld 64 / scienceworld 32 / search_r1 128)"
    )
    parser.add_argument(
        "--vllm-gpu-util",
        type=float,
        default=0.85,
        help="vLLM GPU memory fraction (affects serving throughput only; not part of evaluation identity)",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=32768,
        help="vLLM serving window; tightening it to context_limit+1 reduces scheduling headroom (does not affect results)",
    )
    args = parser.parse_args()
    if args.evals_per_sample < 1:
        parser.error("evals-per-sample must be positive")

    cfg = load_config(args.config, args.overrides)
    environment = cfg.stepopd.environment.removesuffix("_f2b")
    default_root = Path(cfg.trainer.default_local_dir)
    strategy = cfg.actor_rollout_ref.actor.strategy
    output_root = (args.output_root or default_root).expanduser().resolve()
    hf_dir = prepare_hf_checkpoint(output_root, strategy)
    patch_non_thinking(hf_dir)
    evaluation_dir = output_root / "evaluation" / hf_dir.parent.parent.name
    served_model = f"{SERVED_MODEL}-{uuid.uuid4().hex}"
    print(f"final_eval: {environment}, output root {output_root}, checkpoint={hf_dir}", flush=True)

    with ExitStack() as resources:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, lambda number, frame: sys.exit(128 + number))
            resources.callback(signal.signal, signum, previous)
        process = start_vllm(
            hf_dir,
            args.port,
            evaluation_dir / "vllm.log",
            served_model,
            gpu_util=args.vllm_gpu_util,
            max_model_len=args.vllm_max_model_len,
        )
        resources.callback(stop_vllm, process)
        if environment == "search_r1" and cfg.stepopd.retrieval_url in (
            "http://127.0.0.1:8000/retrieve",
            "http://localhost:8000/retrieve",
        ):
            from stepopd.opd.search_r1.service import bm25_service

            resources.enter_context(bm25_service(CODE_ROOT, output_root, time.time() + 86400))
        wait_vllm_ready(args.port, process, served_model)
        summary = asyncio.run(
            run_evaluation(
                environment,
                evaluation_dir,
                args.port,
                args.temperature,
                args.evals_per_sample,
                args.concurrency or CONCURRENCY[environment],
                data_dir=Path(cfg.stepopd.data_dir).expanduser().resolve(),
                retrieval_url=cfg.stepopd.retrieval_url,
                hf_dir=hf_dir,
                served_model=served_model,
            )
        )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
