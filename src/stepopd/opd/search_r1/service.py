import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path


def search_paths() -> tuple[Path, Path]:
    root = Path(os.environ["STEPOPD_ROOT"])
    return root / "envs/search_r1/wiki-18-bm25-index/bm25", root / "envs/search_r1/wiki-18-corpus/wiki-18.jsonl"


@contextmanager
def bm25_service(code_root: Path, output: Path, deadline: float):
    index, corpus = search_paths()
    if not index.is_dir() or not any(index.iterdir()) or not corpus.is_file():
        raise FileNotFoundError(
            f"retrieval assets missing: index {index} or corpus {corpus} (derived from STEPOPD_ROOT; see the Search-R1 README)"
        )
    if time.time() >= deadline:
        raise TimeoutError("run budget exhausted before starting BM25")
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
            port_taken = True
    except ConnectionRefusedError:
        port_taken = False
    if port_taken:
        payload = None
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/health", timeout=2
            ) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            payload = None
        if not (
            isinstance(payload, dict)
            and payload.get("ready") is True
            and payload.get("retriever") == "bm25"
        ):
            raise RuntimeError("BM25 port 8000 already occupied; refusing to connect to an unknown service")
        print(
            f"reusing running BM25 service (127.0.0.1:8000, {payload.get('documents')} documents); "
            "parallel training slots on this host share the read-only retrieval service",
            flush=True,
        )
        yield None
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / "bm25.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "stepopd.opd.search_r1.bm25",
                "--index-path",
                str(index),
                "--corpus-path",
                str(corpus),
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=code_root,
            env=os.environ
            | {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": f"{code_root / 'src'}:{code_root / 'verl'}",
            },
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session="STEPOPD_MANAGED_STAGE" not in os.environ,
        )
        try:
            started = time.time()
            last_progress = 0.0
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("BM25 failed to start; see bm25.log")
                if time.time() - last_progress >= 15:
                    print(
                        f"BM25 loading corpus/cache: elapsed {time.time() - started:.0f}s, "
                        f"budget remaining {deadline - time.time():.0f}s; first-time Arrow cache may take several minutes",
                        flush=True,
                    )
                    last_progress = time.time()
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:8000/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            else:
                raise TimeoutError("BM25 not ready within the remaining run budget")
            yield process
        finally:
            if process.poll() is None:
                if "STEPOPD_MANAGED_STAGE" in os.environ:
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a training command is required")
    with bm25_service(args.code_root, args.output_dir, time.time() + 1800):
        result = subprocess.run(args.command, cwd=args.code_root, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
