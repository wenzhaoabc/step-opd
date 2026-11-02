import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

GPU_REQUIREMENTS = {
    "torch": "2.11.0+cu130",
    "vllm": "0.24.0",
    "transformers": "5.9.0",
    "flash-attn": "2.8.3",
    "tensordict": "0.10.0",
    "transferqueue": "0.1.8",
}


def inspect_gpu() -> dict:
    report = {
        "ok": True,
        "python": sys.version.split()[0],
        "packages": {},
        "devices": [],
        "errors": [],
    }
    errors = report["errors"]
    try:
        if sys.version_info[:2] != (3, 12):
            errors.append("GPU image requires Python 3.12")
        for name, expected in GPU_REQUIREMENTS.items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"missing dependency: {name}")
                continue
            report["packages"][name] = actual
            if name == "flash-attn":
                actual = actual.split("+")[0]
            if actual != expected:
                errors.append(f"{name}: expected {expected}, got {actual}")
        torch = importlib.import_module("torch")
        importlib.import_module("flash_attn")
        importlib.import_module("vllm")
        verl = importlib.import_module("verl")
        if verl.__version__ != "0.9.0":
            errors.append(f"verl: expected 0.9.0, got {verl.__version__}")
        if torch.version.cuda != "13.0":
            errors.append(f"CUDA runtime: expected 13.0, got {torch.version.cuda}")
        count = torch.cuda.device_count()
        if count < 2:
            errors.append(f"expected at least 2 GPUs, got {count}")
        if count != 8:
            report["gpu_count_warning"] = f"non-standard 8-GPU setup: {count} GPUs, for quick pipeline testing only"
        for index in range(count):
            prop = torch.cuda.get_device_properties(index)
            report["devices"].append(
                {
                    "index": index,
                    "name": prop.name,
                    "memory_bytes": prop.total_memory,
                }
            )
            tensor = torch.ones((16, 16), device=f"cuda:{index}", dtype=torch.bfloat16)
            if (tensor @ tensor)[0, 0].item() != 16:
                errors.append(f"GPU {index}: BF16 matmul check failed")
            torch.cuda.synchronize(index)
            print(f"GPU preflight {index + 1}/{count}", file=sys.stderr, flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            errors.append("pip check: " + result.stdout.strip())
    except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        errors.append(f"GPU runtime check failed: {exc}")
    report["ok"] = not errors
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_gpu()
    from stepopd.common import write_json

    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
