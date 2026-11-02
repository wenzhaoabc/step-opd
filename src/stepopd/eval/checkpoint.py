import codecs
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]


def validate_hf_directory(directory: Path, *, weights: bool = True) -> None:
    for name in ("config.json", "tokenizer_config.json"):
        content = json.loads((directory / name).read_text())
        if not isinstance(content, dict) or not content:
            raise ValueError(f"{directory / name} must be a non-empty JSON object")
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "spiece.model")
    if not any((directory / name).is_file() and (directory / name).stat().st_size for name in tokenizer_files):
        if not all(
            (directory / name).is_file() and (directory / name).stat().st_size for name in ("vocab.json", "merges.txt")
        ):
            raise FileNotFoundError(f"{directory} has no non-empty tokenizer file")
    if not weights:
        return
    for name in ("model.safetensors", "pytorch_model.bin"):
        index = directory / f"{name}.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"weight_map in {index} must be a non-empty object")
            for shard in set(weight_map.values()):
                if not isinstance(shard, str) or not shard or Path(shard).is_absolute() or ".." in Path(shard).parts:
                    raise ValueError(f"{index} contains an invalid weight path: {shard!r}")
                path = directory / shard
                if not path.is_file() or not path.stat().st_size:
                    raise FileNotFoundError(f"weight referenced by {index} is missing or empty: {path}")
            return
        path = directory / name
        if path.is_file() and path.stat().st_size:
            return
    raise FileNotFoundError(f"{directory} has no complete HF weights")


def merge_to_hf(shard_dir: Path, strategy: str, run_merge) -> Path:
    merged = shard_dir / "huggingface_merged"
    for directory in (shard_dir / "huggingface", merged):
        try:
            validate_hf_directory(directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"HF weights not directly usable: {directory} ({exc})", flush=True)
        else:
            print(f"reusing HF weights {directory}", flush=True)
            return directory
    if strategy not in ("fsdp", "fsdp2"):
        raise ValueError(f"{shard_dir} needs merging but strategy={strategy!r} is unsupported; only fsdp/fsdp2")
    validate_hf_directory(shard_dir / "huggingface", weights=False)
    world_size = json.loads((shard_dir / "fsdp_config.json").read_text())["world_size"]
    if type(world_size) is not int or world_size < 1:
        raise ValueError(f"world_size in {shard_dir / 'fsdp_config.json'} must be a positive integer")
    shards = [
        shard_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
    ]
    for shard in shards:
        if not shard.is_file() or not shard.stat().st_size:
            raise FileNotFoundError(f"training shard missing or empty: {shard}")
    size_mib = sum(p.stat().st_size for p in shards) / 1024**2
    print(
        f"Merging {shard_dir}: {len(shards)} shards, {size_mib / 1024:.1f} GiB; "
        f"rough estimate {60 + 2 * size_mib / 500:.0f}-{180 + 2 * size_mib / 100:.0f}s"
        " (assumes 100-500 MiB/s I/O and 60-180s init; actual time depends on CPU/memory/disk).",
        flush=True,
    )
    temporary = shard_dir / "huggingface_merged.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    started = time.monotonic()
    run_merge(
        [
            sys.executable,
            "-u",
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(shard_dir),
            "--target_dir",
            str(temporary),
        ]
    )
    validate_hf_directory(temporary)
    if merged.exists():
        shutil.rmtree(merged)
    temporary.rename(merged)
    print(f"merge finished ({time.monotonic() - started:.0f}s), kept {merged}", flush=True)
    return merged


def latest_checkpoint_step(output_root: Path) -> int | None:
    steps = [
        int(path.name[12:])
        for path in output_root.glob("global_step_*")
        if path.is_dir() and path.name[12:].isdigit()
    ]
    return max(steps, default=None)


def prepare_hf_checkpoint(output_root: Path, strategy: str) -> Path:
    output_root = output_root.expanduser().resolve()
    step = latest_checkpoint_step(output_root)
    if step is None:
        raise FileNotFoundError(f"no global_step_<number> checkpoint under {output_root}")
    actor = output_root / f"global_step_{step}" / "actor"
    if not actor.is_dir():
        raise FileNotFoundError(f"latest checkpoint is missing actor: {actor}")

    def run_merge(command):
        log_path = actor / "merge.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(CODE_ROOT / "src"), str(CODE_ROOT / "verl"), *sys.path]
        )
        with log_path.open("ab", buffering=0) as log:
            log.write((f"$ {' '.join(command)}\n").encode())
            with subprocess.Popen(
                command,
                cwd=CODE_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ) as process:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                try:
                    while chunk := process.stdout.read1(8192):
                        log.write(chunk)
                        print(decoder.decode(chunk), end="", flush=True)
                    print(decoder.decode(b"", final=True), end="", flush=True)
                    returncode = process.wait()
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
        if returncode:
            raise RuntimeError(f"checkpoint merge failed (code={returncode}); see {log_path}")

    return merge_to_hf(actor, strategy, run_merge)
