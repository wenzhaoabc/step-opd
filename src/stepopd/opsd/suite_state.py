import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from stepopd.common import write_json
from stepopd.eval.checkpoint import latest_checkpoint_step
from stepopd.eval.final_eval import load_config

STATE_DIR = "suite"
TRAIN_ARTIFACTS = ("skills_manifest.json", "latest_checkpointed_iteration.txt")


def digest(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def config_key(config: Path, overrides: list[str]) -> str:
    return digest("\n".join([str(config), *sorted(overrides)]))[:16]


def resolve(config: Path, overrides: list[str]) -> tuple[Path, str]:
    cfg = load_config(Path(config), list(overrides))
    output = Path(cfg.trainer.default_local_dir).expanduser().resolve()
    plain = OmegaConf.to_container(cfg, resolve=True)
    return output, digest(json.dumps(plain, sort_keys=True))


def marker_path(output: Path, phase: str, key: str) -> Path:
    return output / STATE_DIR / f"{phase}_complete.{key}.json"


def check(args) -> int:
    config = Path(args.config).expanduser().resolve()
    try:
        output, current = resolve(config, args.overrides)
        marker = marker_path(output, args.phase, config_key(config, args.overrides))
        if not marker.is_file():
            print(f"incomplete (no marker): {marker}")
            return 1
        stored = json.loads(marker.read_text())
        if stored.get("resolved_sha256") != current:
            print(f"config changed, rerun needed: {config}")
            return 1
        missing = [name for name in stored.get("artifacts", []) if not Path(name).exists()]
        if missing:
            print(f"artifacts missing, rerun needed: {', '.join(missing)}")
            return 1
        if args.phase == "eval":
            step = latest_checkpoint_step(output)
            if stored.get("checkpoint_step") != step:
                print(
                    f"new checkpoint available (step {stored.get('checkpoint_step')} -> {step}), "
                    "re-evaluation needed"
                )
                return 1
    except Exception as exc:  # noqa: BLE001
        print(f"completion status undetermined ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    print(f"complete: {config} {args.phase} (marker {marker})")
    return 0


def mark(args) -> int:
    config = Path(args.config).expanduser().resolve()
    try:
        output, current = resolve(config, args.overrides)
        step = latest_checkpoint_step(output)
        if args.phase == "train":
            if step is None or not (output / "skills_manifest.json").is_file():
                print(
                    f"no training artifacts (global_step_* / skills_manifest.json), "
                    f"refusing to write completion marker: {output}",
                    file=sys.stderr,
                )
                return 1
            artifacts = [output / name for name in TRAIN_ARTIFACTS]
            checkpoint_step = None
        else:
            summary = output / "evaluation" / f"global_step_{step}" / "summary.json"
            if step is None or not summary.is_file():
                print(f"evaluation produced no summary.json, refusing to write completion marker: {summary}", file=sys.stderr)
                return 1
            artifacts = [summary]
            checkpoint_step = step
        marker = marker_path(output, args.phase, config_key(config, args.overrides))
        write_json(
            marker,
            {
                "phase": args.phase,
                "config": str(config),
                "overrides": list(args.overrides),
                "resolved_sha256": current,
                "artifacts": [str(path) for path in artifacts],
                "checkpoint_step": checkpoint_step,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"cannot write completion marker ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    print(f"completion marker written: {marker} (checkpoint_step={step})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("check", check), ("mark", mark)):
        subparser = subparsers.add_parser(name, allow_abbrev=False)
        subparser.add_argument("--config", type=Path, required=True, help="absolute path to the experiment YAML")
        subparser.add_argument("--phase", choices=("train", "eval"), required=True)
        subparser.add_argument("overrides", nargs="*", help="Hydra overrides, same as training/evaluation")
        subparser.set_defaults(handler=handler)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
