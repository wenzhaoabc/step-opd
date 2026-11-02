import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from stepopd.common import append_jsonl, read_jsonl, write_json


def audit_training(
    logs: list[Path],
    before: Path,
    after: Path,
    output: Path,
    algorithm: str,
    rollout_dir: Path | None = None,
    group_size: int | None = None,
    teacher_evidence: Path | None = None,
) -> dict:
    report = {
        "status": "partial",
        "algorithm": algorithm,
        "metrics": {},
        "steps_by_log": {},
        "checkpoints": [],
        "parameter_change": {"status": "unverified"},
        "group_reward_variance": {"status": "unverified"},
        "teacher_logprob": {
            "status": "unverified",
            "reason": "vanilla console does not log teacher per-token logprob values",
        },
    }
    categories = {
        "loss": [],
        "grad_norm": [],
        "memory_peak": [],
        "distillation_abs_loss": [],
    }
    pattern = re.compile(r"([\w/.-]+):(-?(?:\d+(?:\.\d*)?(?:e[+-]?\d+)?|inf|nan))", re.IGNORECASE)
    for path in logs:
        steps = set()
        if not path.is_file():
            report["steps_by_log"][str(path)] = []
            continue
        with path.open(errors="replace") as handle:
            for line in handle:
                if not re.search(r"\bstep:\d+", line):
                    continue
                values = dict(pattern.findall(line))
                step = int(values["step"])
                steps.add(step)
                for key, raw in values.items():
                    category = (
                        "grad_norm"
                        if "grad_norm" in key
                        else "memory_peak"
                        if "max_memory_" in key
                        else "distillation_abs_loss"
                        if "distillation/abs_loss" in key
                        else "loss"
                        if "loss" in key
                        else None
                    )
                    if category is not None:
                        value = float(raw)
                        categories[category].append(
                            {
                                "step": step,
                                "key": key,
                                "value": value if math.isfinite(value) else None,
                                "finite": math.isfinite(value),
                                "log": str(path),
                            }
                        )
        report["steps_by_log"][str(path)] = sorted(steps)
    expected_steps = [[1], [2]]
    report["resume_updates"] = {
        "status": "verified" if list(report["steps_by_log"].values()) == expected_steps else "unverified",
        "expected_steps_per_log": expected_steps,
    }
    for category, records in categories.items():
        report["metrics"][category] = {
            "status": "unverified"
            if not records
            else ("verified" if all(item["finite"] for item in records) else "nonfinite"),
            "records": records,
        }
    if algorithm != "opd":
        report["teacher_logprob"] = {"status": "not_applicable"}
    elif teacher_evidence is not None and not teacher_evidence.is_file():
        report["teacher_logprob"] = {
            "status": "unverified",
            "reason": f"teacher evidence file does not exist: {teacher_evidence}",
        }
    elif teacher_evidence is not None:
        rows = read_jsonl(teacher_evidence)
        checked = []
        for row in rows:
            finite = row["finite"] and all(
                math.isfinite(row[key])
                for key in (
                    "teacher_logprob_min",
                    "teacher_logprob_max",
                    "teacher_logprob_mean",
                )
            )
            checked.append(
                {
                    "global_step": row["global_step"],
                    "uid": row["uid"],
                    "finite": finite,
                    "aligned": row["aligned"],
                    "scored_tokens": row["scored_tokens"],
                    "teacher_logprob_min": row["teacher_logprob_min"],
                    "teacher_logprob_max": row["teacher_logprob_max"],
                    "teacher_logprob_mean": row["teacher_logprob_mean"],
                }
            )
        valid = bool(checked) and {row["global_step"] for row in checked} == {1, 2}
        valid = valid and all(row["finite"] and row["aligned"] and row["scored_tokens"] > 0 for row in checked)
        teacher_status = (
            "misaligned"
            if any(not row["aligned"] for row in checked)
            else "nonfinite"
            if any(not row["finite"] and row["scored_tokens"] > 0 for row in checked)
            else "verified"
            if valid
            else "unverified"
        )
        report["teacher_logprob"] = {
            "status": teacher_status,
            "source": str(teacher_evidence),
            "records": checked,
        }
    for checkpoint in (before, after):
        shards = defaultdict(set)
        empty = []
        for path in checkpoint.rglob("*.pt"):
            match = re.fullmatch(r"(model|optim|extra_state)_world_size_(\d+)_rank_(\d+)\.pt", path.name)
            if match:
                shards[(str(path.parent.relative_to(checkpoint)), match[1], int(match[2]))].add(int(match[3]))
                if path.stat().st_size == 0:
                    empty.append(str(path))
        complete = bool(shards) and not empty and {key[1] for key in shards} == {"model", "optim", "extra_state"}
        complete = complete and all(ranks == set(range(key[2])) for key, ranks in shards.items())
        record = {
            "path": str(checkpoint),
            "status": "verified" if complete else "unverified",
            "shard_groups": [
                {
                    "subdir": key[0],
                    "kind": key[1],
                    "world_size": key[2],
                    "ranks": sorted(ranks),
                }
                for key, ranks in shards.items()
            ],
            "empty_files": empty,
        }
        report["checkpoints"].append(record)
    write_json(output, report)
    try:
        from safetensors import safe_open

        maps = []
        for checkpoint in (before, after):
            mapping = {}
            for path in sorted(checkpoint.rglob("*.safetensors")):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        mapping[key] = path
            maps.append(mapping)
        common = sorted(set(maps[0]) & set(maps[1]))
        if not common:
            raise ValueError("no directly comparable HF safetensors; distributed PT format not sampled, original checkpoints kept")
        records = []
        for key in common[:32]:
            with safe_open(maps[0][key], framework="pt", device="cpu") as first:
                with safe_open(maps[1][key], framework="pt", device="cpu") as second:
                    left, right = first.get_slice(key), second.get_slice(key)
                    if left.get_shape() != right.get_shape():
                        raise ValueError(f"same-name tensor shapes differ: {key}")
                    slices = tuple(slice(0, min(size, 4)) for size in left.get_shape())
                    a, b = left[slices].float(), right[slices].float()
                    difference = (a - b).abs()
                    delta = float(difference.max())
                    record = {
                        "tensor": key,
                        "sampled_values": a.numel(),
                        "changed_values": int((a != b).sum()),
                        "max_abs_change": delta if math.isfinite(delta) else None,
                        "finite": math.isfinite(delta),
                    }
                    records.append(record)
                    append_jsonl(output.with_suffix(".parameters.jsonl"), record)
        report["parameter_change"] = {
            "status": "nonfinite"
            if any(not item["finite"] for item in records)
            else "verified"
            if any(item["changed_values"] for item in records)
            else "unverified",
            "scope": "first 4 elements/axes of same-name tensors, up to 32 tensors; no change does not imply the whole model is unchanged",
            "samples": records,
        }
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        report["parameter_change"] = {"status": "unverified", "reason": str(exc)}
    write_json(output, report)
    if algorithm == "grpo" and rollout_dir is not None:
        groups = defaultdict(list)
        missing_id = 0
        seen = set()
        duplicate_ids = 0
        nonfinite_rewards = 0
        for path in sorted(rollout_dir.glob("*.jsonl")):
            for row in read_jsonl(path):
                if "uid" not in row or not re.fullmatch(r".+_\d+_\d+", str(row["uid"])):
                    missing_id += 1
                    continue
                group_id = str(row["uid"]).rsplit("_", 2)[0]
                identity = (row["step"], row["uid"])
                duplicate_ids += int(identity in seen)
                seen.add(identity)
                reward = float(row["score"])
                if not math.isfinite(reward):
                    nonfinite_rewards += 1
                    continue
                groups[(row["step"], group_id)].append(reward)
        records = [
            {
                "step": key[0],
                "group_id": key[1],
                "rewards": rewards,
                "variance": statistics.pvariance(rewards),
            }
            for key, rewards in groups.items()
            if len(rewards) >= 2
        ]
        if nonfinite_rewards:
            report["group_reward_variance"] = {
                "status": "nonfinite",
                "nonfinite_rewards": nonfinite_rewards,
            }
        elif (
            records
            and len(records) == len(groups)
            and not missing_id
            and not duplicate_ids
            and {row["step"] for row in records} == {1, 2}
            and all(len(row["rewards"]) == group_size for row in records)
        ):
            report["group_reward_variance"] = {
                "status": "no_learning_signal" if all(row["variance"] == 0 for row in records) else "verified",
                "groups": records,
            }
        else:
            report["group_reward_variance"] = {
                "status": "unverified",
                "missing_group_ids": missing_id,
                "duplicate_ids": duplicate_ids,
                "reason": "no complete native uid groups; same prompt/task is not a substitute for a GRPO group",
                "groups": records,
            }
    elif algorithm != "grpo":
        report["group_reward_variance"] = {"status": "not_applicable"}
    evidence = [report["metrics"][key]["status"] for key in ("loss", "grad_norm", "memory_peak")]
    evidence += [row["status"] for row in report["checkpoints"]]
    evidence += [report["resume_updates"]["status"]]
    evidence += [report[key]["status"] for key in ("parameter_change", "group_reward_variance", "teacher_logprob")]
    report["status"] = (
        "misaligned"
        if "misaligned" in evidence
        else "nonfinite"
        if "nonfinite" in evidence
        else "no_learning_signal"
        if "no_learning_signal" in evidence
        else "verified"
        if all(item in {"verified", "not_applicable"} for item in evidence)
        else "partial"
    )
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("sft", "grpo", "opd"), required=True)
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--teacher-evidence", type=Path)
    args = parser.parse_args()
    if args.algorithm == "grpo" and (args.rollout_dir is None or args.group_size is None):
        parser.error("--algorithm grpo requires both --rollout-dir and --group-size")
    report = audit_training(
        args.log,
        args.before,
        args.after,
        args.output,
        args.algorithm,
        args.rollout_dir,
        args.group_size,
        args.teacher_evidence,
    )
    print(
        json.dumps(
            {"report": str(args.output), "evidence": report},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
