import argparse
import json
from pathlib import Path

from stepopd.common import append_jsonl, read_jsonl, write_json

from .alfworld import data as alfworld_data
from .episode import PROMPTS
from .scienceworld import data as scienceworld_data
from .search_r1 import data as search_r1_data


def prepare_records(environment, data_dir, source_dir=None, val_size=128):
    if environment == "alfworld":
        return alfworld_data.prepare_records(data_dir)
    if environment == "scienceworld":
        return scienceworld_data.prepare_records()
    if environment == "search_r1":
        return search_r1_data.prepare_records(source_dir, val_size)
    raise ValueError(f"Unknown environment: {environment}")


def prepare(environment, data_dir, source_dir=None, val_size=128):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if val_size < 1:
        raise ValueError("val-size must be positive")
    if source_dir is not None and Path(source_dir).resolve() == data_dir.resolve():
        raise ValueError("Search-R1 source-dir and data-dir must differ")
    paths = {split: data_dir / f"{split}.jsonl" for split in ["train", "val", "test"]}
    existing = {}
    for split, path in paths.items():
        rows = read_jsonl(path)
        existing[split] = {row["id"]: row for row in rows}
        if len(existing[split]) != len(rows):
            raise ValueError(f"Duplicate task IDs in {path}")
    seen = {split: set() for split in paths}
    records = list(prepare_records(environment, data_dir, source_dir, val_size))
    for split, row in tqdm(records, desc=f"Preparing {environment}", unit="task"):
        key = row["id"]
        if key in seen[split]:
            raise ValueError(f"Duplicate task id: {key}")
        seen[split].add(key)
        if key in existing[split]:
            if existing[split][key] != row:
                raise ValueError(
                    f"Existing task differs: {key}; use a new output directory"
                )
        else:
            append_jsonl(paths[split], row)
    counts = {}
    system = PROMPTS[environment]
    header = [{"role": "system", "content": system}] if system else []
    for split, path in paths.items():
        rows = list(read_jsonl(path))
        if {row["id"] for row in rows} != seen[split]:
            raise ValueError(f"Stale tasks found in {path}; use a new output directory")
        converted = [
            {
                "data_source": environment,
                "agent_name": environment,
                "prompt": header + [{"role": "user", "content": "Complete the environment task."}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "index": i,
                    "task_id": row["id"],
                    "task": row["task"],
                    "split": split,
                },
            }
            for i, row in enumerate(rows)
        ]
        target = data_dir / f"{split}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(converted), temporary)
        temporary.replace(target)
        counts[split] = len(rows)
    write_json(
        data_dir / "manifest.json",
        {"environment": environment, "counts": counts, "seed": 42},
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=list(PROMPTS))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Search-R1 canonical QA input directory; contains train.jsonl/test.jsonl",
    )
    parser.add_argument(
        "--val-size", type=int, default=128, help="number of validation questions held out from train for Search-R1"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.environment, args.data_dir, args.source_dir, args.val_size),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
