import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from stepopd.common import append_jsonl, read_jsonl, write_json
from stepopd.opd.episode import PROMPTS


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def tokenizer_fingerprint(tokenizer):
    return digest(
        {
            "vocab": tokenizer.get_vocab(),
            "template": tokenizer.chat_template,
            "special_tokens": tokenizer.special_tokens_map,
            "backend": json.loads(tokenizer.backend_tokenizer.to_str())["model"],
        }
    )


def checked_messages(result, rollout_dir, environment):
    filename = result["id"].replace("/", "_").replace(":", "_") + ".jsonl"
    events = read_jsonl(rollout_dir / "episodes" / filename)
    resets = [index for index, event in enumerate(events) if event["event"] == "reset"]
    if not resets:
        raise ValueError(f"{result['id']}: missing reset evidence")
    events = events[resets[-1] :]
    if events[-1]["event"] != "complete":
        raise ValueError(f"{result['id']}: missing complete evidence")
    complete = events[-1]
    for key in ("messages", "success", "steps", "score", "termination_reason"):
        if complete[key] != result[key]:
            raise ValueError(f"{result['id']}: complete mismatch with result on {key}")
    steps = events[1:-1]
    if not steps or len(steps) != result["steps"]:
        raise ValueError(f"{result['id']}: incomplete trajectory steps")
    messages = []
    if PROMPTS[environment] is not None:
        messages.append({"role": "system", "content": PROMPTS[environment]})
    messages.append({"role": "user", "content": events[0]["observation"]})
    for index, event in enumerate(steps, 1):
        if (
            event["event"] != "step"
            or event["step"] != index
            or event["id"] != result["id"]
        ):
            raise ValueError(f"{result['id']}: step evidence out of order or wrong ID")
        messages.append({"role": "assistant", "content": event["response"]})
        if index < len(steps):
            if event["done"]:
                raise ValueError("interaction steps after termination")
            messages.append({"role": "user", "content": event["observation"]})
    if steps[-1]["success"] is not True or steps[-1]["done"] is not True:
        raise ValueError(f"{result['id']}: not an environment-confirmed success termination")
    if messages != result["messages"]:
        raise ValueError(f"{result['id']}: messages mismatch step-by-step trajectory")
    return messages


def prepare(args, tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    from .dataset import encode_trajectory

    train_file = args.data_dir / "train.jsonl"
    tasks = read_jsonl(train_file)
    if not tasks or any(
        row["split"] != "train" or row["environment"] != args.environment
        for row in tasks
    ):
        raise ValueError("train.jsonl must be non-empty and match the target environment")
    task_ids = {row["id"] for row in tasks}
    if len(task_ids) != len(tasks):
        raise ValueError("train manifest contains duplicate IDs")
    source = json.loads((args.rollout_dir / "evaluation.json").read_text())
    if (
        source["environment"] != args.environment
        or source["model"] != args.teacher_model
        or source["manifest_sha256"]
        != hashlib.sha256(train_file.read_bytes()).hexdigest()
        or Path(source["data_dir"]).resolve() != args.data_dir.resolve()
    ):
        raise ValueError(
            "teacher source, environment, or train manifest mismatch with collection identity; val/test trajectories are forbidden"
        )
    if args.max_length < 2:
        raise ValueError("max-length must be greater than 1")
    identity = {
        "environment": args.environment,
        "teacher_model": args.teacher_model,
        "source": source,
        "rollout_dir": str(args.rollout_dir.resolve()),
        "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "max_length": args.max_length,
        "encoding": "qwen3_continuous_text_sft_v1",
    }
    manifest = args.output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("SFT inputs, tokenizer, or length settings changed; use a new output directory")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("non-empty SFT output directory missing manifest.json")
        write_json(manifest, identity)
    records_path = args.output_dir / "records.jsonl"
    previous = read_jsonl(records_path)
    completed = {row["id"]: row for row in previous}
    if len(completed) != len(previous):
        raise ValueError("duplicate SFT export record IDs")
    counts = Counter(row["status"] for row in previous)
    message_hashes = {
        row["messages_sha256"] for row in previous if row["status"] == "kept"
    }
    results = read_jsonl(args.rollout_dir / "results.jsonl")
    result_ids = {row["id"] for row in results}
    if len(result_ids) != len(results) or not completed.keys() <= result_ids:
        raise ValueError("duplicate teacher result IDs or deleted processed records")
    for result in tqdm(results, desc="SFT export", unit="episode"):
        if (
            result["sample_id"] not in task_ids
            or result["environment"] != args.environment
            or result["model"] != args.teacher_model
        ):
            raise ValueError("trajectory does not belong to the specified teacher and train split")
        result_hash = digest(result)
        if result["id"] in completed:
            record = completed[result["id"]]
            if record["result_sha256"] != result_hash:
                raise ValueError("exported teacher trajectory changed")
            if (
                record["status"] == "kept"
                and not (args.output_dir / record["shard"]).is_file()
            ):
                raise FileNotFoundError(args.output_dir / record["shard"])
            continue
        record = {"id": result["id"], "result_sha256": result_hash, "status": "failed"}
        if type(result["success"]) is not bool:
            raise ValueError("success must be a boolean returned by the environment")
        if result["success"]:
            messages = checked_messages(result, args.rollout_dir, args.environment)
            message_hash = digest(messages)
            ids, mask = encode_trajectory(tokenizer, messages)
            record.update(messages_sha256=message_hash, num_tokens=len(ids))
            if len(ids) > args.max_length:
                record["status"] = "over_length"
            elif message_hash in message_hashes:
                record["status"] = "duplicate"
            else:
                shard = Path("train") / (digest(result["id"]) + ".parquet")
                path = args.output_dir / shard
                path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "id": result["id"],
                    "sample_id": result["sample_id"],
                    "environment": args.environment,
                    "teacher_model": args.teacher_model,
                    "split": "train",
                    "success": True,
                    "enable_thinking": False,
                    "messages": messages,
                    "num_tokens": len(ids),
                    "supervised_tokens": sum(mask),
                }
                temporary = path.with_suffix(".parquet.tmp")
                pq.write_table(pa.Table.from_pylist([row]), temporary)
                temporary.replace(path)
                record.update(status="kept", shard=str(shard))
                message_hashes.add(message_hash)
        append_jsonl(records_path, record)
        counts[record["status"]] += 1
    summary = {"processed": len(results), "counts": dict(counts)}
    write_json(args.output_dir / "summary.json", summary)
    if not counts["kept"]:
        raise ValueError("no in-length successful train trajectories; cannot train")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=list(PROMPTS))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--tokenizer", required=True, help="path to the local student tokenizer")
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(prepare(args, tokenizer), ensure_ascii=False))


if __name__ == "__main__":
    main()
