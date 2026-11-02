import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from statistics import stdev

from stepopd.common import (
    OpenAIClient,
    append_jsonl,
    read_jsonl,
    write_json,
)

from .episode import PROMPTS, make_environment, run_episode


async def evaluate(args, *, checkpoint_identity: dict | None = None, request_model: str | None = None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "environment": args.environment,
        "model": args.model,
        "checkpoint": checkpoint_identity,
        "base_url": args.base_url,
        "manifest_sha256": hashlib.sha256(args.data_file.read_bytes()).hexdigest(),
        "data_dir": str(args.data_dir.resolve()),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "context_limit": args.context_limit,
        "retrieval_url": args.retrieval_url,
        "evals_per_sample": args.evals_per_sample,
    }
    identity_path = args.output_dir / "evaluation.json"
    if identity_path.exists():
        stored = json.loads(identity_path.read_text())
        if stored != identity:
            raise ValueError(
                "Evaluation inputs/model/settings changed; use a new output directory"
            )
    else:
        if (args.output_dir / "results.jsonl").exists():
            raise ValueError(
                "Existing results have no evaluation identity; use a new output directory"
            )
        write_json(identity_path, identity)
    results_path = args.output_dir / "results.jsonl"
    old_results = read_jsonl(results_path)
    completed = {item["id"] for item in old_results}
    if len(completed) != len(old_results):
        raise ValueError("Duplicate completed episode IDs")
    all_tasks = list(read_jsonl(args.data_file))
    if not all_tasks:
        raise ValueError("Evaluation manifest is empty")
    selected = (
        all_tasks[: args.max_episodes] if args.max_episodes is not None else all_tasks
    )
    if len({row["id"] for row in all_tasks}) != len(all_tasks):
        raise ValueError("Duplicate manifest episode IDs")
    runs = [
        (row, run, row["id"] if args.evals_per_sample == 1 else f"{row['id']}::{run}")
        for row in selected
        for run in range(args.evals_per_sample)
    ]
    client = OpenAIClient(
        args.base_url,
        os.environ.get("LLM_APIKEY", ""),
        args.model if request_model is None else request_model,
    )
    started = time.monotonic()
    pending = sum(result_id not in completed for _, _, result_id in runs)
    processed = 0
    print(
        f"Evaluation: {pending} pending episodes "
        f"(samples={len(selected)}×n={args.evals_per_sample}), concurrency={args.concurrency}; "
        "duration depends on environment steps and API latency",
        flush=True,
    )
    for row in selected:
        if row["environment"] != args.environment:
            raise ValueError(f"Manifest environment differs: {row['environment']}")

    async def run_one(row, run, result_id):
        episode_path = (
            args.output_dir
            / "episodes"
            / (result_id.replace("/", "_").replace(":", "_") + ".jsonl")
        )
        environment = make_environment(
            args.environment, row["task"], args.data_dir, args.retrieval_url
        )
        last_prompt_tokens = 0
        max_turn_growth = 0

        async def generate(messages):
            nonlocal last_prompt_tokens, max_turn_growth
            if last_prompt_tokens and (
                last_prompt_tokens + max(max_turn_growth, args.max_tokens)
                >= args.context_limit
            ):
                return None
            response = await client.complete(
                messages, max_tokens=args.max_tokens, temperature=args.temperature
            )
            prompt_tokens = int(response["usage"]["prompt_tokens"])
            if last_prompt_tokens:
                max_turn_growth = max(
                    max_turn_growth, prompt_tokens - last_prompt_tokens
                )
            last_prompt_tokens = prompt_tokens
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Model returned empty/non-text content")
            return content

        result = await run_episode(
            environment,
            args.environment,
            generate,
            lambda event: append_jsonl(episode_path, {"id": result_id, "run": run, **event}),
        )
        return {
            "id": result_id,
            "sample_id": row["id"],
            "run": run,
            "environment": args.environment,
            "model": args.model,
            **result,
        }

    async def run_indexed(task):
        row, run, result_id = task
        nonlocal processed
        async with semaphore:
            result = await run_one(row, run, result_id)
        append_jsonl(results_path, result)
        processed += 1
        elapsed = time.monotonic() - started
        eta = elapsed / processed * (pending - processed)
        print(
            f"[{processed}/{pending}] id={result_id}, score={result['score']:.3f}, "
            f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s",
            flush=True,
        )

    semaphore = asyncio.Semaphore(args.concurrency)
    todo = [task for task in runs if task[2] not in completed]
    await asyncio.gather(*(run_indexed(task) for task in todo))
    result_ids = {result_id for _, _, result_id in runs}
    results = [row for row in read_jsonl(results_path) if row["id"] in result_ids]
    total_steps = sum(row["steps"] for row in results)
    invalid_actions = sum(row["invalid_actions"] for row in results)
    by_sample: dict[str, list[dict]] = {}
    for row in results:
        by_sample.setdefault(row["sample_id"], []).append(row)

    def per_sample_std(key: str) -> float:
        stds = []
        for rows in by_sample.values():
            values = [row[key] for row in rows]
            stds.append(stdev(values) if len(values) > 1 else 0.0)
        return sum(stds) / len(stds)

    summary = {
        "environment": args.environment,
        "model": args.model,
        "samples": len(by_sample),
        "episodes": len(results),
        "evals_per_sample": args.evals_per_sample,
        "mean_score": sum(row["score"] for row in results) / len(results),
        "score_std_mean": per_sample_std("score"),
        "success_rate": sum(row["success"] for row in results) / len(results),
        "success_std_mean": per_sample_std("success"),
        "mean_steps": total_steps / len(results),
        "total_steps": total_steps,
        "invalid_actions": invalid_actions,
        "invalid_action_rate": invalid_actions / total_steps if total_steps else 0.0,
    }
    if args.environment == "search_r1":
        summary["mean_searches"] = sum(row["searches"] for row in results) / len(
            results
        )
    elif args.environment == "scienceworld":
        summary["mean_final_score"] = sum(row["final_score"] for row in results) / len(
            results
        )
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=list(PROMPTS))
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-episodes", type=int, default=1,
                        help="max number of samples selected (each re-evaluated --evals-per-sample times)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--context-limit",
        type=int,
        default=16384,
        help="total context limit of the model server; episodes end at context_limit when prompts approach it",
    )
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="concurrent episodes; 1 is serial (local vLLM can reach 64-128, depending on environment)",
    )
    parser.add_argument(
        "--evals-per-sample",
        type=int,
        default=1,
        help="evaluations per sample; >1 saves each run separately and computes per-sample variance",
    )
    args = parser.parse_args()
    if (
        args.max_episodes < 1
        or args.max_tokens < 1
        or args.concurrency < 1
        or args.evals_per_sample < 1
    ):
        parser.error("max-episodes, max-tokens, concurrency and evals-per-sample must be positive")
    if args.context_limit <= args.max_tokens:
        parser.error("context-limit must exceed max-tokens")
    print(json.dumps(asyncio.run(evaluate(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
