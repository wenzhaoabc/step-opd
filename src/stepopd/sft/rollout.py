import asyncio
import hashlib
import json
import sys
import time

from stepopd.common import (
    append_jsonl,
    read_jsonl,
    write_json,
)
from stepopd.opd.episode import make_environment, run_episode

from .generation import TeacherClient, TeacherRollout
from .prepare import checked_messages


def collection_state(output_dir, tasks, max_attempts):
    if max_attempts < 1 or not tasks:
        raise ValueError("max_attempts must be positive; train manifest must be non-empty")
    state = {row["id"]: {"runs": [], "success": False, "completed": 0} for row in tasks}
    if len(state) != len(tasks):
        raise ValueError("train manifest contains duplicate IDs")
    filenames = [
        f"{row['id']}::{run}".replace("/", "_").replace(":", "_") for row in tasks for run in range(max_attempts)
    ]
    if len(set(filenames)) != len(filenames):
        raise ValueError("trajectory filename collision for manifest IDs")
    attempts = read_jsonl(output_dir / "attempts.jsonl")
    results = read_jsonl(output_dir / "results.jsonl")
    by_id = {row["id"]: row for row in results}
    if len(by_id) != len(results):
        raise ValueError("Duplicate completed episode IDs")
    started = set()
    task_environments = {row["id"]: row["environment"] for row in tasks}
    for attempt in attempts:
        sample_id, run = attempt["sample_id"], attempt["run"]
        if sample_id not in state or type(run) is not int:
            raise ValueError("collection attempt not in train manifest")
        item = state[sample_id]
        result_id = f"{sample_id}::{run}"
        if (
            attempt["id"] != result_id
            or run != len(item["runs"])
            or run >= max_attempts
            or item["success"]
            or attempt["environment"] != task_environments[sample_id]
        ):
            raise ValueError("collection attempt duplicated, out of range, or after success")
        started.add(result_id)
        item["runs"].append(run)
        filename = result_id.replace("/", "_").replace(":", "_") + ".jsonl"
        events = read_jsonl(output_dir / "episodes" / filename)
        if any(event["id"] != result_id or event["run"] != run for event in events):
            raise ValueError("trajectory ID/run mismatch with attempt record")
        complete = [event for event in events if event["event"] == "complete"]
        if len(complete) > 1:
            raise ValueError("multiple complete events in one attempt")
        if not complete:
            if result_id in by_id:
                raise ValueError("saved result missing complete evidence")
            continue
        result = dict(attempt)
        result.update({key: value for key, value in complete[0].items() if key != "event"})
        if type(result["success"]) is not bool:
            raise ValueError("success must be a boolean returned by the environment")
        if result["success"]:
            checked_messages(result, output_dir, result["environment"])
        if result_id in by_id:
            if by_id[result_id] != result:
                raise ValueError("complete mismatch with saved result")
        else:
            append_jsonl(output_dir / "results.jsonl", result)
        item["completed"] += 1
        item["success"] = result["success"]
    if not by_id.keys() <= started:
        raise ValueError("results missing a started attempt record")
    expected = {result_id.replace("/", "_").replace(":", "_") + ".jsonl" for result_id in started}
    if any(path.name not in expected for path in (output_dir / "episodes").glob("*.jsonl")):
        raise ValueError("trajectory not belonging to train manifest")
    return state


async def collect_trajectories(
    args, *, tokenizer, checkpoint_identity, request_model, target_successes
):
    if target_successes < 1:
        raise ValueError("target_successes must be positive")
    tasks = read_jsonl(args.data_file)
    identity = {
        "environment": args.environment,
        "model": args.model,
        "checkpoint": checkpoint_identity,
        "base_url": args.base_url,
        "manifest_sha256": hashlib.sha256(args.data_file.read_bytes()).hexdigest(),
        "data_dir": str(args.data_dir.resolve()),
        "max_attempts": args.max_attempts,
        "sampling": args.sampling,
        "prompt_length": args.prompt_length,
        "response_length": args.response_length,
        "max_tokens": args.max_tokens,
        "context_limit": args.context_limit,
        "retrieval_url": args.retrieval_url,
    }
    identity_path = args.output_dir / "evaluation.json"
    if identity_path.exists():
        stored = json.loads(identity_path.read_text())
        if stored != identity:
            raise ValueError("collection identity/settings changed; use a new output directory")
    else:
        if any((args.output_dir / name).exists() for name in ("attempts.jsonl", "results.jsonl", "episodes")):
            raise ValueError("non-empty collection directory missing evaluation.json")
        write_json(identity_path, identity)
    state = collection_state(args.output_dir, tasks, args.max_attempts)
    started = time.monotonic()
    pending = sum(not item["success"] and len(item["runs"]) < args.max_attempts for item in state.values())
    initial_successes = sum(item["success"] for item in state.values())
    finished = 0

    def save_summary():
        success = sum(item["success"] for item in state.values())
        exhausted = sum(not item["success"] and len(item["runs"]) == args.max_attempts for item in state.values())
        summary = {
            "environment": args.environment,
            "model": args.model,
            "samples": len(tasks),
            "successful_samples": success,
            "exhausted_samples": exhausted,
            "pending_samples": len(tasks) - success - exhausted,
            "success_coverage": success / len(tasks),
            "started_attempts": sum(len(item["runs"]) for item in state.values()),
            "completed_attempts": sum(item["completed"] for item in state.values()),
            "max_attempts": args.max_attempts,
            "target_successes": target_successes,
            "target_reached": success >= target_successes,
        }
        write_json(args.output_dir / "summary.json", summary)
        return summary

    summary = save_summary()
    if not pending or summary["target_reached"]:
        return summary
    client = TeacherClient(args, request_model)
    semaphore = asyncio.Semaphore(args.concurrency)
    stopping = asyncio.Event()
    print(
        f"Teacher collection: {pending}/{len(tasks)} pending samples; "
        f"successes={initial_successes}/{target_successes}; "
        f"at most {args.max_attempts} attempts/sample, concurrency={args.concurrency}; "
        "ETA will be estimated after the first sample",
        flush=True,
    )

    async def collect_sample(row):
        nonlocal finished
        async with semaphore:
            if stopping.is_set() or summary["target_reached"]:
                return
            item = state[row["id"]]
            for run in range(len(item["runs"]), args.max_attempts):
                if stopping.is_set() or summary["target_reached"]:
                    break
                metadata = {
                    "id": f"{row['id']}::{run}",
                    "sample_id": row["id"],
                    "run": run,
                    "environment": args.environment,
                    "model": args.model,
                }
                append_jsonl(args.output_dir / "attempts.jsonl", metadata)
                item["runs"].append(run)
                summary["started_attempts"] += 1
                write_json(args.output_dir / "summary.json", summary)
                filename = metadata["id"].replace("/", "_").replace(":", "_")
                episode_path = args.output_dir / "episodes" / f"{filename}.jsonl"
                recording_error = None

                async def record(event):
                    nonlocal recording_error
                    if recording_error is not None:
                        raise recording_error
                    try:
                        await asyncio.to_thread(
                            append_jsonl,
                            episode_path,
                            {"id": metadata["id"], "run": run, **event},
                        )
                    except Exception as error:
                        recording_error = error
                        raise

                teacher = TeacherRollout(client, tokenizer, args)

                try:
                    environment = make_environment(args.environment, row["task"], args.data_dir, args.retrieval_url)
                    result = await run_episode(environment, args.environment, teacher, record)
                    append_jsonl(args.output_dir / "results.jsonl", {**metadata, **result})
                    item["completed"] += 1
                    item["success"] = result["success"]
                    summary["completed_attempts"] += 1
                    if item["success"]:
                        summary["successful_samples"] += 1
                        summary["success_coverage"] = summary["successful_samples"] / len(tasks)
                        summary["pending_samples"] -= 1
                    elif run == args.max_attempts - 1:
                        summary["exhausted_samples"] += 1
                        summary["pending_samples"] -= 1
                    summary["target_reached"] = summary["successful_samples"] >= target_successes
                    write_json(args.output_dir / "summary.json", summary)
                    completed = finished + int(item["success"] or run == args.max_attempts - 1)
                    elapsed = time.monotonic() - started
                    estimates = []
                    if completed:
                        estimates.append(elapsed / completed * (pending - completed))
                    new_successes = summary["successful_samples"] - initial_successes
                    if new_successes:
                        remaining = max(0, target_successes - summary["successful_samples"])
                        estimates.append(elapsed / new_successes * remaining)
                    eta = f"{min(estimates):.0f}s" if estimates else "pending first completed sample"
                    if summary["target_reached"]:
                        eta = "target reached; waiting for in-flight episodes"
                    print(
                        f"[{completed}/{pending} samples] sample={row['id']}, "
                        f"attempt={run + 1}/{args.max_attempts}, "
                        f"success={item['success']}, "
                        f"successes={summary['successful_samples']}/{target_successes}, "
                        f"elapsed={elapsed:.0f}s, ETA={eta}",
                        flush=True,
                    )
                    if item["success"]:
                        break
                except BaseException as error:
                    stopping.set()
                    try:
                        append_jsonl(
                            args.output_dir / "errors.jsonl",
                            {
                                **metadata,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                        )
                    except Exception as record_error:
                        print(
                            f"[collect] failed to write errors.jsonl; keeping original exception: {record_error!r}",
                            file=sys.stderr,
                        )
                    raise
            finished += int(item["success"] or len(item["runs"]) == args.max_attempts)

    workers = [
        asyncio.create_task(collect_sample(row))
        for row in tasks
        if not state[row["id"]]["success"] and len(state[row["id"]]["runs"]) < args.max_attempts
    ]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    finally:
        in_flight = sys.exc_info()[0] is not None
        try:
            summary = save_summary()
        except Exception as save_error:
            if not in_flight:
                raise
            print(
                f"[collect] failed to save summary; keeping original exception: {save_error!r}",
                file=sys.stderr,
            )
    reason = "success target reached" if summary["target_reached"] else "candidate tasks exhausted"
    print(
        f"Teacher collection stopped: {reason}; "
        f"successes {summary['successful_samples']}/{target_successes}, "
        f"pending samples {summary['pending_samples']}.",
        flush=True,
    )
    return summary
