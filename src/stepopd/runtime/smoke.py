import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from stepopd.common import append_jsonl, write_json

GPU_CASES = ("alfworld_grpo", "scienceworld_opd", "swe_pipeline")


def build_stages(mode: str, code_root: Path, storage_root: Path, output: Path, cases: list[str]) -> list[dict]:
    python = sys.executable
    stages = []
    rollout_groups = {"swe_grpo": 2}
    if mode == "cpu":
        for name in cases:
            if name == "swe":
                stages.append(
                    {
                        "name": "swe_validate",
                        "command": [
                            python,
                            "-m",
                            "stepopd.swe.cli",
                            "validate",
                            "--metadata",
                            os.environ["SWE_METADATA"],
                            "--routes",
                            os.environ["SWE_ROUTES"],
                        ],
                        "env": {},
                        "artifacts": [],
                    }
                )
                continue
            if name not in ("alfworld", "scienceworld", "search_r1"):
                raise ValueError(f"unknown CPU environment: {name}")
            stages.append(
                {
                    "name": name,
                    "command": [
                        python,
                        "-m",
                        "stepopd.opd.check",
                        "--environment",
                        name,
                        "--data-dir",
                        str(storage_root / "envs" / name),
                        "--episodes",
                        "1",
                    ],
                    "env": {},
                    "artifacts": [],
                }
            )
        return stages
    for name in cases:
        if name in (
            "alfworld_grpo",
            "search_r1_grpo",
            "scienceworld_grpo",
            "scienceworld_opd",
            "alfworld_opd",
            "search_r1_opd",
        ):
            environment, algorithm = name.rsplit("_", 1)
            case_dir = output / name
            data_dir = storage_root / "envs" / environment
            experiment = (
                "grpo_qwen3_8b"
                if algorithm == "grpo"
                else ("opd_30b_to_4b" if environment == "search_r1" else "opd_30b_to_1p7b")
            )
            training_overrides = [
                f"{environment}/scripts@_global_={experiment}",
                f"stepopd.data_dir={json.dumps(str(data_dir))}",
                f"trainer.default_local_dir={json.dumps(str(case_dir))}",
            ]
            from stepopd.opd.train import hydra

            config_dir = Path(__file__).resolve().parents[1] / "opd" / environment / "scripts"
            with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
                config = hydra.compose(config_name="smoke", overrides=training_overrides)
            rollout_groups[name] = config.actor_rollout_ref.rollout.n
            command = [
                python,
                "-m",
                "stepopd.opd.prepare",
                "--environment",
                environment,
                "--data-dir",
                str(data_dir),
            ]
            if environment == "search_r1":
                command += ["--source-dir", os.environ["SEARCH_SOURCE_DIR"]]
            stages.append(
                {
                    "name": f"{name}_prepare",
                    "command": command,
                    "env": {},
                    "artifacts": [str(data_dir / "manifest.json")],
                }
            )
            stages.append(
                {
                    "name": f"{name}_check",
                    "command": [
                        python,
                        "-m",
                        "stepopd.opd.check",
                        "--environment",
                        environment,
                        "--data-dir",
                        str(data_dir),
                        "--episodes",
                        "1",
                    ],
                    "env": {},
                    "artifacts": [],
                }
            )
            command = [
                python,
                "-m",
                "stepopd.opd.train",
                "--config-path",
                str(code_root / "src/stepopd/opd" / environment / "scripts"),
                "--config-name",
                "smoke",
                *training_overrides,
            ]
            for step in (1, 2):
                invocation = command + [f"trainer.total_training_steps={step}"]
                if step == 2:
                    invocation += [
                        "trainer.resume_mode=resume_path",
                        f"trainer.resume_from_path={json.dumps(str(case_dir / 'global_step_1'))}",
                    ]
                stages.append(
                    {
                        "name": f"{name}_{'save' if step == 1 else 'resume'}",
                        "command": invocation,
                        "env": {},
                        "artifacts": [str(case_dir / f"global_step_{step}")],
                    }
                )
        elif name == "swe_pipeline":
            data_dir = os.environ["SWE_DATA_DIR"]
            model = os.environ["MODEL_PATH"]
            stages.append(
                {
                    "name": "swe_pipeline_check",
                    "command": [
                        python,
                        "-m",
                        "stepopd.swe.cli",
                        "validate",
                        "--metadata",
                        os.environ["SWE_METADATA"],
                        "--routes",
                        os.environ["SWE_ROUTES"],
                    ],
                    "env": {},
                    "artifacts": [],
                }
            )
            for algorithm in ("sft", "grpo"):
                case_dir = output / f"swe_{algorithm}"
                environment = {
                    "MODEL_PATH": model,
                    "DATA_DIR": data_dir,
                    "OUTPUT_DIR": str(case_dir),
                    "SWE_METADATA": os.environ["SWE_METADATA"],
                    "SWE_ROUTES": os.environ["SWE_ROUTES"],
                }
                script = code_root / "src" / "stepopd" / "swe" / "scripts" / f"train_{algorithm}.sh"
                for step in (1, 2):
                    command = [
                        "bash",
                        str(script),
                        f"trainer.total_training_steps={step}",
                        "trainer.save_freq=1",
                    ]
                    if algorithm == "sft":
                        command += [
                            "data.train_batch_size=4",
                            "data.max_length=8192",
                            "data.max_token_len_per_gpu=8192",
                            "checkpoint.save_contents=[model,optimizer,extra,hf_model]",
                            "trainer.max_ckpt_to_keep=2",
                        ]
                    else:
                        command += [
                            "data.train_batch_size=4",
                            "data.max_prompt_length=2048",
                            "data.max_response_length=6144",
                            "actor_rollout_ref.actor.ppo_mini_batch_size=4",
                            "actor_rollout_ref.rollout.n=2",
                            "actor_rollout_ref.rollout.prompt_length=2048",
                            "actor_rollout_ref.rollout.response_length=6144",
                            "actor_rollout_ref.rollout.max_model_len=8193",
                            "trainer.test_freq=1",
                            "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
                            "trainer.max_actor_ckpt_to_keep=2",
                            f"trainer.rollout_data_dir={case_dir / 'rollouts'}",
                        ]
                    if step == 1:
                        command += ["trainer.resume_mode=disable"]
                    else:
                        command += [
                            "trainer.resume_mode=resume_path",
                            f"trainer.resume_from_path={case_dir / 'global_step_1'}",
                        ]
                    stages.append(
                        {
                            "name": f"swe_{algorithm}_{'save' if step == 1 else 'resume'}",
                            "command": command,
                            "env": environment,
                            "artifacts": [str(case_dir / f"global_step_{step}")],
                        }
                    )
                if algorithm == "sft":
                    model = str(output / "swe_sft_hf")
                    stages.append(
                        {
                            "name": "swe_sft_export",
                            "command": [
                                python,
                                "-m",
                                "verl.model_merger",
                                "merge",
                                "--backend",
                                "fsdp",
                                "--local_dir",
                                str(case_dir / "global_step_2"),
                                "--target_dir",
                                model,
                            ],
                            "env": {},
                            "artifacts": [str(Path(model) / "config.json")],
                        }
                    )
        else:
            raise ValueError(f"unknown GPU smoke case: {name}")
    expanded = []
    for stage in stages:
        expanded.append(stage)
        if stage["name"].endswith("_resume"):
            name = stage["name"].removesuffix("_resume")
            algorithm = name.rsplit("_", 1)[1]
            checkpoint_dir = Path(stage["artifacts"][0]).parent
            report = output / "audits" / f"{name}.json"
            command = [
                python,
                "-m",
                "stepopd.runtime.audit",
                "--algorithm",
                algorithm,
                "--log",
                str(output / "logs" / f"{name}_save.log"),
                "--log",
                str(output / "logs" / f"{name}_resume.log"),
                "--before",
                str(checkpoint_dir / "global_step_1"),
                "--after",
                str(checkpoint_dir / "global_step_2"),
                "--output",
                str(report),
            ]
            if algorithm == "grpo":
                command += [
                    "--rollout-dir",
                    str(checkpoint_dir / "rollouts"),
                    "--group-size",
                    str(rollout_groups[name]),
                ]
            if algorithm == "opd":
                command += [
                    "--teacher-evidence",
                    str(checkpoint_dir / "teacher_evidence.jsonl"),
                ]
            expanded.append(
                {
                    "name": f"{name}_audit",
                    "command": command,
                    "env": {"CUDA_VISIBLE_DEVICES": ""},
                    "artifacts": [str(report)],
                }
            )
    return expanded


def run_stage(
    stage: dict,
    code_root: Path,
    output: Path,
    deadline: float,
    index: int,
    total: int,
    *,
    cleanup_ray: bool = False,
) -> dict:
    started = time.time()
    log_path = output / "logs" / f"{stage['name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = (
        os.environ
        | stage["env"]
        | {
            "PYTHONPATH": f"{code_root / 'src'}:{code_root / 'verl'}",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    result = {
        "name": stage["name"],
        "started_at": started,
        "command": stage["command"],
        "log": str(log_path),
        "status": "running",
        "returncode": None,
        "code_root": str(code_root),
        "env": stage["env"],
    }
    append_jsonl(output / "events.jsonl", result)
    process = None
    try:
        if started >= deadline:
            raise TimeoutError("run budget exhausted before starting the command")
        with log_path.open("a") as log:
            process = subprocess.Popen(
                stage["command"],
                cwd=code_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            last_progress = 0.0
            while process.poll() is None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                    raise TimeoutError("run budget exhausted")
                if time.time() - last_progress >= 15:
                    width = round(20 * (index - 1) / total)
                    print(
                        f"[{'=' * width}{'.' * (20 - width)}] {index}/{total} {stage['name']} "
                        f"elapsed {time.time() - started:.0f}s, budget remaining {remaining:.0f}s",
                        flush=True,
                    )
                    last_progress = time.time()
                try:
                    process.wait(timeout=min(5, remaining))
                except subprocess.TimeoutExpired:
                    pass
            result["returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"command exited with code {process.returncode}; see the stage log")
        for artifact in stage["artifacts"]:
            if not Path(artifact).exists():
                raise FileNotFoundError(f"command finished but an expected artifact is missing: {artifact}")
        result["status"] = "passed"
    except (OSError, RuntimeError, TimeoutError) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        if cleanup_ray and process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                result.update(
                    status="failed",
                    cleanup_failed=True,
                    error=f"failed to terminate stage process group (Ray cleanup): {exc}",
                )
        result["finished_at"] = time.time()
        append_jsonl(output / "events.jsonl", result)
        print(
            f"stage finished {stage['name']}: {result['status']}, exit={result['returncode']}, log={log_path}",
            flush=True,
        )
        if "error" in result:
            print(f"stage error: {result['error']}", flush=True)
        if log_path.is_file():
            with log_path.open("rb") as log:
                size = log.seek(0, os.SEEK_END)
                log.seek(max(0, size - 48 * 1024))
                tail = log.read().decode("utf-8", errors="replace")
            print(
                f"--- {stage['name']} log tail (up to 48KiB; total {size} bytes) ---\n{tail}\n--- end of log tail ---",
                flush=True,
            )
    return result


def run_smoke(
    mode: str,
    code_root: Path,
    storage_root: Path,
    output: Path,
    cases: list[str],
    max_seconds: int,
) -> dict:
    if not 0 < max_seconds <= (5400 if mode == "gpu" else 1800):
        raise ValueError("GPU smoke allows at most 5400s, CPU smoke at most 1800s")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "smoke.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["mode"] != mode or state["cases"] != cases:
            raise ValueError("resumed run must use the same mode/cases as the first run")
    else:
        state = {
            "mode": mode,
            "cases": cases,
            "started_at": time.time(),
            "deadline_at": time.time() + max_seconds,
            "stages": [],
            "execution_ok": False,
        }
    if "STEPOPD_DEADLINE_EPOCH" in os.environ:
        state["deadline_at"] = min(state["deadline_at"], float(os.environ["STEPOPD_DEADLINE_EPOCH"]))
    state["execution_ok"] = False
    write_json(state_path, state)
    if mode == "gpu":
        stage = {
            "name": "preflight",
            "command": [
                sys.executable,
                "-m",
                "stepopd.runtime.preflight",
                "--output",
                str(output / "preflight.json"),
            ],
            "env": {},
            "artifacts": [str(output / "preflight.json")],
        }
        state["preflight"] = run_stage(stage, code_root, output, state["deadline_at"], 1, 1)
        if state["preflight"]["status"] != "passed":
            write_json(state_path, state)
            return state
    stages = []
    for case in cases:
        try:
            if (
                mode == "gpu"
                and case == "swe_pipeline"
                and ("SANDBOX_TOKEN" not in os.environ or not os.environ["SANDBOX_TOKEN"])
            ):
                raise ValueError("missing container credential SANDBOX_TOKEN")
            stages += build_stages(mode, code_root, storage_root, output, [case])
        except (KeyError, ValueError) as exc:
            stages.append({"name": f"{case}_precondition", "skip_reason": f"precondition not met: {exc}"})
    completed = {item["name"]: item for item in state["stages"] if item["status"] == "passed"}
    failed_cases = set()
    gpu_contaminated = False
    with ExitStack() as resources:
        bm25_started = False
        for index, stage in enumerate(stages, 1):
            case = stage["name"].rsplit("_", 1)[0]
            if stage["name"].startswith("swe_"):
                case = "swe_pipeline"
            if "skip_reason" in stage:
                record = {
                    "name": stage["name"],
                    "status": "skipped",
                    "reason": stage["skip_reason"],
                }
            elif stage["name"] in completed and all(Path(path).exists() for path in stage["artifacts"]):
                previous = completed[stage["name"]]
                old_command = [item.replace(previous["code_root"], "<CODE_ROOT>") for item in previous["command"]]
                new_command = [item.replace(str(code_root), "<CODE_ROOT>") for item in stage["command"]]
                if old_command != new_command or previous["env"] != stage["env"]:
                    raise ValueError(f"resumed stage command differs from the previous run: {stage['name']}")
                continue
            elif time.time() >= state["deadline_at"]:
                record = {
                    "name": stage["name"],
                    "status": "skipped",
                    "reason": "budget exhausted",
                }
            elif (case in failed_cases or gpu_contaminated) and not stage["name"].endswith("_audit"):
                record = {
                    "name": stage["name"],
                    "status": "skipped",
                    "reason": "upstream failure or GPU state affected by cleanup failure",
                }
            else:
                try:
                    if (
                        mode == "gpu"
                        and case.startswith("search_r1")
                        and stage["name"].endswith(("_check", "_save", "_resume"))
                        and not bm25_started
                    ):
                        from stepopd.opd.search_r1.service import bm25_service

                        resources.enter_context(bm25_service(code_root, output, state["deadline_at"]))
                        bm25_started = True
                    record = run_stage(
                        stage,
                        code_root,
                        output,
                        state["deadline_at"],
                        index,
                        len(stages),
                        cleanup_ray=mode == "gpu" and stage["name"].endswith(("_save", "_resume")),
                    )
                except (OSError, RuntimeError, TimeoutError) as exc:
                    record = {
                        "name": stage["name"],
                        "status": "failed",
                        "error": str(exc),
                    }
                if record["status"] != "passed":
                    failed_cases.add(case)
            state["stages"] = [item for item in state["stages"] if item["name"] != stage["name"]] + [record]
            write_json(state_path, state)
            if record["status"] == "skipped":
                print(f"stage skipped {record['name']}: {record['reason']}", flush=True)
            if "cleanup_failed" in record:
                gpu_contaminated = True
    state["execution_ok"] = len(state["stages"]) == len(stages) and all(
        item["status"] == "passed" for item in state["stages"]
    )
    state["audits"] = {path.stem: json.loads(path.read_text())["status"] for path in (output / "audits").glob("*.json")}
    state["learning_verified"] = (
        state["execution_ok"]
        and bool(state["audits"])
        and all(status == "verified" for status in state["audits"].values())
    )
    state["finished_at"] = time.time()
    write_json(state_path, state)
    print(
        json.dumps(
            {
                "execution_ok": state["execution_ok"],
                "learning_verified": state["learning_verified"],
                "audits": state["audits"],
                "summary_path": str(state_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cases = args.cases or (list(GPU_CASES) if args.mode == "gpu" else ["alfworld", "scienceworld", "search_r1"])
    if args.dry_run:
        print(
            json.dumps(
                build_stages(
                    args.mode,
                    args.code_root.resolve(),
                    args.storage_root,
                    args.output_dir,
                    cases,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    max_seconds = args.max_seconds if args.max_seconds is not None else (5400 if args.mode == "gpu" else 600)
    result = run_smoke(
        args.mode,
        args.code_root.resolve(),
        args.storage_root,
        args.output_dir,
        cases,
        max_seconds,
    )
    ok = result["execution_ok"]
    print(json.dumps({"execution_ok": ok, "output_dir": str(args.output_dir)}, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
