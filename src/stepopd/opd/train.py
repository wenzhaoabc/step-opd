import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import hydra
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig, OmegaConf

from stepopd.common import write_json


class TrainingConfigPaths(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        source = Path(__file__).resolve()
        search_path.append(provider="stepopd", path=str(source.parent))
        search_path.append(provider="stepopd-grpo", path=str(source.parent.parent / "grpo"))
        search_path.append(provider="stepopd-opsd", path=str(source.parent.parent / "opsd"))
        search_path.append(provider="verl", path=str(source.parents[3] / "verl/verl/trainer/config"))


Plugins.instance().register(TrainingConfigPaths)


def validate_resources(config: DictConfig, visible_gpus: int) -> None:
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    teacher = config.distillation
    student_gpus = config.trainer.n_gpus_per_node
    if config.stepopd.algorithm not in ("grpo", "opd", "opsd") or teacher.enabled != (
        config.stepopd.algorithm == "opd"
    ):
        raise ValueError("algorithm and distillation.enabled are inconsistent")
    if config.stepopd.algorithm == "opsd":
        from stepopd.opsd.config import validate_opsd

        validate_opsd(config)
    if config.trainer.nnodes != 1 or (teacher.enabled and teacher.nnodes != 1):
        raise ValueError("these experiments require explicit single-node GPU pools")
    pools = [(student_gpus, rollout.tensor_model_parallel_size)]
    if teacher.enabled:
        model = teacher.teacher_models.teacher_model
        pools.append((teacher.n_gpus_per_node, model.inference.tensor_model_parallel_size))
        if model.num_replicas < 1 or model.num_replicas * pools[-1][1] != teacher.n_gpus_per_node:
            raise ValueError("teacher replicas x TP must equal the teacher GPU count")
    if any(gpus < 1 or tp < 1 or gpus % tp for gpus, tp in pools):
        raise ValueError("each GPU pool must be positive and divisible by its TP")
    if sum(gpus for gpus, _ in pools) > visible_gpus:
        raise ValueError(f"config requires {sum(gpus for gpus, _ in pools)} GPUs but only {visible_gpus} are visible; adjust the config explicitly")
    batch = config.data.train_batch_size
    mini = actor.ppo_mini_batch_size
    if batch < 1 or mini < 1 or rollout.n < 1 or batch % mini:
        raise ValueError("batch, mini batch and rollout.n must be positive; batch must be divisible by mini batch")
    sp = actor.fsdp_config.ulysses_sequence_parallel_size
    if sp < 1 or student_gpus % sp:
        raise ValueError("student GPU count must be divisible by the sequence parallel size")
    dp = student_gpus // sp
    if batch * rollout.n % dp or mini * rollout.n % dp:
        raise ValueError("batch and mini batch x rollout.n must be divisible by actor DP")
    context = config.data.max_prompt_length + config.data.max_response_length
    if rollout.max_model_len < context + 1 or (
        teacher.enabled and teacher.teacher_models.teacher_model.inference.max_model_len < context + 1
    ):
        raise ValueError("max_model_len must reserve one extra token for the full context")


def run_training(config: DictConfig, launch=None) -> None:
    if launch is None and config.stepopd.algorithm == "opsd":
        raise ValueError("SKILL OPSD must be launched via python -m stepopd.opsd.train")
    agent = config.actor_rollout_ref.rollout.agent
    agent.agent_loop_config_path = str((Path(__file__).parent / agent.agent_loop_config_path).resolve())
    config.stepopd.data_dir = str(Path(config.stepopd.data_dir).expanduser().resolve())
    config.trainer.default_local_dir = str(Path(config.trainer.default_local_dir).expanduser().resolve())
    config.trainer.rollout_data_dir = str(Path(config.trainer.rollout_data_dir).expanduser().resolve())
    config.actor_rollout_ref.model.path = str(Path(config.actor_rollout_ref.model.path).expanduser().resolve())
    for split in ("train_files", "val_files"):
        files = config.data[split]
        paths = [files] if isinstance(files, str) else files
        resolved = [str(Path(value).expanduser().resolve()) for value in paths]
        for value in resolved:
            if not Path(value).is_file():
                raise FileNotFoundError(value)
        config.data[split] = resolved[0] if isinstance(files, str) else resolved
    teacher_model = None
    if config.distillation.enabled:
        teacher = config.distillation.teacher_models.teacher_model
        teacher.model_path = str(Path(teacher.model_path).expanduser().resolve())
        teacher_model = teacher.model_path
    for model in (config.actor_rollout_ref.model.path, teacher_model):
        if model is not None and not Path(model).is_dir():
            raise FileNotFoundError(model)
    if config.trainer.resume_mode == "resume_path":
        config.trainer.resume_from_path = str(Path(config.trainer.resume_from_path).expanduser().resolve())
        if not Path(config.trainer.resume_from_path).is_dir():
            raise FileNotFoundError(config.trainer.resume_from_path)
    import torch

    validate_resources(config, torch.cuda.device_count())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "stepopd.runtime.preflight",
            "--output",
            str(Path(config.trainer.default_local_dir) / "preflight.json"),
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )

    local_search = config.stepopd.environment == "search_r1" and config.stepopd.retrieval_url in (
        "http://127.0.0.1:8000/retrieve",
        "http://localhost:8000/retrieve",
    )
    with ExitStack() as resources:
        if local_search:
            from stepopd.opd.search_r1.service import bm25_service

            resources.enter_context(
                bm25_service(
                    Path(__file__).resolve().parents[3], Path(config.trainer.default_local_dir), time.time() + 1800
                )
            )
        output = Path(config.trainer.default_local_dir)
        launch_dir = output / "launches" / uuid4().hex
        write_json(launch_dir / "config.json", OmegaConf.to_container(config, resolve=True))
        record = {
            "argv": sys.argv,
            "environment": config.stepopd.environment,
            "algorithm": config.stepopd.algorithm,
            "student": config.actor_rollout_ref.model.path,
            "teacher": teacher_model,
            "config": str(launch_dir / "config.json"),
        }
        write_json(launch_dir / "launch.json", record)
        write_json(output / "launch.json", record)
        if launch is None:
            from verl.trainer.main_ppo import main as launch
        launch(config)


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig) -> None:
    run_training(config)


if __name__ == "__main__":
    main()
