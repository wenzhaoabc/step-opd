import time
from functools import partial
from pathlib import Path

import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from verl.trainer.main_ppo import TaskRunnerV1
from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, register_trainer
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
from verl.workers.config import DistillationConfig, DistillationLossConfig

from stepopd.common import append_jsonl

from .batching import align_teacher_scores, build_teacher_batch
from .config import validate_opsd
from .losses import skill_opsd_loss
from .skills import SkillBank


@register_trainer("skill_opsd")
class SkillOPSDTrainer(PPOTrainerColocateAsync):
    def on_init_end(self):
        validate_opsd(self.config)
        cfg = self.config.stepopd.opsd
        self.skill_bank = SkillBank(
            cfg.skill_root, self.config.stepopd.environment, cfg.default_category
        )
        if self.skill_bank.sha256 != cfg.manifest_sha256:
            raise ValueError("SKILL files changed after launch snapshot")
        self.skill_builder = create_continuous_token_builder(
            self.tokenizer,
            model_family=self.config.data.continuous_token.model_family,
            model_path=self.config.actor_rollout_ref.model.path,
            chat_template_kwargs=OmegaConf.to_container(
                self.config.data.apply_chat_template_kwargs, resolve=True
            ),
        )
        actor_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
        loss_config = DistillationLossConfig(
            **OmegaConf.to_container(cfg.loss, resolve=True)
        )
        loss_container = DistillationConfig(distillation_loss=loss_config)
        self.actor_rollout_wg.set_loss_fn(
            partial(
                skill_opsd_loss,
                config=actor_config,
                distillation_config=loss_container,
            )
        )
        super().on_init_end()

    def _compute_old_log_prob(self, batch, metrics):
        batch = super()._compute_old_log_prob(batch, metrics)
        started = time.monotonic()
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=[
                "prompts",
                "responses",
                "response_mask",
                "extra_info",
                "extra_fields",
                "old_log_probs",
            ],
        )
        cfg = self.config.stepopd.opsd
        teacher, selections = build_teacher_batch(
            data,
            self.skill_bank,
            self.skill_builder.build_initial_tokens,
            max_extra_tokens=cfg.max_extra_tokens,
            max_sequence_length=cfg.max_teacher_sequence_length,
            skills_enabled=cfg.skills_enabled,
        )
        tu.assign_non_tensor(
            teacher,
            calculate_entropy=False,
            compute_loss=False,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )
        tu.assign_non_tensor_data(
            teacher,
            "global_token_num",
            [len(row) for row in teacher["input_ids"].unbind()],
        )
        output = self.actor_rollout_wg.compute_log_prob(teacher)
        fields, q_responses = align_teacher_scores(data, teacher, output["log_probs"])
        updated = tq.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id, fields=fields
        )
        updated.extra_info.update(batch.extra_info)
        sums, count, added_tokens = 0.0, 0, []
        for index, q in enumerate(q_responses):
            if batch.tags[index].get("is_padding", False):
                continue
            mask = data["response_mask"][index].bool()
            old = data["old_log_probs"][index].detach().cpu().float()
            ratio = (old - q)[mask]
            if not torch.isfinite(ratio).all():
                raise ValueError("Non-finite student/teacher log ratio")
            delta = len(teacher["prompts"][index]) - len(data["prompts"][index])
            sums += ratio.sum().item()
            count += int(mask.sum())
            added_tokens.append(delta)
            extra = data["extra_fields"][index]
            extra = getattr(extra, "data", extra)
            append_jsonl(
                Path(self.config.trainer.default_local_dir)
                / "opsd_teacher_evidence.jsonl",
                {
                    "global_step": int(self.global_steps),
                    "local_trigger_step": int(self.local_trigger_step),
                    "batch_key": str(batch.keys[index]),
                    "task_id": extra["task_id"],
                    "trajectory_id": extra["trajectory_id"],
                    "teacher": "pre_update_actor",
                    "skills_enabled": bool(cfg.skills_enabled),
                    "manifest_sha256": self.skill_bank.sha256,
                    "skill": selections[index],
                    "student_prompt_tokens": len(data["prompts"][index]),
                    "teacher_prompt_tokens": len(teacher["prompts"][index]),
                    "response_tokens": len(q),
                    "scored_tokens": int(mask.sum()),
                    "k1_logratio_mean": ratio.mean().item(),
                    "teacher_logprob_mean": q[mask].mean().item(),
                    "aligned": True,
                    "finite": True,
                    "scope": "accepted_training_batch_before_update",
                },
            )
        metrics.update(
            {
                "opsd/teacher_seconds": time.monotonic() - started,
                "opsd/k1_logratio_mean": sums / max(count, 1),
                "opsd/scored_tokens": count,
                "opsd/extra_prompt_tokens_mean": sum(added_tokens)
                / max(len(added_tokens), 1),
            }
        )
        return updated


@ray.remote
class SkillOPSDTaskRunner(TaskRunnerV1.__ray_metadata__.modified_class):
    def run(self, config):
        from stepopd.opsd import trainer as _registered  # noqa: F401

        return super().run(config)
