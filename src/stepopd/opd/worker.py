import asyncio
from pathlib import Path

import ray
import torch
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ

from stepopd.common import append_jsonl


def _mask_segments(mask: torch.Tensor) -> list[int]:
    segments, in_segment = [], False
    for index, value in enumerate(mask.tolist()):
        if value and not in_segment:
            in_segment = True
        elif not value and in_segment:
            segments.append(index)
            in_segment = False
    if in_segment:
        segments.append(len(mask))
    return segments


class TeacherEvidenceWorker(AgentLoopWorkerTQ.__ray_metadata__.modified_class):
    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        prompt = {**prompt, "stepopd_validate": bool(trajectory["validate"])}
        await super()._run_prompt(prompt, sampling_params, trajectory, trace)

    async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
        await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
        if not self.distillation_enabled or validate:
            return
        teacher_ids = output.extra_fields["teacher_ids"].detach().cpu()
        teacher_logprobs = output.extra_fields["teacher_logprobs"].detach().cpu()
        sequence = torch.tensor(prompt_ids + response_ids, dtype=torch.int64)
        mask = torch.tensor(output.response_mask, dtype=torch.bool)
        shape_valid = teacher_ids.shape == teacher_logprobs.shape == (len(sequence), 1)
        aligned = (
            shape_valid and len(mask) == len(response_ids) and bool(torch.equal(teacher_ids[:-1, 0], sequence[1:]))
        )
        aligned = aligned and output.prompt_ids == prompt_ids and output.response_ids == response_ids
        aligned = aligned and all(value in (0, 1) for value in output.response_mask)
        record = {
            "global_step": int(sample_kwargs["global_steps"]),
            "uid": str(sample_kwargs["uid"]),
            "session_id": int(sample_kwargs["session_id"]),
            "task_id": output.extra_fields["task_id"],
            "trajectory_id": output.extra_fields["trajectory_id"],
            "teacher_shape": list(teacher_logprobs.shape),
            "teacher_ids_shape": list(teacher_ids.shape),
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "masked_environment_tokens": int((~mask).sum()),
            "scored_tokens": int(mask.sum()),
            "aligned": aligned,
            "finite": False,
            "teacher_logprob_min": None,
            "teacher_logprob_max": None,
            "teacher_logprob_mean": None,
            "turns": [],
            "token_samples": [],
        }
        if output.extra_fields.get("curriculum_k") is not None:
            record["curriculum_k"] = output.extra_fields["curriculum_k"]
        if shape_valid and len(mask) == len(response_ids) and mask.any():
            start = len(prompt_ids) - 1
            values = teacher_logprobs[start : start + len(response_ids), 0][mask]
            record["finite"] = bool(torch.isfinite(values).all())
            if record["finite"]:
                record.update(
                    teacher_logprob_min=float(values.min()),
                    teacher_logprob_max=float(values.max()),
                    teacher_logprob_mean=float(values.float().mean()),
                )
                student_logprobs = torch.tensor(output.response_logprobs, dtype=torch.float32)[mask]
                turn_start = 0
                masked_positions = mask.nonzero().flatten()
                for seg_end in _mask_segments(mask):
                    seg = (masked_positions >= turn_start) & (masked_positions < seg_end)
                    if seg.any():
                        ratio = (student_logprobs[seg] - values[seg]).float()
                        record["turns"].append(
                            {
                                "turn": len(record["turns"]) + 1,
                                "supervised_tokens": int(seg.sum()),
                                "k1_logratio_sum": float(ratio.sum()),
                                "k1_logratio_mean": float(ratio.mean()),
                            }
                        )
                    turn_start = seg_end
                selected = mask.nonzero().flatten()
                indices = torch.linspace(0, len(selected) - 1, steps=min(8, len(selected))).long()
                for index in selected[indices].tolist():
                    record["token_samples"].append(
                        {
                            "response_position": index,
                            "teacher_position": start + index,
                            "student_token_id": response_ids[index],
                            "teacher_token_id": int(teacher_ids[start + index, 0]),
                            "teacher_logprob": float(teacher_logprobs[start + index, 0]),
                        }
                    )
        await asyncio.to_thread(
            append_jsonl,
            Path(self.config.trainer.default_local_dir) / "teacher_evidence.jsonl",
            record,
        )


class TeacherEvidenceManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.config.trainer.use_v1:
            raise ValueError("TeacherEvidenceManager requires the native v1 TransferQueue trainer")
        self.agent_loop_workers_class = ray.remote(TeacherEvidenceWorker)
