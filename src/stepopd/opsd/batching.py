import torch
from tensordict import TensorDict

from .prompting import build_teacher_prefix, response_score_slice


def nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _python(value):
    return getattr(value, "data", value)


def build_teacher_batch(
    data, bank, render, *, max_extra_tokens, max_sequence_length, skills_enabled=True
):
    rows = []
    selections = []
    for index in range(len(data)):
        prompt = data["prompts"][index]
        response = data["responses"][index]
        mask = data["response_mask"][index]
        info = _python(data["extra_info"][index])
        extra = _python(data["extra_fields"][index])
        selection = bank.select(info["task"])
        teacher_prompt = build_teacher_prefix(
            render,
            extra["initial_messages"],
            prompt.tolist(),
            selection.text if skills_enabled else "",
            max_extra_tokens,
        )
        prefix = torch.tensor(teacher_prompt, dtype=torch.long)
        sequence = torch.cat((prefix, response.cpu()))
        if len(sequence) > max_sequence_length:
            raise ValueError(
                f"Teacher sequence uses {len(sequence)} tokens, "
                f"limit={max_sequence_length}"
            )
        if len(mask) != len(response) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Invalid student response mask")
        if not bool(mask.any()):
            raise ValueError(
                "SKILL OPSD requires at least one generated token per trajectory"
            )
        rows.append(
            {
                "prompts": prefix,
                "responses": response.cpu(),
                "input_ids": sequence,
                "position_ids": torch.arange(len(sequence), dtype=torch.long),
                "response_mask": mask.cpu(),
                "loss_mask": mask.cpu(),
            }
        )
        selections.append(selection.evidence())
    teacher = TensorDict(
        {key: nested([row[key] for row in rows]) for key in rows[0]},
        batch_size=[len(rows)],
    )
    return teacher, selections


def align_teacher_scores(student, teacher, scores):
    if len(scores) != len(student):
        raise ValueError("Teacher returned a different batch size")
    aligned, target_ids, response_scores = [], [], []
    for index in range(len(student)):
        prompt, response = student["prompts"][index], student["responses"][index]
        teacher_prompt = teacher["prompts"][index]
        if not torch.equal(response.cpu(), teacher["responses"][index].cpu()):
            raise ValueError(
                "Teacher response IDs differ from the student's sampled IDs"
            )
        values = scores[index].detach().cpu().float()
        if values.ndim != 1:
            raise ValueError(
                "Expected one sampled-token log probability per sequence position"
            )
        teacher_slice = response_score_slice(
            len(teacher_prompt), len(response), len(values)
        )
        q_response = values[teacher_slice]
        if not torch.isfinite(q_response).all():
            raise ValueError("Non-finite teacher response log probabilities")
        sequence = torch.cat((prompt.cpu(), response.cpu()))
        student_slice = response_score_slice(len(prompt), len(response), len(sequence))
        q_aligned = torch.zeros((len(sequence), 1), dtype=torch.float32)
        q_aligned[student_slice, 0] = q_response
        ids = torch.cat((sequence[1:], torch.zeros(1, dtype=torch.long))).unsqueeze(-1)
        if not torch.equal(ids[student_slice, 0], response.cpu()):
            raise ValueError("Next-token teacher targets are misaligned")
        aligned.append(q_aligned)
        target_ids.append(ids)
        response_scores.append(q_response)
    return TensorDict(
        {"teacher_logprobs": nested(aligned), "teacher_ids": nested(target_ids)},
        batch_size=[len(student)],
    ), response_scores
