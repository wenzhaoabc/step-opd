import copy
from collections.abc import Callable, Sequence

SKILL_HEADER = (
    "Historical task-solving skills follow. Use them to assess the next action. "
    "The current task instructions, observations, and available actions "
    "take precedence. "
    "Examples in these skills are not facts about the current environment."
)


def build_teacher_prefix(
    render: Callable,
    initial_messages: list[dict],
    student_prompt_ids: Sequence[int],
    skill_text: str,
    max_added_tokens: int,
) -> list[int]:
    if not initial_messages or not student_prompt_ids:
        raise ValueError("SKILL OPSD requires the real initial environment messages")
    if list(render(initial_messages)) != list(student_prompt_ids):
        raise ValueError(
            "Initial messages do not reproduce the student prompt token IDs"
        )
    if not skill_text.strip():
        return list(student_prompt_ids)
    messages = copy.deepcopy(initial_messages)
    block = f"{SKILL_HEADER}\n\n<teacher_skills>\n{skill_text}\n</teacher_skills>"
    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + block
    else:
        messages.insert(0, {"role": "system", "content": block})
    teacher_prompt = list(render(messages))
    added = len(teacher_prompt) - len(student_prompt_ids)
    if added > max_added_tokens:
        raise ValueError(
            f"SKILL prefix adds {added} tokens, "
            f"above max_extra_tokens={max_added_tokens}; "
            "shorten the skill or increase the teacher context budget explicitly"
        )
    return teacher_prompt


def response_score_slice(
    prompt_length: int, response_length: int, score_length: int
) -> slice:
    if (
        prompt_length < 1
        or response_length < 1
        or score_length != prompt_length + response_length
    ):
        raise ValueError("Teacher output length does not match prompt + response")
    return slice(prompt_length - 1, prompt_length - 1 + response_length)
