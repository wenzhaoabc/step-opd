import asyncio
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .search_r1.protocol import PROMPT as SEARCH_R1_PROMPT

PROMPTS = {
    "alfworld": None,
    "scienceworld": None,
    "search_r1": SEARCH_R1_PROMPT,
}


@dataclass
class State:
    text: str
    score: float = 0.0
    done: bool = False
    success: bool = False
    reason: str = ""
    invalid_action: bool = False
    final_score: float | None = None


def parse_command(text, tags):
    matches = list(re.finditer(r"<(action|search|answer)>(.*?)</\1>", text, re.DOTALL))
    if len(matches) != 1 or matches[0][1] not in tags or not matches[0][2].strip():
        raise ValueError(f"Output exactly one nonempty command using {', '.join(tags)} tags.")
    tag, value = matches[0][1], matches[0][2].strip()
    if (
        sum(text.count(f"<{name}>") for name in ["action", "search", "answer"]) != 1
        or sum(text.count(f"</{name}>") for name in ["action", "search", "answer"]) != 1
    ):
        raise ValueError("Repeated command tags are not allowed.")
    return tag, value


def make_environment(name, task, data_dir, retrieval_url):
    if name == "alfworld":
        from .alfworld.environment import ALFWorld

        return ALFWorld(task, Path(data_dir))
    if name == "scienceworld":
        from .scienceworld.environment import ScienceWorld

        return ScienceWorld(task)
    if name == "search_r1":
        from .search_r1.environment import SearchR1

        return SearchR1(task, retrieval_url)
    raise ValueError(f"Unknown environment: {name}")


async def _record_async(record, event):
    if inspect.iscoroutinefunction(record):
        await record(event)
        return

    await asyncio.to_thread(record, event)


async def run_episode(environment, name, generate, record):
    steps = 0
    invalid_actions = 0
    messages = []
    try:
        state = await asyncio.to_thread(environment.reset)
        system = PROMPTS[name]
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": state.text},
        ]
        await _record_async(record, {"event": "reset", "observation": state.text})
        for turn in range(environment.max_steps):
            response = await generate(messages)
            if response is None:
                state.reason = "context_limit"
                break
            messages.append({"role": "assistant", "content": response})
            state = await asyncio.to_thread(environment.step, response)
            steps = turn + 1
            invalid_actions += int(state.invalid_action)
            await _record_async(
                record,
                {
                    "event": "step",
                    "step": steps,
                    "response": response,
                    "observation": state.text,
                    "score": state.score,
                    "final_score": state.final_score,
                    "done": state.done,
                    "success": state.success,
                    "invalid_action": state.invalid_action,
                    "termination_reason": state.reason,
                },
            )
            if state.done:
                break
            if steps == environment.max_steps:
                state.reason = "step_limit"
                break
            messages.append({"role": "user", "content": state.text})
        result = {
            "score": state.score,
            "final_score": state.final_score,
            "success": state.success,
            "steps": steps,
            "invalid_actions": invalid_actions,
            "invalid_action_rate": invalid_actions / steps if steps else 0.0,
            "termination_reason": state.reason,
            "messages": messages,
        }
        if name == "search_r1":
            result["searches"] = environment.searches
        await _record_async(record, {"event": "complete", **result})
        return result
    except Exception as error:
        try:
            await _record_async(
                record,
                {
                    "event": "error",
                    "steps": steps,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except Exception as record_error:
            print(
                f"[episode] failed to record error event, preserving original exception: {record_error!r}",
                file=sys.stderr,
            )
        raise
    finally:
        in_flight = sys.exc_info()[0] is not None
        try:
            await asyncio.to_thread(environment.close)
        except Exception as close_error:
            if in_flight:
                print(
                    f"[episode] failed to close environment, preserving original exception: {close_error!r}",
                    file=sys.stderr,
                )
            else:
                raise
