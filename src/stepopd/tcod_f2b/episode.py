import asyncio
import inspect
import sys

from stepopd.opd.episode import PROMPTS, State, parse_command  # noqa: F401


async def _record_async(record, event):
    if inspect.iscoroutinefunction(record):
        await record(event)
        return

    await asyncio.to_thread(record, event)


async def run_episode(environment, name, generate, record, max_turns):
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
            if steps >= max_turns:
                state.reason = "curriculum_limit"
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
