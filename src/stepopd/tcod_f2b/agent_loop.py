import copy
import time
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from stepopd.common import append_jsonl
from stepopd.opd.episode import make_environment

from .episode import run_episode


class CurriculumAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.enable_continuous_token:
            raise ValueError("CurriculumAgentLoop requires data.continuous_token.enable=true")
        self.environment = self.config.stepopd.environment.removesuffix("_f2b")
        if self.environment not in ("alfworld", "scienceworld", "search_r1"):
            raise ValueError(f"unknown environment: {self.config.stepopd.environment}")
        self.data_dir = self.config.stepopd.data_dir
        self.output_dir = Path(self.config.trainer.default_local_dir)
        self.retrieval_url = self.config.stepopd.retrieval_url
        self.max_tokens_per_turn = self.config.stepopd.max_tokens_per_turn
        tcod = self.config.stepopd.tcod
        self.k_start = int(tcod.k_start)
        self.eta = int(tcod.eta)
        if self.k_start < 1 or self.eta < 1:
            raise ValueError("stepopd.tcod.k_start and eta must be positive integers")

    def _curriculum_k(self, global_steps: int, env_max_steps: int) -> int:
        return min(self.k_start + global_steps // self.eta, env_max_steps)

    async def run(self, sampling_params, **kwargs):
        task_info = kwargs["extra_info"]
        environment = make_environment(self.environment, task_info["task"], self.data_dir, self.retrieval_url)
        if kwargs.get("stepopd_validate", False):
            max_turns = environment.max_steps
        else:
            max_turns = self._curriculum_k(int(kwargs["global_steps"]), environment.max_steps)
        request_id = uuid4().hex
        trace = self.output_dir / "trajectories" / f"{request_id}.jsonl"
        tokens, mask, logprobs, previous_messages = [], [], [], []
        prompt_length = 0
        generate_seconds = 0.0
        generation_count = 0
        min_global_steps, max_global_steps = None, None

        async def generate(messages):
            nonlocal tokens, mask, logprobs, previous_messages, prompt_length, generate_seconds, generation_count
            nonlocal min_global_steps, max_global_steps
            if not tokens:
                tokens = await self.loop.run_in_executor(
                    None, lambda: self.continuous_token_builder.build_initial_tokens(messages)
                )
                if len(tokens) > self.rollout_config.prompt_length:
                    raise ValueError(
                        f"Initial observation uses {len(tokens)} tokens, above prompt_length={self.rollout_config.prompt_length}"
                    )
                prompt_length = len(tokens)
            else:
                merged, new_mask, new_logprobs = await self.ct_merge_non_assistant_msg(
                    previous_messages, messages, tokens, mask, logprobs
                )
                if len(new_mask) >= self.rollout_config.response_length:
                    return None
                tokens, mask, logprobs = merged.token_ids, new_mask, new_logprobs
            remaining = self.rollout_config.response_length - len(mask)
            if remaining <= 0:
                return None
            params = {**sampling_params, "max_tokens": min(self.max_tokens_per_turn, remaining), "logprobs": True}
            started = time.monotonic()
            generated = await self.server_manager.generate(
                request_id=request_id, prompt_ids=list(tokens), sampling_params=params
            )
            generate_seconds += time.monotonic() - started
            if not generated.token_ids or generated.log_probs is None:
                raise ValueError("Rollout returned empty tokens or missing log_probs")
            if len(generated.token_ids) != len(generated.log_probs) or len(generated.token_ids) > remaining:
                raise ValueError("Rollout token/logprob lengths violate the requested budget")
            lower, upper = generated.extra_fields["min_global_steps"], generated.extra_fields["max_global_steps"]
            min_global_steps = lower if min_global_steps is None else min(min_global_steps, lower)
            max_global_steps = upper if max_global_steps is None else max(max_global_steps, upper)
            generation_count += 1
            append_jsonl(
                trace,
                {
                    "task_id": task_info["task_id"],
                    "event": "generation",
                    "turn": generation_count,
                    "prompt_tokens": len(tokens),
                    "response_start": len(mask),
                    "min_global_steps": lower,
                    "max_global_steps": upper,
                    "token_ids": generated.token_ids,
                    "log_probs": generated.log_probs,
                },
            )
            merged, mask, logprobs = await self.ct_merge_assistant_token(
                tokens, generated.token_ids, mask, logprobs, assistant_logprobs=generated.log_probs
            )
            tokens = merged.token_ids
            text = self.tokenizer.decode(generated.token_ids, skip_special_tokens=True)
            previous_messages = copy.deepcopy(messages) + [{"role": "assistant", "content": text}]
            return text

        result = await run_episode(
            environment,
            self.environment,
            generate,
            lambda event: append_jsonl(trace, {"task_id": task_info["task_id"], **event}),
            max_turns,
        )
        if not mask or mask[-1] != 1:
            raise ValueError("Trajectory must finish at a model-generated token")
        if len(tokens) != prompt_length + len(mask) or len(mask) != len(logprobs):
            raise ValueError("Trajectory tokens, response mask and rollout logprobs are misaligned")
        return AgentLoopOutput(
            prompt_ids=tokens[:prompt_length],
            response_ids=tokens[prompt_length:],
            response_mask=mask,
            response_logprobs=logprobs,
            reward_score=result["score"],
            num_turns=result["steps"] * 2,
            metrics=AgentLoopMetrics(generate_sequences=generate_seconds),
            extra_fields={
                "task_id": task_info["task_id"],
                "trajectory_id": request_id,
                "success": result["success"],
                "min_global_steps": min_global_steps,
                "max_global_steps": max_global_steps,
                "termination_reason": result["termination_reason"],
                "invalid_actions": result["invalid_actions"],
                "curriculum_k": max_turns,
            },
        )
