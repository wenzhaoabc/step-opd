import asyncio
import copy
import os

import httpx
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

from stepopd.common import OpenAIClient


class TeacherClient(OpenAIClient):
    def __init__(self, args, request_model, *, transport=None):
        super().__init__(
            args.base_url,
            os.environ.get("LLM_APIKEY", ""),
            request_model,
            transport=transport,
        )
        self.timeout = httpx.Timeout(90, read=args.request_timeout, connect=15)
        self.request_retries = args.request_retries

    async def generate(self, prompt, max_tokens, sampling):
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            **sampling,
            "n": 1,
            "return_token_ids": True,
            "add_special_tokens": False,
        }
        async with httpx.AsyncClient(
            transport=self.transport, timeout=self.timeout
        ) as client:
            for attempt in range(self.request_retries + 1):
                try:
                    response = await client.post(
                        self.base_url + "/completions", headers=headers, json=payload
                    )
                    break
                except httpx.TimeoutException as error:
                    if attempt == self.request_retries:
                        raise
                    delay = 5 * 2**attempt
                    print(
                        f"[teacher HTTP] {type(error).__name__}: "
                        f"prompt={len(prompt)} tokens, "
                        f"max_tokens={max_tokens}; retrying in {delay}s "
                        f"{attempt + 1}/{self.request_retries}",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
            response.raise_for_status()
            result = response.json()
        tokens = result["choices"][0]["token_ids"]
        usage = result["usage"]
        if (
            result["choices"][0]["prompt_token_ids"] != prompt
            or not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or len(tokens) > max_tokens
            or usage["prompt_tokens"] != len(prompt)
            or usage["completion_tokens"] != len(tokens)
        ):
            raise ValueError("vLLM native token/usage mismatch with the request")
        self.calls += 1
        self.prompt_tokens += len(prompt)
        self.completion_tokens += len(tokens)
        return tokens


class TeacherRollout:
    def __init__(self, client, tokenizer, args):
        self.client, self.tokenizer, self.args = client, tokenizer, args
        self.builder = QwenContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})
        self.tokens, self.mask, self.previous_messages = [], [], []

    async def __call__(self, messages):
        if not self.tokens:
            self.tokens = self.builder.build_initial_tokens(messages)
            if len(self.tokens) > self.args.prompt_length:
                raise ValueError(f"initial prompt exceeds the OPD limit {self.args.prompt_length}")
        else:
            merged = self.builder.merge_non_assistant_tokens(self.previous_messages, messages, self.tokens)
            mask, _ = self.builder.align_response_metadata(merged, self.mask)
            if len(mask) >= self.args.response_length:
                return None
            self.tokens, self.mask = merged.token_ids, mask
        remaining = self.args.response_length - len(self.mask)
        if remaining <= 0:
            return None
        max_tokens = min(self.args.max_tokens, remaining, self.args.context_limit - len(self.tokens))
        if max_tokens < 1:
            raise ValueError("OPD max_model_len cannot fit the next token")
        generated = await self.client.generate(self.tokens, max_tokens, self.args.sampling)
        merged = self.builder.merge_assistant_tokens(self.tokens, generated)
        self.mask, _ = self.builder.align_response_metadata(merged, self.mask)
        self.tokens = merged.token_ids
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.previous_messages = copy.deepcopy(messages) + [{"role": "assistant", "content": text}]
        return text
