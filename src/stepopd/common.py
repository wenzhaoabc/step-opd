from __future__ import annotations

import fcntl
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

LEGACY_BUDGET_FIELDS = ("budget_path", "max_calls", "budget_tokens")


def without_legacy_budget_fields(stored: dict) -> dict:
    return {key: value for key, value in stored.items() if key not in LEGACY_BUDGET_FIELDS}


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end:
            stream.seek(end - 1)
            if stream.read(1) != b"\n":
                cursor = end
                while cursor:
                    start = max(0, cursor - 65536)
                    stream.seek(start)
                    tail = stream.read(cursor - start)
                    newline = tail.rfind(b"\n")
                    if newline >= 0:
                        stream.truncate(start + newline + 1)
                        break
                    cursor = start
                else:
                    stream.truncate(0)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            if line.strip():
                records.append(json.loads(line))
    return records


class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.transport = transport
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    async def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if os.environ.get("LLM_CHAT_TEMPLATE_KWARGS"):
            payload["chat_template_kwargs"] = json.loads(os.environ["LLM_CHAT_TEMPLATE_KWARGS"])
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        async with httpx.AsyncClient(
            transport=self.transport, timeout=httpx.Timeout(90, connect=15)
        ) as client:
            response = await client.post(
                self.base_url + "/chat/completions", json=payload, headers=headers
            )
            if response.status_code != 200:
                raise RuntimeError(f"Chat endpoint returned HTTP {response.status_code}")
            result = response.json()
        usage = result["usage"]
        self.calls += 1
        self.prompt_tokens += int(usage["prompt_tokens"])
        self.completion_tokens += int(usage["completion_tokens"])
        return result
