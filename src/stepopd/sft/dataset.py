from glob import glob

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder


def encode_trajectory(tokenizer, messages):
    if not messages:
        raise ValueError("SFT trajectory cannot be empty")
    first_user = int(messages[0]["role"] == "system")
    dialogue = messages[first_user:]
    if len(dialogue) < 2 or len(dialogue) % 2:
        raise ValueError("SFT trajectory must be an optional system message followed by alternating user/assistant messages ending with assistant")
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"]
        expected = "system" if index < first_user else ("user" if (index - first_user) % 2 == 0 else "assistant")
        if (
            role != expected
            or not isinstance(content, str)
            or (role != "assistant" and not content.strip())
        ):
            raise ValueError("SFT trajectory requires strict user/assistant alternation; non-assistant messages cannot be empty")
        if "reasoning_content" in message and message["reasoning_content"]:
            raise ValueError("non-thinking SFT rejects reasoning_content")
        if any(token in content for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
            raise ValueError("SFT message content must not contain ChatML control tokens")
        if "tool_calls" in message and message["tool_calls"]:
            raise ValueError("AgenticRL SFT accepts text actions only; tool_calls are not supported")
    if tokenizer.eos_token_id != tokenizer.convert_tokens_to_ids("<|im_end|>"):
        raise ValueError("AgenticRL SFT requires the Qwen3 <|im_end|> EOS")
    builder = QwenContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})
    ids = builder.build_initial_tokens(messages[: first_user + 1])
    mask = [0] * len(ids)
    for index in range(first_user + 1, len(messages), 2):
        response = tokenizer.encode(messages[index]["content"], add_special_tokens=False) + [tokenizer.eos_token_id]
        merged = builder.merge_assistant_tokens(ids, response)
        mask, _ = builder.align_response_metadata(merged, mask)
        ids = merged.token_ids
        if index + 1 < len(messages):
            merged = builder.merge_non_assistant_tokens(messages[: index + 1], messages[: index + 2], ids)
            mask, _ = builder.align_response_metadata(merged, mask)
            ids = merged.token_ids
    if len(ids) != len(mask) or not any(mask) or mask[-1] != 1:
        raise ValueError("SFT tokens and loss mask mismatch")
    return ids, mask


class AgentSFTDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if processor is not None or config["pad_mode"] != "no_padding" or config["truncation"] != "error":
            raise ValueError("AgentSFTDataset supports text only, no_padding, and no truncation")
        patterns = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        paths = set()
        for pattern in patterns:
            matched = glob(pattern)
            if not matched:
                raise FileNotFoundError(f"no SFT parquet matched: {pattern}")
            paths.update(matched)
        self.rows = [row for path in sorted(paths) for row in pq.read_table(path).to_pylist()]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError("no successful teacher train trajectories")
        for row in self.rows:
            if row["split"] != "train" or row["success"] is not True or row["enable_thinking"] is not False:
                raise ValueError("SFT accepts only non-thinking successful teacher train trajectories")
        self.tokenizer = tokenizer
        self.max_length = int(config["max_length"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ids, mask = encode_trajectory(self.tokenizer, row["messages"])
        if len(ids) > self.max_length:
            raise ValueError(f"{row['sample_id']} exceeds max_length={self.max_length}; truncating success trajectories is forbidden")
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.long),
            "position_ids": torch.arange(len(ids), dtype=torch.long),
        }
