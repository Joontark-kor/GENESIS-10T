"""Chat examples -> token ids with loss only on assistant replies, for any HF chat template.

The conversation is rendered once with the base model's own chat template.
Each assistant reply is located in the rendered text, and only the tokens
covering it (plus the end-of-turn token right after it) carry loss. This
works for ChatML, Llama 3, Gemma, Qwen and similar templates without
template-specific code.
"""
from __future__ import annotations

import json
from pathlib import Path

IGNORE = -100


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render(tokenizer, messages: list[dict], add_generation_prompt: bool = False,
           template_kwargs: dict | None = None) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         **(template_kwargs or {}))


def encode_example(tokenizer, messages: list[dict], max_length: int,
                   template_kwargs: dict | None = None) -> dict | None:
    """``{"input_ids", "labels"}`` or ``None`` when too long or a reply cannot be located."""
    text = render(tokenizer, messages, template_kwargs=template_kwargs)
    spans = []
    cursor = 0
    for message in messages:
        if message["role"] != "assistant":
            continue
        content = message["content"]
        start = text.find(content, cursor)
        if start < 0:
            content = content.strip()
            start = text.find(content, cursor)
        if start < 0 or not content:
            return None
        spans.append((start, start + len(content)))
        cursor = start + len(content)
    if not spans:
        return None
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) > max_length:
        return None
    labels = [IGNORE] * len(ids)
    for span_start, span_end in spans:
        last = None
        for i, (s, e) in enumerate(offsets):
            if e > s and s < span_end and e > span_start:
                labels[i] = ids[i]
                last = i
        if last is not None and last + 1 < len(ids):  # end-of-turn token
            labels[last + 1] = ids[last + 1]
    if all(label == IGNORE for label in labels):
        return None
    return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            n = len(item["input_ids"])
            ids[row, :n] = torch.tensor(item["input_ids"])
            labels[row, :n] = torch.tensor(item["labels"])
            mask[row, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
