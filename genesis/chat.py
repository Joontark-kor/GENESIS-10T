"""Chat format for instruction tuning (ChatML-style, using the reserved tokens).

    <|im_start|>system\n{system}<|im_end|>\n
    <|im_start|>user\n{question}<|im_end|>\n
    <|im_start|>assistant\n{answer}<|im_end|>\n

Only assistant *content* and its closing ``<|im_end|>`` carry loss, so the model
learns to answer and to stop, not to imitate users. The same layout is shipped
as a Jinja ``chat_template`` in Hugging Face exports.
"""
from __future__ import annotations

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
ROLES = {"system", "user", "assistant"}

HF_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def validate(messages: list[dict]) -> None:
    if not messages:
        raise ValueError("empty conversation")
    for message in messages:
        if message.get("role") not in ROLES or not isinstance(message.get("content"), str):
            raise ValueError(f"bad message: {message!r}")
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("conversation has no assistant turn to learn from")


def render(messages: list[dict], add_generation_prompt: bool = False) -> str:
    text = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    return text + (f"{IM_START}assistant\n" if add_generation_prompt else "")


def encode_conversation(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids and a 0/1 loss mask of the same length."""
    validate(messages)
    start, end = tokenizer.token_id(IM_START), tokenizer.token_id(IM_END)
    newline = tokenizer.encode("\n")
    ids: list[int] = []
    mask: list[int] = []

    def add(chunk: list[int], learn: bool) -> None:
        ids.extend(chunk)
        mask.extend([int(learn)] * len(chunk))

    for message in messages:
        learn = message["role"] == "assistant"
        # Encode header + content together, exactly as the rendered string is tokenized
        # (a content starting with whitespace can merge with the header's newline).
        header = tokenizer.encode(message["role"] + "\n")
        joint = tokenizer.encode(message["role"] + "\n" + message["content"])
        shared = 0
        while shared < min(len(header), len(joint)) and header[shared] == joint[shared]:
            shared += 1
        add([start] + joint[:shared], False)
        add(joint[shared:], learn)
        add([end], learn)
        add(newline, False)
    return ids, mask


def generation_prompt(tokenizer, messages: list[dict]) -> list[int]:
    """Ids for ``messages`` followed by an open assistant turn."""
    ids: list[int] = []
    start, end = tokenizer.token_id(IM_START), tokenizer.token_id(IM_END)
    for message in messages:
        ids += [start] + tokenizer.encode(message["role"] + "\n" + message["content"])
        ids += [end] + tokenizer.encode("\n")
    return ids + [start] + tokenizer.encode("assistant\n")
