"""A reversible, training-free baseline tokenizer using UTF-8 bytes.

Replace with a trained BPE/unigram tokenizer at scale. This is not a claimed
production multilingual tokenizer.
"""


class ByteTokenizer:
    vocab_size = 257
    eos_id = 256

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        payload = []
        for token in ids:
            if token == self.eos_id and skip_special:
                continue
            if not 0 <= token < 256:
                raise ValueError(f"invalid byte token: {token}")
            payload.append(token)
        return bytes(payload).decode("utf-8", errors="replace")
