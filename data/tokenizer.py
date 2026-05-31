"""BPE tokenizer wrapper.

Uses tiktoken's GPT-2 encoding (50257 tokens). Our ModelConfig.vocab_size is
50304 = 50257 rounded up to a multiple of 64 — the extra rows are never emitted
by the tokenizer but make the embedding matmul tile cleanly on the GPU. The
rounding is a real throughput trick, not cosmetic.

We append an explicit end-of-document token (GPT-2's <|endoftext|>, id 50256)
between documents so the model learns document boundaries.
"""

import tiktoken

EOT_TOKEN = 50256  # GPT-2 <|endoftext|>


class Tokenizer:
    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")
        self.eot = EOT_TOKEN

    @property
    def n_vocab(self) -> int:
        return self._enc.n_vocab  # 50257

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        ids = self._enc.encode_ordinary(text)  # ignores special tokens in text
        if add_eot:
            ids.append(self.eot)
        return ids

    def encode_batch(self, texts: list[str], add_eot: bool = True) -> list[int]:
        out: list[int] = []
        for t in texts:
            out.extend(self.encode(t, add_eot=add_eot))
        return out

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)
