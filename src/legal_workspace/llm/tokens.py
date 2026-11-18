"""Offline prompt budgeting. Qwen vocabulary comes from the running local Ollama."""

import json
from dataclasses import dataclass


@dataclass
class TokenCounter:
    tokenizer: object | None = None
    method: str = "utf8_upper_bound"

    def count(self, text):
        if self.tokenizer is None:
            # Safe fallback for unknown tokenizers; explicitly NOT an exact token count.
            return len(text.encode())
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def prompt(self, messages, schema=None):
        # Chat-template overhead is provider-specific; reserve separately as well.
        count = sum(self.count(m["content"]) + self.count(m["role"]) + 32 for m in messages)
        if schema:
            count += self.count(json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
        return count


def qwen_counter(metadata):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    info = metadata.get("model_info", {})
    if info.get("tokenizer.ggml.pre") not in {"qwen2", "qwen3"}:
        raise ValueError("Unsupported tokenizer")
    tokens, merges = info["tokenizer.ggml.tokens"], info["tokenizer.ggml.merges"]
    if not isinstance(tokens, list) or len(tokens) < 256 or not isinstance(merges, list):
        raise ValueError("Missing tokenizer vocabulary")
    if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(tokens):
        raise ValueError("Unsupported byte vocabulary")
    tokenizer = Tokenizer(
        models.BPE(
            vocab={token: index for index, token in enumerate(tokens)},
            merges=[tuple(pair.split(" ", 1)) for pair in merges],
            byte_fallback=False,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(
                    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}|"
                    r" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s*[\r\n]|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    return TokenCounter(tokenizer, "ollama_qwen_bpe_with_template_reserve")
