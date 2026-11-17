"""Deterministic vectors for storage/ranking tests; NOT a semantic-quality benchmark."""

import numpy as np


class FixedEmbedder:
    identity = "TEST-ONLY-fixed-vectors-v1"
    dimension = 5
    max_tokens = 512

    def token_count(self, text):
        return len(text.split()) + 2

    def embed(self, texts):
        vectors = []
        for text in texts:
            value = text.lower()
            vector = np.array(
                [
                    1 + sum(value.count(word) for word in ("срок", "дней", "когда", "вовремя")),
                    1 + sum(value.count(word) for word in ("неустой", "штраф", "просроч")),
                    1 + sum(value.count(word) for word in ("тайн", "секрет", "конфиден")),
                    1 + sum(value.count(word) for word in ("оплат", "платеж", "деньг")),
                    1,
                ],
                dtype=np.float32,
            )
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors)
