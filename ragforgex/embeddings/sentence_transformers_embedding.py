"""SentenceTransformers embedding adapter."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ragforgex.embeddings.base import BaseEmbedding


def deterministic_vector(text: str, dimension: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = np.frombuffer((digest * ((dimension // len(digest)) + 1))[:dimension], dtype=np.uint8)
    vector = (values.astype(np.float32) - 127.5) / 127.5
    norm = np.linalg.norm(vector) or 1.0
    return (vector / norm).astype(float).tolist()


class SentenceTransformersEmbedding(BaseEmbedding):
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        allow_fallback: bool = True,
        **_: Any,
    ) -> None:
        self.model_name = model
        self.dimension = dimension
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model)
            detected = getattr(self._model, "get_sentence_embedding_dimension", lambda: None)()
            if detected:
                self.dimension = int(detected)
        except Exception:
            if not allow_fallback:
                raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return vectors.tolist()
        return [deterministic_vector(text, self.dimension) for text in texts]

