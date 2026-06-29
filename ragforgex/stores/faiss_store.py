"""FAISS adapter with a NumPy fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from ragforgex.core.schema import Chunk, SearchResult
from ragforgex.stores.base import BaseStore


class FAISSStore(BaseStore):
    def __init__(self, dimension: int | None = None, **_: Any) -> None:
        self.dimension = dimension
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._index = None
        try:
            import faiss

            self._faiss = faiss
        except Exception:
            self._faiss = None

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        array = np.asarray(vectors, dtype=np.float32)
        self.dimension = int(array.shape[1])
        self._chunks.extend(chunks)
        if self._faiss is not None:
            if self._index is None:
                self._index = self._faiss.IndexFlatIP(self.dimension)
            self._index.add(array)
        self._vectors = array if self._vectors is None else np.vstack([self._vectors, array])

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if self._vectors is None or not self._chunks:
            return []
        query = np.asarray([vector], dtype=np.float32)
        if self._index is not None:
            scores, ids = self._index.search(query, top_k)
            pairs = zip(ids[0].tolist(), scores[0].tolist(), strict=False)
        else:
            scores = (self._vectors @ query[0]).tolist()
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
            pairs = [(idx, score) for idx, score in ranked]
        return [
            SearchResult(text=self._chunks[idx].text, score=float(score), metadata=self._chunks[idx].metadata)
            for idx, score in pairs
            if idx >= 0
        ]

