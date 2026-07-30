"""FAISS adapter with a NumPy fallback."""

from __future__ import annotations

import json
from pathlib import Path
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

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

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

    def save(self, path: str | Path) -> None:
        if self._vectors is None:
            raise ValueError("Cannot save an empty vector store.")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "vectors.npy", self._vectors)
        payload = {
            "dimension": self.dimension,
            "chunks": [{"text": chunk.text, "metadata": chunk.metadata} for chunk in self._chunks],
        }
        (target / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        target = Path(path)
        vectors_path = target / "vectors.npy"
        chunks_path = target / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"No persisted FAISSStore index found in `{target}`.")
        self._vectors = np.load(vectors_path).astype(np.float32)
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.dimension = int(payload["dimension"])
        self._chunks = [
            Chunk(text=item["text"], metadata=item.get("metadata", {}))
            for item in payload.get("chunks", [])
        ]
        if self._faiss is not None:
            self._index = self._faiss.IndexFlatIP(self.dimension)
            self._index.add(self._vectors)

