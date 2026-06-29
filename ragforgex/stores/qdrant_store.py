"""Qdrant store adapter with a local fallback."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ragforgex.core.schema import Chunk, SearchResult
from ragforgex.stores.faiss_store import FAISSStore


class QdrantStore(FAISSStore):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "ragforgex",
        recreate: bool = False,
        dimension: int | None = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(dimension=dimension, **kwargs)
        self.url = url
        self.collection_name = collection_name
        self.allow_fallback = allow_fallback
        self._client = None
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._qdrant_models = (Distance, VectorParams)
            self._client = QdrantClient(url=url)
            if recreate and dimension:
                self._client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )
        except Exception:
            if not allow_fallback:
                raise
            self._client = None

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if self._client is None:
            super().add(chunks, vectors)
            return
        from qdrant_client.models import PointStruct

        if self.dimension is None and vectors:
            self.dimension = len(vectors[0])
        existing = [collection.name for collection in self._client.get_collections().collections]
        if self.collection_name not in existing:
            Distance, VectorParams = self._qdrant_models
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        points = [
            PointStruct(id=str(uuid4()), vector=vector, payload={"text": chunk.text, **chunk.metadata})
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if self._client is None:
            return super().search(vector, top_k)
        hits = self._client.search(collection_name=self.collection_name, query_vector=vector, limit=top_k)
        return [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=float(hit.score),
                metadata={key: value for key, value in hit.payload.items() if key != "text"},
            )
            for hit in hits
        ]

