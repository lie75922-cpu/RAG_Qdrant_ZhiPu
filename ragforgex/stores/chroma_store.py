"""Chroma vector store adapter with a local fallback."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ragforgex.core.schema import Chunk, SearchResult
from ragforgex.stores.faiss_store import FAISSStore


class ChromaStore(FAISSStore):
    def __init__(
        self,
        collection_name: str = "ragforgex",
        persist_directory: str | None = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._collection = None
        try:
            import chromadb

            client = (
                chromadb.PersistentClient(path=persist_directory)
                if persist_directory
                else chromadb.Client()
            )
            self._collection = client.get_or_create_collection(collection_name)
        except Exception:
            if not allow_fallback:
                raise

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if self._collection is None:
            super().add(chunks, vectors)
            return
        ids = [str(uuid4()) for _ in chunks]
        self._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if self._collection is None:
            return super().search(vector, top_k)
        result = self._collection.query(query_embeddings=[vector], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            SearchResult(text=text, score=1.0 / (1.0 + float(distance)), metadata=metadata or {})
            for text, metadata, distance in zip(documents, metadatas, distances, strict=False)
        ]

