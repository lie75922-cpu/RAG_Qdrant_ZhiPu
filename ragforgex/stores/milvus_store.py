"""Milvus vector store adapter with a local fallback."""

from __future__ import annotations

from typing import Any

from ragforgex.stores.faiss_store import FAISSStore


class MilvusStore(FAISSStore):
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "ragforgex",
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.uri = uri
        self.collection_name = collection_name
        self._client = None
        try:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=uri)
        except Exception:
            if not allow_fallback:
                raise

    # The v0.2 adapter intentionally falls back to FAISS-style local behavior
    # until collection schema management is expanded in v0.3.

