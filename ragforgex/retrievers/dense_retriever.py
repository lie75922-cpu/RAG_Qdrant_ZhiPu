"""Dense vector retrieval."""

from __future__ import annotations

from typing import Any

from ragforgex.embeddings.base import BaseEmbedding
from ragforgex.retrievers.base import BaseRetriever
from ragforgex.stores.base import BaseStore


class DenseRetriever(BaseRetriever):
    def __init__(self, embedding: BaseEmbedding, store: BaseStore, **_: Any) -> None:
        self.embedding = embedding
        self.store = store

    def retrieve(self, question: str, top_k: int = 5):
        return self.store.search(self.embedding.embed_query(question), top_k=top_k)

