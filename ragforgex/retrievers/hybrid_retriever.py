"""Hybrid dense plus BM25 retrieval."""

from __future__ import annotations

from typing import Any

from ragforgex.core.schema import Chunk
from ragforgex.embeddings.base import BaseEmbedding
from ragforgex.retrievers.base import BaseRetriever
from ragforgex.retrievers.bm25_retriever import BM25Retriever
from ragforgex.retrievers.dense_retriever import DenseRetriever
from ragforgex.retrievers.rrf_fusion import rrf_fusion
from ragforgex.stores.base import BaseStore


class HybridRetriever(BaseRetriever):
    def __init__(self, embedding: BaseEmbedding, store: BaseStore, chunks: list[Chunk], **_: Any) -> None:
        self.dense = DenseRetriever(embedding=embedding, store=store)
        self.bm25 = BM25Retriever(chunks=chunks)

    def retrieve(self, question: str, top_k: int = 5):
        dense_results = self.dense.retrieve(question, top_k=top_k)
        bm25_results = self.bm25.retrieve(question, top_k=top_k)
        return rrf_fusion([dense_results, bm25_results], top_k=top_k)

