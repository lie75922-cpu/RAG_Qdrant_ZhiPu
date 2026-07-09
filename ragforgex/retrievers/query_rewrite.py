"""Query rewriting retriever."""

from __future__ import annotations

from typing import Any

from ragforgex.core.schema import Chunk
from ragforgex.embeddings.base import BaseEmbedding
from ragforgex.retrievers.base import BaseRetriever
from ragforgex.retrievers.dense_retriever import DenseRetriever
from ragforgex.retrievers.rrf_fusion import rrf_fusion
from ragforgex.stores.base import BaseStore


class QueryRewriteRetriever(BaseRetriever):
    def __init__(
        self,
        embedding: BaseEmbedding,
        store: BaseStore,
        chunks: list[Chunk] | None = None,
        rewrites: list[str] | None = None,
        **_: Any,
    ) -> None:
        self.dense = DenseRetriever(embedding=embedding, store=store)
        self.rewrites = rewrites or [
            "{question}",
            "Explain {question}",
            "Key facts about {question}",
        ]

    def retrieve(self, question: str, top_k: int = 5):
        result_sets = [
            self.dense.retrieve(template.format(question=question), top_k=top_k)
            for template in self.rewrites
        ]
        return rrf_fusion(result_sets, top_k=top_k)
