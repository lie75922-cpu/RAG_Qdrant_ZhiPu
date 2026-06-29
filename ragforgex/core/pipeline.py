"""Config-driven RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragforgex.core.config import load_config
from ragforgex.core.registry import (
    CHUNKERS,
    EMBEDDINGS,
    EVALUATORS,
    GENERATORS,
    PARSERS,
    RERANKERS,
    RETRIEVERS,
    STORES,
)
from ragforgex.core.schema import Chunk, Document, RAGAnswer

# Import packages for registry side effects.
import ragforgex.chunkers  # noqa: F401, E402
import ragforgex.embeddings  # noqa: F401, E402
import ragforgex.evaluators  # noqa: F401, E402
import ragforgex.generators  # noqa: F401, E402
import ragforgex.parsers  # noqa: F401, E402
import ragforgex.rerankers  # noqa: F401, E402
import ragforgex.retrievers  # noqa: F401, E402
import ragforgex.stores  # noqa: F401, E402


def _component_config(config: dict[str, Any], key: str, default_name: str) -> tuple[str, dict[str, Any]]:
    value = config.get(key, {})
    if value is None:
        value = {}
    name = value.get("name", default_name)
    kwargs = {item_key: item_value for item_key, item_value in value.items() if item_key != "name"}
    return name, kwargs


class Pipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.embedding = None
        self.store = None
        self.retriever = None
        self.reranker = None
        self.generator = None
        self.evaluators = []

    @classmethod
    def from_config(cls, path: str | Path) -> "Pipeline":
        return cls(load_config(path))

    def index(self) -> list[Chunk]:
        data = self.config.get("data", {})
        input_dir = Path(data.get("input_dir", "."))
        parser_name, parser_kwargs = _component_config(data, "parser", "docling")
        chunker_name, chunker_kwargs = _component_config(data, "chunker", "recursive")
        parser = PARSERS.create(parser_name, **parser_kwargs)
        chunker = CHUNKERS.create(chunker_name, **chunker_kwargs)

        paths = sorted(input_dir.rglob("*.txt")) if input_dir.is_dir() else [input_dir]
        self.documents = [document for path in paths for document in parser.parse(path)]
        self.chunks = chunker.split(self.documents)

        embedding_name, embedding_kwargs = _component_config(self.config, "embedding", "sentence_transformers")
        self.embedding = EMBEDDINGS.create(embedding_name, **embedding_kwargs)
        vectors = self.embedding.embed_texts([chunk.text for chunk in self.chunks])

        store_name, store_kwargs = _component_config(self.config, "store", "faiss")
        store_kwargs.setdefault("dimension", getattr(self.embedding, "dimension", None))
        self.store = STORES.create(store_name, **store_kwargs)
        self.store.add(self.chunks, vectors)
        self._build_runtime_components()
        return self.chunks

    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        if self.retriever is None:
            self.index()
        retrieval_cfg = self.config.get("retrieval", {})
        effective_top_k = top_k or int(retrieval_cfg.get("top_k", 5))
        contexts = self.retriever.retrieve(question, top_k=effective_top_k)
        contexts = self.reranker.rerank(question, contexts, top_k=effective_top_k)
        answer_text = self.generator.generate(question, contexts)
        answer = RAGAnswer(question=question, answer=answer_text, contexts=contexts)
        answer.metadata["evaluation"] = {
            evaluator.__class__.__name__: evaluator.evaluate(answer) for evaluator in self.evaluators
        }
        return answer

    def _build_runtime_components(self) -> None:
        retrieval_name, retrieval_kwargs = _component_config(self.config, "retrieval", "dense")
        retrieval_kwargs.setdefault("embedding", self.embedding)
        retrieval_kwargs.setdefault("store", self.store)
        retrieval_kwargs.setdefault("chunks", self.chunks)
        self.retriever = RETRIEVERS.create(retrieval_name, **retrieval_kwargs)

        reranker_name, reranker_kwargs = _component_config(self.config, "reranker", "none")
        self.reranker = RERANKERS.create(reranker_name, **reranker_kwargs)

        generator_name, generator_kwargs = _component_config(self.config, "generator", "openai_compatible")
        self.generator = GENERATORS.create(generator_name, **generator_kwargs)

        evaluation = self.config.get("evaluation", {})
        if evaluation.get("enabled", False):
            self.evaluators = [
                EVALUATORS.create(name) for name in evaluation.get("evaluators", ["retrieval_metrics"])
            ]
