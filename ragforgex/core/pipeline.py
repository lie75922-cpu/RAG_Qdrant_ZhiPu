"""Config-driven RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from ragforgex.core.config import load_config
from ragforgex.core.config_schema import ConfigValidationError, validate_config
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
from ragforgex.evaluators.report_writer import write_markdown_report, write_report

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
        self.typed_config = validate_config(config)
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
        input_dir = self.typed_config.data.input_dir
        parser_name, parser_kwargs = _component_config(data, "parser", "docling")
        chunker_name, chunker_kwargs = _component_config(data, "chunker", "recursive")
        self._ensure_registered(parser_name, PARSERS.names(), "parser")
        self._ensure_registered(chunker_name, CHUNKERS.names(), "chunker")
        parser = PARSERS.create(parser_name, **parser_kwargs)
        chunker = CHUNKERS.create(chunker_name, **chunker_kwargs)

        paths = sorted(input_dir.rglob("*.txt")) if input_dir.is_dir() else [input_dir]
        self.documents = [document for path in paths for document in parser.parse(path)]
        self.chunks = chunker.split(self.documents)

        embedding_name, embedding_kwargs = _component_config(self.config, "embedding", "sentence_transformers")
        self._ensure_registered(embedding_name, EMBEDDINGS.names(), "embedding")
        self.embedding = EMBEDDINGS.create(embedding_name, **embedding_kwargs)
        vectors = self.embedding.embed_texts([chunk.text for chunk in self.chunks])

        store_name, store_kwargs = _component_config(self.config, "store", "faiss")
        self._ensure_registered(store_name, STORES.names(), "store")
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
        started = perf_counter()
        contexts = self.retriever.retrieve(question, top_k=effective_top_k)
        contexts = self.reranker.rerank(question, contexts, top_k=effective_top_k)
        answer_text = self.generator.generate(question, contexts)
        answer = RAGAnswer(question=question, answer=answer_text, contexts=contexts)
        answer.metadata["evaluation"] = {
            evaluator.__class__.__name__: evaluator.evaluate(answer) for evaluator in self.evaluators
        }
        answer.metadata.update(
            {
                "project": self.config.get("project", {}).get("name", "ragforgex"),
                "retriever": retrieval_cfg.get("name", "dense"),
                "top_k": effective_top_k,
                "duration_seconds": round(perf_counter() - started, 6),
            }
        )
        return answer

    def save_index(self, path: str | Path | None = None) -> Path:
        if self.store is None:
            self.index()
        target = Path(path) if path else self.output_dir / "index"
        self.store.save(target)
        return target

    def load_index(self, path: str | Path | None = None) -> None:
        embedding_name, embedding_kwargs = _component_config(self.config, "embedding", "sentence_transformers")
        self._ensure_registered(embedding_name, EMBEDDINGS.names(), "embedding")
        self.embedding = EMBEDDINGS.create(embedding_name, **embedding_kwargs)

        store_name, store_kwargs = _component_config(self.config, "store", "faiss")
        self._ensure_registered(store_name, STORES.names(), "store")
        store_kwargs.setdefault("dimension", getattr(self.embedding, "dimension", None))
        self.store = STORES.create(store_name, **store_kwargs)
        self.store.load(Path(path) if path else self.output_dir / "index")
        self.chunks = self.store.chunks
        self._build_runtime_components()

    def write_answer_report(self, answer: RAGAnswer, output_dir: str | Path | None = None) -> dict[str, Path]:
        target_dir = Path(output_dir) if output_dir else self.output_dir
        report = self.answer_report(answer)
        json_path = target_dir / "report.json"
        markdown_path = target_dir / "report.md"
        write_report(json_path, report)
        write_markdown_report(markdown_path, report)
        return {"json": json_path, "markdown": markdown_path}

    def answer_report(self, answer: RAGAnswer) -> dict[str, Any]:
        return {
            "project": answer.metadata.get("project", self.config.get("project", {}).get("name", "ragforgex")),
            "question": answer.question,
            "answer": answer.answer,
            "retriever": answer.metadata.get("retriever", "unknown"),
            "top_k": answer.metadata.get("top_k", len(answer.contexts)),
            "duration_seconds": answer.metadata.get("duration_seconds", 0),
            "contexts": [asdict(context) for context in answer.contexts],
            "evaluation": answer.metadata.get("evaluation", {}),
        }

    @property
    def output_dir(self) -> Path:
        return Path(self.config.get("project", {}).get("output_dir", "./outputs/ragforgex"))

    def _build_runtime_components(self) -> None:
        retrieval_name, retrieval_kwargs = _component_config(self.config, "retrieval", "dense")
        self._ensure_registered(retrieval_name, RETRIEVERS.names(), "retriever")
        retrieval_kwargs.setdefault("embedding", self.embedding)
        retrieval_kwargs.setdefault("store", self.store)
        retrieval_kwargs.setdefault("chunks", self.chunks)
        self.retriever = RETRIEVERS.create(retrieval_name, **retrieval_kwargs)

        reranker_name, reranker_kwargs = _component_config(self.config, "reranker", "none")
        self._ensure_registered(reranker_name, RERANKERS.names(), "reranker")
        self.reranker = RERANKERS.create(reranker_name, **reranker_kwargs)

        generator_name, generator_kwargs = _component_config(self.config, "generator", "openai_compatible")
        self._ensure_registered(generator_name, GENERATORS.names(), "generator")
        self.generator = GENERATORS.create(generator_name, **generator_kwargs)

        evaluation = self.config.get("evaluation", {})
        if evaluation.get("enabled", False):
            for name in evaluation.get("evaluators", ["retrieval_metrics"]):
                self._ensure_registered(name, EVALUATORS.names(), "evaluator")
            self.evaluators = [
                EVALUATORS.create(name) for name in evaluation.get("evaluators", ["retrieval_metrics"])
            ]

    @staticmethod
    def _ensure_registered(name: str, available: list[str], component_type: str) -> None:
        if name not in available:
            choices = ", ".join(available) or "none"
            raise ConfigValidationError(
                f"Unknown {component_type} component `{name}`. Available options: {choices}."
            )
