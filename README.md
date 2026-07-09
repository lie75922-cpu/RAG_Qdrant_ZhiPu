# RAGForgeX

RAGForgeX is a modular toolkit for building, switching, and evaluating Retrieval-Augmented Generation pipelines. It provides unified interfaces for parsers, chunkers, embedding models, vector stores, retrievers, rerankers, generators, and evaluators, so users can compare RAG designs through YAML configuration instead of rewriting glue code.

Read this introduction in other languages:

- [中文介绍](README.zh-CN.md)
- [한국어 소개](README.ko.md)

## What Is New In V2

- Chroma and Milvus vector store adapters with local fallback behavior.
- Semantic and parent-child chunkers for more flexible indexing layouts.
- Query rewrite retrieval for simple multi-query retrieval experiments.
- BGE reranker adapter and DeepEval evaluator adapter with graceful optional dependency handling.
- Expanded configs, docs, tests, and multilingual README introductions.

## Why RAGForgeX

RAG experiments often become hard to compare because each parser, vector store, retriever, and generator brings its own API shape. RAGForgeX keeps the orchestration layer small and explicit: mature open-source packages are used as optional dependencies behind adapters, while the project owns the common schema, configuration, CLI, examples, and evaluation flow.

## Features

| Layer | V2 support | Next |
| --- | --- | --- |
| Parser | Text fallback, Docling adapter, LlamaIndex adapter | Unstructured, MinerU, Marker |
| Chunker | Recursive, Token, Semantic, Parent-child | Markdown-aware, layout-aware |
| Embedding | Deterministic local fallback, SentenceTransformers, OpenAI-compatible | BGE, hosted embedding providers |
| Vector Store | In-memory, Qdrant, FAISS, Chroma, Milvus | Neo4j, persisted local snapshots |
| Retriever | Dense, BM25, Hybrid, RRF, Query rewrite | Multi-query, graph retrieval |
| Reranker | No-op, BGE adapter | CrossEncoder, hosted rerank APIs |
| Generator | Echo fallback, OpenAI-compatible, Zhipu, Ollama | Streaming generation |
| Evaluation | Retrieval metrics, Ragas adapter, DeepEval adapter | TruLens, Phoenix, Langfuse |

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e ".[dev]"
ragforgex run --config configs/faiss_dense.yaml --question "What is RAGForgeX?"
```

Try a V2 config:

```bash
ragforgex run --config configs/chroma_semantic.yaml --question "What does the toolkit compare?"
```

Start Qdrant for the Qdrant example:

```bash
docker compose up -d qdrant
ragforgex run --config configs/qdrant_dense.yaml --question "What does the toolkit compare?"
```

## Python API

```python
from ragforgex import Pipeline

pipeline = Pipeline.from_config("configs/chroma_semantic.yaml")
pipeline.index()
answer = pipeline.ask("What is retrieval augmented generation?")
print(answer.answer)
print(answer.contexts)
```

## Configuration

```yaml
project:
  name: chroma_semantic
data:
  input_dir: ./examples/faiss_local_rag/docs
  parser:
    name: docling
  chunker:
    name: semantic
embedding:
  name: sentence_transformers
  dimension: 384
store:
  name: chroma
  collection_name: chroma_semantic
retrieval:
  name: query_rewrite
  top_k: 4
reranker:
  name: bge
generator:
  name: openai_compatible
evaluation:
  enabled: true
  evaluators:
    - retrieval_metrics
    - deepeval
```

Environment variables such as `${OPENAI_API_KEY}` are expanded when loading YAML.

## Architecture

```text
Parser -> Chunker -> Embedding -> Store -> Retriever -> Reranker -> Generator
                                      \-> Evaluator -> Report
```

## Installation Notes

The default test path uses lightweight fallbacks and does not require paid API keys. Optional integrations fail gracefully or fall back to local behavior when their dependency is missing.

Useful extras:

```bash
pip install qdrant-client faiss-cpu sentence-transformers rank-bm25 ragas chromadb pymilvus FlagEmbedding deepeval
```

## Examples

- `examples/basic_qdrant_rag`: Qdrant-backed dense retrieval.
- `examples/faiss_local_rag`: local FAISS-style retrieval with a deterministic fallback.
- `examples/hybrid_retrieval`: dense plus BM25 retrieval with RRF fusion.
- `examples/ragas_evaluation`: retrieval metrics and optional Ragas evaluation wiring.
- `configs/chroma_semantic.yaml`: semantic chunking, Chroma adapter, query rewrite, BGE reranking, and DeepEval wiring.
- `configs/milvus_parent_child.yaml`: parent-child chunking with the Milvus adapter.

## Optimization Plan

- Persist indexes and experiment artifacts for repeatable benchmark runs.
- Add typed config validation with richer diagnostics and schema export.
- Add async ingestion, batch embedding, and streaming generation.
- Add graph extraction, Neo4j storage, and GraphRAG examples.
- Add reproducible evaluation reports across datasets, retrievers, and generators.

## Acknowledgements

RAGForgeX is designed to wrap, not copy, mature open-source RAG ecosystem projects such as Qdrant, FAISS, Chroma, Milvus, sentence-transformers, rank-bm25, Docling, LlamaIndex, Ragas, DeepEval, and FlagEmbedding.

