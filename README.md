# RAGForgeX

RAGForgeX is a modular toolkit for rapidly building, switching, and evaluating Retrieval-Augmented Generation pipelines. It provides unified interfaces for document parsers, chunkers, embedding models, vector stores, retrievers, rerankers, generators, and evaluators, allowing users to compare RAG designs through configuration files instead of rewriting code.

## Why RAGForgeX

RAG experiments often become hard to compare because each parser, vector store, retriever, and generator brings its own API shape. RAGForgeX keeps the orchestration layer small and explicit: mature open-source packages are used as optional dependencies behind adapters, while the project owns the common schema, configuration, CLI, examples, and evaluation flow.

## Features

| Layer | v0.1 | Roadmap |
| --- | --- | --- |
| Parser | Text fallback, Docling adapter, LlamaIndex adapter | Unstructured, MinerU, Marker |
| Chunker | Recursive, Token | Semantic, parent-child, Markdown |
| Embedding | Deterministic local fallback, SentenceTransformers, OpenAI-compatible | BGE, hosted embedding providers |
| Vector Store | In-memory, Qdrant, FAISS | Chroma, Milvus, Neo4j |
| Retriever | Dense, BM25, Hybrid, RRF | Query rewrite, multi-query, graph |
| Reranker | No-op | BGE reranker, CrossEncoder |
| Generator | Echo fallback, OpenAI-compatible, Zhipu, Ollama | More OpenAI-compatible providers |
| Evaluation | Retrieval metrics, optional Ragas | DeepEval, TruLens, tracing |

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e ".[dev]"
ragforgex run --config configs/faiss_dense.yaml --question "What is RAGForgeX?"
```

Start Qdrant for the Qdrant example:

```bash
docker compose up -d qdrant
ragforgex run --config configs/qdrant_dense.yaml --question "What does the toolkit compare?"
```

## Python API

```python
from ragforgex import Pipeline

pipeline = Pipeline.from_config("configs/faiss_dense.yaml")
pipeline.index()
answer = pipeline.ask("What is retrieval augmented generation?")
print(answer.answer)
print(answer.contexts)
```

## Configuration

```yaml
project:
  name: faiss_local_rag
  output_dir: ./outputs/faiss_local_rag
data:
  input_dir: ./examples/faiss_local_rag/docs
  parser:
    name: docling
  chunker:
    name: recursive
    chunk_size: 500
    chunk_overlap: 80
embedding:
  name: sentence_transformers
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
store:
  name: faiss
retrieval:
  name: dense
  top_k: 4
generator:
  name: openai_compatible
  model: gpt-4o-mini
```

Environment variables such as `${OPENAI_API_KEY}` are expanded when loading YAML.

## Architecture

```text
Parser -> Chunker -> Embedding -> Store -> Retriever -> Reranker -> Generator
                                      \-> Evaluator -> Report
```

## Installation Notes

The default test path uses lightweight fallbacks and does not require paid API keys. Optional integrations fail with clear installation messages if their dependency is missing.

Useful extras:

```bash
pip install qdrant-client faiss-cpu sentence-transformers rank-bm25 ragas
```

## Examples

- `examples/basic_qdrant_rag`: Qdrant-backed dense retrieval.
- `examples/faiss_local_rag`: local FAISS-style retrieval with a deterministic fallback.
- `examples/hybrid_retrieval`: dense plus BM25 retrieval with RRF fusion.
- `examples/ragas_evaluation`: retrieval metrics and optional Ragas evaluation wiring.

## Acknowledgements

RAGForgeX is designed to wrap, not copy, mature open-source RAG ecosystem projects such as Qdrant, FAISS, sentence-transformers, rank-bm25, Docling, LlamaIndex, and Ragas.

