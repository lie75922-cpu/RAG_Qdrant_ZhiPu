# Roadmap

## v0.1

- Unified schemas, config loader, registry, pipeline, and CLI.
- Parser, chunker, embedding, store, retriever, generator, and evaluator adapters.
- Qdrant, FAISS-style local retrieval, dense/BM25/hybrid retrieval, and Ragas wiring.

## v0.2

- Chroma and Milvus adapters.
- Semantic and parent-child chunkers.
- BGE reranker and DeepEval evaluator.
- Query rewrite retriever.

## v0.3

- LightRAG adapter.
- Neo4j-backed graph retrieval.
- GraphRAG examples.

## Optimization Plan

- Add persisted local indexes for FAISS, Chroma, and Qdrant snapshots.
- Add benchmark datasets and comparable experiment reports.
- Add streaming generation and async ingestion.
- Add typed config validation with richer error messages.
- Add graph extraction and Neo4j examples.

