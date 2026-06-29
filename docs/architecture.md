# Architecture

RAGForgeX uses a small orchestration layer around interchangeable components:

```text
Parser -> Chunker -> Embedding -> Store -> Retriever -> Reranker -> Generator
                                      \-> Evaluator
```

Each component is selected by name from YAML and instantiated through a registry.

