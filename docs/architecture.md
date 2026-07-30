# Architecture

RAGForgeX uses a small orchestration layer around interchangeable components:

```text
Parser -> Chunker -> Embedding -> Store -> Retriever -> Reranker -> Generator
                                      \-> Evaluator
```

Each component is selected by name from YAML and instantiated through a registry.

V3 adds a validation and reporting layer:

```text
YAML -> Config validation -> Pipeline -> Index persistence
                              \-> Answer report -> JSON / Markdown
```

