# Quickstart

Install the project in editable mode and run the local FAISS-style example:

```bash
pip install -e ".[dev]"
ragforgex run --config configs/faiss_dense.yaml --question "What is RAGForgeX?"
```

Use the staged workflow when you want to reuse a local index:

```bash
ragforgex index --config configs/faiss_dense.yaml --output outputs/faiss_local_rag/index
ragforgex ask --config configs/faiss_dense.yaml --load-index --index-path outputs/faiss_local_rag/index --question "What is RAGForgeX?"
ragforgex evaluate --config configs/faiss_dense.yaml --question "What is RAGForgeX?" --output outputs/faiss_local_rag
```

