# Vector Stores

`faiss` uses FAISS when installed and a NumPy fallback otherwise. `qdrant` uses `qdrant-client` when available and falls back to local search if Qdrant is unavailable.

V2 adds `chroma` and `milvus` adapters. Chroma can use `chromadb` directly. Milvus currently exposes the configuration and fallback path while deeper schema management is planned.

