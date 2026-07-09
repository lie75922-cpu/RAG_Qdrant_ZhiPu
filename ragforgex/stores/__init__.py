from ragforgex.core.registry import STORES
from ragforgex.stores.chroma_store import ChromaStore
from ragforgex.stores.faiss_store import FAISSStore
from ragforgex.stores.milvus_store import MilvusStore
from ragforgex.stores.qdrant_store import QdrantStore

STORES.register("faiss", FAISSStore)
STORES.register("qdrant", QdrantStore)
STORES.register("chroma", ChromaStore)
STORES.register("milvus", MilvusStore)
STORES.register("memory", FAISSStore)

