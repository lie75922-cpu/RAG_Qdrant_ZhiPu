from ragforgex.core.registry import STORES
from ragforgex.stores.faiss_store import FAISSStore
from ragforgex.stores.qdrant_store import QdrantStore

STORES.register("faiss", FAISSStore)
STORES.register("qdrant", QdrantStore)
STORES.register("memory", FAISSStore)

