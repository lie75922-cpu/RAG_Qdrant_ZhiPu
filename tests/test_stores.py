from ragforgex.core.schema import Chunk
from ragforgex.stores.chroma_store import ChromaStore
from ragforgex.stores.faiss_store import FAISSStore
from ragforgex.stores.milvus_store import MilvusStore


def test_faiss_store_fallback_searches():
    store = FAISSStore(dimension=2)
    store.add([Chunk("alpha"), Chunk("beta")], [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 0.0], top_k=1)

    assert results[0].text == "alpha"


def test_faiss_store_saves_and_loads(tmp_path):
    store = FAISSStore(dimension=2)
    store.add([Chunk("alpha"), Chunk("beta")], [[1.0, 0.0], [0.0, 1.0]])
    store.save(tmp_path / "index")

    loaded = FAISSStore(dimension=2)
    loaded.load(tmp_path / "index")

    assert loaded.search([1.0, 0.0], top_k=1)[0].text == "alpha"
    assert loaded.chunks[0].text == "alpha"


def test_chroma_store_fallback_searches():
    store = ChromaStore(dimension=2)
    store.add([Chunk("alpha"), Chunk("beta")], [[1.0, 0.0], [0.0, 1.0]])

    assert store.search([1.0, 0.0], top_k=1)[0].text == "alpha"


def test_milvus_store_fallback_searches():
    store = MilvusStore(dimension=2)
    store.add([Chunk("alpha"), Chunk("beta")], [[1.0, 0.0], [0.0, 1.0]])

    assert store.search([0.0, 1.0], top_k=1)[0].text == "beta"

