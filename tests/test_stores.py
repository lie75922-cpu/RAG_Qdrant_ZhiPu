from ragforgex.core.schema import Chunk
from ragforgex.stores.faiss_store import FAISSStore


def test_faiss_store_fallback_searches():
    store = FAISSStore(dimension=2)
    store.add([Chunk("alpha"), Chunk("beta")], [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 0.0], top_k=1)

    assert results[0].text == "alpha"

