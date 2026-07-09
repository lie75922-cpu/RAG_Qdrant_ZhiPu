from ragforgex.core.pipeline import Pipeline


def test_pipeline_runs_from_example_config():
    pipeline = Pipeline.from_config("configs/faiss_dense.yaml")
    chunks = pipeline.index()
    answer = pipeline.ask("What is RAGForgeX?", top_k=2)

    assert chunks
    assert answer.contexts
    assert "RAGForgeX" in answer.answer


def test_pipeline_runs_v2_config():
    pipeline = Pipeline.from_config("configs/chroma_semantic.yaml")
    chunks = pipeline.index()
    answer = pipeline.ask("What does RAGForgeX compare?", top_k=2)

    assert chunks
    assert answer.contexts
    assert "evaluation" in answer.metadata

