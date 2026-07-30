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


def test_pipeline_persists_index_and_writes_report(tmp_path):
    pipeline = Pipeline.from_config("configs/faiss_dense.yaml")
    pipeline.index()
    index_path = pipeline.save_index(tmp_path / "index")

    loaded = Pipeline.from_config("configs/faiss_dense.yaml")
    loaded.load_index(index_path)
    answer = loaded.ask("What is RAGForgeX?", top_k=1)
    paths = loaded.write_answer_report(answer, output_dir=tmp_path / "reports")

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert answer.contexts

