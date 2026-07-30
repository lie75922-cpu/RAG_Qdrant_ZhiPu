from typer.testing import CliRunner

from ragforgex.cli.main import app


def test_components_command_lists_groups():
    result = CliRunner().invoke(app, ["components", "--json"])

    assert result.exit_code == 0
    assert "retrievers" in result.stdout


def test_ask_command_outputs_json():
    result = CliRunner().invoke(
        app,
        [
            "ask",
            "--config",
            "configs/faiss_dense.yaml",
            "--question",
            "What is RAGForgeX?",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert "RAGForgeX" in result.stdout


def test_evaluate_command_writes_reports(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config",
            "configs/faiss_dense.yaml",
            "--question",
            "What is RAGForgeX?",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
