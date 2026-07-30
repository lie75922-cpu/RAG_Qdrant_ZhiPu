from pathlib import Path

import pytest

from ragforgex.core.config import load_config
from ragforgex.core.config_schema import ConfigValidationError


def test_load_config_expands_default(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n  input_dir: ./examples/faiss_local_rag/docs\nmodel: ${MISSING_MODEL:-fallback}\n",
        encoding="utf-8",
    )

    assert load_config(config)["model"] == "fallback"


def test_load_config_requires_data_section(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("project:\n  name: missing_data\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="data"):
        load_config(config)


def test_load_config_rejects_invalid_overlap(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
data:
  input_dir: ./examples/faiss_local_rag/docs
  chunker:
    name: recursive
    chunk_size: 100
    chunk_overlap: 100
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="overlap"):
        load_config(config)

