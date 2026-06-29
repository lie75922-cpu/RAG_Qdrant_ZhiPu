from pathlib import Path

from ragforgex.core.config import load_config


def test_load_config_expands_default(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("model: ${MISSING_MODEL:-fallback}\n", encoding="utf-8")

    assert load_config(config)["model"] == "fallback"

