"""Typed configuration validation for RAGForgeX."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when a pipeline configuration is invalid."""


@dataclass(slots=True)
class ComponentConfig:
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataConfig:
    input_dir: Path
    parser: ComponentConfig
    chunker: ComponentConfig


@dataclass(slots=True)
class PipelineConfig:
    project: dict[str, Any]
    data: DataConfig
    embedding: ComponentConfig
    store: ComponentConfig
    retrieval: ComponentConfig
    reranker: ComponentConfig
    generator: ComponentConfig
    evaluation: dict[str, Any]


def _mapping(config: dict[str, Any], key: str, required: bool = False) -> dict[str, Any]:
    value = config.get(key, {})
    if required and not value:
        raise ConfigValidationError(f"Missing required config section `{key}`.")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigValidationError(f"Config section `{key}` must be a mapping.")
    return value


def _component(section: dict[str, Any], default_name: str) -> ComponentConfig:
    name = section.get("name", default_name)
    if not isinstance(name, str) or not name.strip():
        raise ConfigValidationError("Component `name` must be a non-empty string.")
    return ComponentConfig(
        name=name,
        options={key: value for key, value in section.items() if key != "name"},
    )


def validate_config(config: dict[str, Any]) -> PipelineConfig:
    project = _mapping(config, "project")
    data = _mapping(config, "data", required=True)
    input_dir = data.get("input_dir")
    if not isinstance(input_dir, str) or not input_dir:
        raise ConfigValidationError("`data.input_dir` must be a non-empty string.")

    parser = _component(_mapping(data, "parser"), "docling")
    chunker = _component(_mapping(data, "chunker"), "recursive")
    _validate_chunker(chunker)

    return PipelineConfig(
        project=project,
        data=DataConfig(input_dir=Path(input_dir), parser=parser, chunker=chunker),
        embedding=_component(_mapping(config, "embedding"), "sentence_transformers"),
        store=_component(_mapping(config, "store"), "faiss"),
        retrieval=_component(_mapping(config, "retrieval"), "dense"),
        reranker=_component(_mapping(config, "reranker"), "none"),
        generator=_component(_mapping(config, "generator"), "openai_compatible"),
        evaluation=_mapping(config, "evaluation"),
    )


def _validate_chunker(chunker: ComponentConfig) -> None:
    chunk_size = chunker.options.get("chunk_size") or chunker.options.get("child_size")
    overlap = chunker.options.get("chunk_overlap") or chunker.options.get("child_overlap")
    if chunk_size is not None and not isinstance(chunk_size, int):
        raise ConfigValidationError("Chunk size values must be integers.")
    if overlap is not None and not isinstance(overlap, int):
        raise ConfigValidationError("Chunk overlap values must be integers.")
    if isinstance(chunk_size, int) and isinstance(overlap, int) and overlap >= chunk_size:
        raise ConfigValidationError("Chunk overlap must be smaller than chunk size.")

