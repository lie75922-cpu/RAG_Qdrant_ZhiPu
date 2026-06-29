"""Typer CLI for RAGForgeX."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from ragforgex.core.logging import configure_logging
from ragforgex.core.pipeline import Pipeline

app = typer.Typer(help="Run modular RAG pipelines from YAML configuration.")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML pipeline config."),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Question to ask after indexing."),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Override retrieval top_k."),
) -> None:
    configure_logging()
    pipeline = Pipeline.from_config(config)
    chunks = pipeline.index()
    typer.echo(f"Indexed {len(chunks)} chunks.")
    if question:
        answer = pipeline.ask(question, top_k=top_k)
        typer.echo(json.dumps({"answer": answer.answer, "contexts": [asdict(c) for c in answer.contexts]}, indent=2))


if __name__ == "__main__":
    app()
