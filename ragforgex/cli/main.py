"""Typer CLI for RAGForgeX."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from ragforgex.core.logging import configure_logging
from ragforgex.core.pipeline import Pipeline
from ragforgex.core.registry import CHUNKERS, EMBEDDINGS, EVALUATORS, GENERATORS, PARSERS, RERANKERS, RETRIEVERS, STORES

# Import packages for registry side effects.
import ragforgex.chunkers  # noqa: F401, E402
import ragforgex.embeddings  # noqa: F401, E402
import ragforgex.evaluators  # noqa: F401, E402
import ragforgex.generators  # noqa: F401, E402
import ragforgex.parsers  # noqa: F401, E402
import ragforgex.rerankers  # noqa: F401, E402
import ragforgex.retrievers  # noqa: F401, E402
import ragforgex.stores  # noqa: F401, E402

app = typer.Typer(help="Run modular RAG pipelines from YAML configuration.")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML pipeline config."),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Question to ask after indexing."),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Override retrieval top_k."),
    report: bool = typer.Option(False, "--report", help="Write JSON and Markdown reports."),
) -> None:
    configure_logging()
    pipeline = Pipeline.from_config(config)
    chunks = pipeline.index()
    typer.echo(f"Indexed {len(chunks)} chunks.")
    if question:
        answer = pipeline.ask(question, top_k=top_k)
        typer.echo(json.dumps({"answer": answer.answer, "contexts": [asdict(c) for c in answer.contexts]}, indent=2))
        if report:
            paths = pipeline.write_answer_report(answer)
            typer.echo(f"Wrote reports to {paths['json']} and {paths['markdown']}.")


@app.command("index")
def index_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML pipeline config."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory for the persisted index."),
) -> None:
    configure_logging()
    pipeline = Pipeline.from_config(config)
    chunks = pipeline.index()
    target = pipeline.save_index(output)
    typer.echo(f"Indexed {len(chunks)} chunks and saved index to {target}.")


@app.command()
def ask(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML pipeline config."),
    question: str = typer.Option(..., "--question", "-q", help="Question to ask."),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Override retrieval top_k."),
    load_index: bool = typer.Option(False, "--load-index", help="Load a persisted local index first."),
    index_path: Optional[Path] = typer.Option(None, "--index-path", help="Persisted local index directory."),
    json_output: bool = typer.Option(False, "--json", help="Print structured JSON."),
) -> None:
    configure_logging()
    pipeline = Pipeline.from_config(config)
    if load_index:
        pipeline.load_index(index_path)
    answer = pipeline.ask(question, top_k=top_k)
    payload = pipeline.answer_report(answer)
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(answer.answer)


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to a YAML pipeline config."),
    question: str = typer.Option(..., "--question", "-q", help="Question to evaluate."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Report output directory."),
) -> None:
    configure_logging()
    pipeline = Pipeline.from_config(config)
    answer = pipeline.ask(question)
    paths = pipeline.write_answer_report(answer, output_dir=output)
    typer.echo(f"Wrote reports to {paths['json']} and {paths['markdown']}.")


@app.command()
def components(json_output: bool = typer.Option(False, "--json", help="Print structured JSON.")) -> None:
    payload = {
        "parsers": PARSERS.names(),
        "chunkers": CHUNKERS.names(),
        "embeddings": EMBEDDINGS.names(),
        "stores": STORES.names(),
        "retrievers": RETRIEVERS.names(),
        "rerankers": RERANKERS.names(),
        "generators": GENERATORS.names(),
        "evaluators": EVALUATORS.names(),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    for group, names in payload.items():
        typer.echo(f"{group}: {', '.join(names)}")


if __name__ == "__main__":
    app()
