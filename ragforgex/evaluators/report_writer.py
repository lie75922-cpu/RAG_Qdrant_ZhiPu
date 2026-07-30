"""Evaluation report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_markdown_report(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    contexts = data.get("contexts", [])
    lines = [
        "# RAGForgeX Evaluation Report",
        "",
        f"- Project: {data.get('project', 'unknown')}",
        f"- Question: {data.get('question', '')}",
        f"- Retriever: {data.get('retriever', 'unknown')}",
        f"- Top K: {data.get('top_k', 'unknown')}",
        f"- Duration seconds: {data.get('duration_seconds', 0)}",
        "",
        "## Answer",
        "",
        str(data.get("answer", "")),
        "",
        "## Contexts",
        "",
    ]
    for index, context in enumerate(contexts, start=1):
        lines.extend(
            [
                f"### Context {index}",
                "",
                f"- Score: {context.get('score', 0)}",
                f"- Metadata: `{json.dumps(context.get('metadata', {}), ensure_ascii=False)}`",
                "",
                str(context.get("text", "")),
                "",
            ]
        )
    lines.extend(["## Evaluation", "", "```json", json.dumps(data.get("evaluation", {}), indent=2), "```", ""])
    target.write_text("\n".join(lines), encoding="utf-8")

