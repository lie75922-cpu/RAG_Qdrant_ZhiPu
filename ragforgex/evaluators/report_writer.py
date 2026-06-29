"""Evaluation report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")

