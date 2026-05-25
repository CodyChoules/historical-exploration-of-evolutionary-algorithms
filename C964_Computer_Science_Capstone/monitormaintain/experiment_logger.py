"""Experiment logging utilities for archiving outputs by experiment id."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _results_root() -> Path:
    return _project_root() / "_Experimental_Results"


def log_experiment_bundle(
    *,
    experiment_id: str,
    experiment_name: str,
    info_panel_lines: Iterable[str],
    artifact_paths: Iterable[Path | str],
) -> Path:
    """
    Create _Experimental_Results/<experiment_id>/ and copy run artifacts into it.

    Also writes info_panel.txt with a plain-text snapshot of panel lines.
    """
    run_dir = _results_root() / experiment_id
    viz_dir = run_dir / "visualizations"
    run_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for raw in artifact_paths:
        path = Path(raw)
        if not path.is_file():
            continue
        dest = viz_dir / path.name
        shutil.copy2(path, dest)
        copied.append(dest.name)

    info_lines = [str(line) for line in info_panel_lines]
    copied_lines = [f"- {name}" for name in copied] if copied else ["- none"]
    info_text = [
        f"experiment_id: {experiment_id}",
        f"experiment_name: {experiment_name}",
        f"logged_utc: {datetime.now(timezone.utc).isoformat()}",
        "",
        "info_panel:",
        *[f"- {line}" for line in info_lines],
        "",
        "copied_visualizations:",
        *copied_lines,
    ]
    (run_dir / "info_panel.txt").write_text("\n".join(info_text), encoding="utf-8")
    return run_dir

