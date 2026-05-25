"""
Lightweight dataset parsing, cleaning, and featurization helpers.

Not needed for the default problem spaces. As they are already clean. But is useful for extending the project to other problem spaces where the data still needs to be processed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Record = Dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_DATASET_EXTENSIONS = {".csv", ".json", ".jsonl"}


def load_dataset(
    path: str | Path,
    *,
    allowed_roots: Optional[Sequence[str | Path]] = None,
) -> List[Record]:
    """
    Load a dataset from CSV, JSON (list[dict]), or JSONL.

    Args:
        path: Path to dataset file.
        allowed_roots: Approved parent locations. Defaults to project root.

    Returns:
        List of row dictionaries.
    """
    dataset_path = _validate_dataset_path(
        path,
        allowed_roots=allowed_roots,
        allowed_extensions=ALLOWED_DATASET_EXTENSIONS,
        must_exist=True,
    )
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [dict(row) for row in data if isinstance(row, dict)]
        raise ValueError("JSON dataset must be a list of objects.")
    if suffix == ".jsonl":
        rows: List[Record] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(dict(obj))
        return rows
    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def write_dataset(
    rows: Iterable[Record],
    path: str | Path,
    *,
    allowed_roots: Optional[Sequence[str | Path]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Safely write dataset rows to CSV/JSON/JSONL with path validation.

    Args:
        rows: Sequence/iterable of row dictionaries.
        path: Destination path.
        allowed_roots: Approved parent locations. Defaults to project root.
        overwrite: If False, reject writing when destination already exists.
    """
    output_path = _validate_dataset_path(
        path,
        allowed_roots=allowed_roots,
        allowed_extensions=ALLOWED_DATASET_EXTENSIONS,
        must_exist=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without overwrite=True: {output_path}"
        )

    rows_list = [dict(r) for r in rows]
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        fieldnames = sorted({k for r in rows_list for k in r.keys()})
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_list:
                writer.writerow(row)
        return output_path
    if suffix == ".json":
        output_path.write_text(json.dumps(rows_list, indent=2), encoding="utf-8")
        return output_path
    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows_list:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return output_path
    raise ValueError(f"Unsupported dataset format: {output_path.suffix}")


def clean_dataset(
    rows: Iterable[Record],
    *,
    required_fields: Optional[Sequence[str]] = None,
    numeric_fields: Optional[Sequence[str]] = None,
    fill_values: Optional[Dict[str, Any]] = None,
    deduplicate: bool = True,
) -> List[Record]:
    """
    Clean dataset rows with basic wrangling steps.

    Steps:
      1) Strip whitespace from string values.
      2) Cast configured numeric fields to float where possible.
      3) Fill missing values using ``fill_values``.
      4) Drop rows missing required fields.
      5) Optionally deduplicate rows.
    """
    required = list(required_fields or [])
    numeric = list(numeric_fields or [])
    fills = dict(fill_values or {})

    cleaned: List[Record] = []
    seen: set[Tuple[Tuple[str, str], ...]] = set()

    for raw in rows:
        row: Record = {}
        for k, v in raw.items():
            if isinstance(v, str):
                v = v.strip()
                if v == "":
                    v = None
            row[k] = v

        for field in numeric:
            if field not in row:
                continue
            value = row[field]
            if value is None:
                continue
            try:
                row[field] = float(value)
            except (TypeError, ValueError):
                row[field] = None

        for field, default in fills.items():
            if row.get(field) is None:
                row[field] = default

        missing_required = any(row.get(field) is None for field in required)
        if missing_required:
            continue

        if deduplicate:
            key = tuple(sorted((k, repr(v)) for k, v in row.items()))
            if key in seen:
                continue
            seen.add(key)

        cleaned.append(row)

    return cleaned


def featurize_dataset(
    rows: Iterable[Record],
    *,
    x_field: str = "x",
    y_field: str = "y",
    objective_field: str = "fitness",
) -> List[Record]:
    """
    Add derived optimization features to each row when fields are available.

    Added fields (when possible):
      - radius_origin: sqrt(x^2 + y^2)
      - abs_x, abs_y
      - objective_sign: -1 (negative), 0 (zero), 1 (positive)
    """
    out: List[Record] = []
    for src in rows:
        row = dict(src)
        x = _coerce_float(row.get(x_field))
        y = _coerce_float(row.get(y_field))
        obj = _coerce_float(row.get(objective_field))

        if x is not None and y is not None:
            row["radius_origin"] = (x * x + y * y) ** 0.5
            row["abs_x"] = abs(x)
            row["abs_y"] = abs(y)

        if obj is not None:
            row["objective_sign"] = 1 if obj > 0 else (-1 if obj < 0 else 0)

        out.append(row)
    return out


def normalize_fields(
    rows: Iterable[Record],
    fields: Sequence[str],
    *,
    method: str = "zscore",
) -> Tuple[List[Record], Dict[str, Dict[str, float]]]:
    """
    Normalize selected numeric fields and append ``<field>_norm`` columns.

    Args:
        rows: Input row dictionaries.
        fields: Numeric field names to normalize.
        method: ``"zscore"`` or ``"minmax"``.

    Returns:
        Tuple of (normalized_rows, stats_by_field).
    """
    records = [dict(r) for r in rows]
    stats: Dict[str, Dict[str, float]] = {}

    for field in fields:
        values = [_coerce_float(r.get(field)) for r in records]
        nums = [v for v in values if v is not None]
        if not nums:
            continue

        if method == "zscore":
            mu = mean(nums)
            sigma = pstdev(nums) or 1.0
            stats[field] = {"mean": float(mu), "std": float(sigma)}
            for r, v in zip(records, values):
                r[f"{field}_norm"] = None if v is None else (v - mu) / sigma
        elif method == "minmax":
            lo = min(nums)
            hi = max(nums)
            span = (hi - lo) or 1.0
            stats[field] = {"min": float(lo), "max": float(hi)}
            for r, v in zip(records, values):
                r[f"{field}_norm"] = None if v is None else (v - lo) / span
        else:
            raise ValueError("method must be 'zscore' or 'minmax'")

    return records, stats


def prepare_for_experiment(
    path: str | Path,
    *,
    allowed_roots: Optional[Sequence[str | Path]] = None,
    required_fields: Optional[Sequence[str]] = None,
    numeric_fields: Optional[Sequence[str]] = None,
    fill_values: Optional[Dict[str, Any]] = None,
    normalize_numeric_fields: Optional[Sequence[str]] = None,
    normalize_method: str = "zscore",
) -> Dict[str, Any]:
    """
    End-to-end dataset preparation utility for experiments.

    Returns a dictionary containing transformed rows and metadata so callers can
    log the wrangling steps in reports.
    """
    loaded = load_dataset(path, allowed_roots=allowed_roots)
    cleaned = clean_dataset(
        loaded,
        required_fields=required_fields,
        numeric_fields=numeric_fields,
        fill_values=fill_values,
    )
    featurized = featurize_dataset(cleaned)

    stats: Dict[str, Dict[str, float]] = {}
    final_rows = featurized
    if normalize_numeric_fields:
        final_rows, stats = normalize_fields(
            featurized,
            normalize_numeric_fields,
            method=normalize_method,
        )

    return {
        "rows": final_rows,
        "meta": {
            "source_path": str(path),
            "n_loaded": len(loaded),
            "n_cleaned": len(cleaned),
            "n_final": len(final_rows),
            "normalized_fields": list(normalize_numeric_fields or []),
            "normalization_method": normalize_method if normalize_numeric_fields else None,
            "normalization_stats": stats,
        },
    }


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None


def _validate_dataset_path(
    path: str | Path,
    *,
    allowed_roots: Optional[Sequence[str | Path]],
    allowed_extensions: set[str],
    must_exist: bool,
) -> Path:
    raw = Path(path)
    resolved = raw.resolve(strict=False)
    suffix = resolved.suffix.lower()
    if suffix not in allowed_extensions:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Allowed: {sorted(allowed_extensions)}"
        )

    roots = [Path(r).resolve() for r in (allowed_roots or [PROJECT_ROOT])]
    if not any(_is_subpath(resolved, root) for root in roots):
        raise ValueError(
            f"Path '{resolved}' is outside approved directories: {[str(r) for r in roots]}"
        )
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    return resolved


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "load_dataset",
    "write_dataset",
    "clean_dataset",
    "featurize_dataset",
    "normalize_fields",
    "prepare_for_experiment",
]
