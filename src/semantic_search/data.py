"""Load, clean, and serialize tabular records for semantic search."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_WHITESPACE = re.compile(r"\s+")


def _is_missing(value: Any) -> bool:
    """Return true for scalar null/NaN values without confusing arrays with nulls."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}
    if pd.api.types.is_scalar(value):
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False
    return False


def as_text(value: Any) -> str:
    """Flatten nested product text fields and normalize whitespace."""
    if _is_missing(value):
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        parts = [as_text(item) for item in value]
        return _WHITESPACE.sub(" ", " ".join(part for part in parts if part)).strip()
    if isinstance(value, dict):
        parts = [f"{key}: {as_text(item)}" for key, item in value.items()]
        return _WHITESPACE.sub(" ", " ".join(part for part in parts if part.split(": ", 1)[-1])).strip()
    return _WHITESPACE.sub(" ", str(value)).strip()


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy and nested values to JSON-serializable values."""
    if _is_missing(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV, JSON Lines, or Parquet into a dataframe."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError("Supported file types are Parquet, CSV, and JSONL.")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read Phase 1 ``id/text/metadata`` JSONL records in file order and validate them.

    Repeated IDs are allowed because Phase 1 reports but keeps them, so the index
    aligns records by row position and checks the ID at every row.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Records file not found: {path}. Create it with: python -m src.semantic_search.data ..."
        )
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{where}: expected a JSON object.")
            record_id, text, metadata = record.get("id"), record.get("text"), record.get("metadata", {})
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{where}: 'id' must be a non-empty string.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{where}: 'text' must be a non-empty string.")
            if not isinstance(metadata, dict):
                raise ValueError(f"{where}: 'metadata' must be a JSON object.")
            records.append({"id": record_id, "text": text, "metadata": metadata})
    if not records:
        raise ValueError(f"Records file is empty: {path}.")
    return records


def make_records(
    frame: pd.DataFrame,
    text_columns: Sequence[str],
    id_column: str | None = None,
    exclude_metadata: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], int]:
    """Build normalized ``id/text/metadata`` records; return empty-text count."""
    if not text_columns:
        raise ValueError("Select at least one text column.")
    missing = [column for column in text_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Text columns not found: {missing}. Available: {list(frame.columns)}")
    if id_column and id_column not in frame.columns:
        raise ValueError(f"ID column {id_column!r} was not found.")

    records: list[dict[str, Any]] = []
    empty_count = 0
    for row_number, (_, row) in enumerate(frame.iterrows()):
        fields = []
        for column in text_columns:
            field_text = as_text(row[column])
            if field_text:
                label = re.sub(r"[_\-]+", " ", str(column)).strip().title()
                fields.append(f"{label}: {field_text}")
        if not fields:
            empty_count += 1
            continue

        raw_id = as_text(row[id_column]) if id_column else ""
        metadata = {
            str(column): _json_safe(value)
            for column, value in row.items()
            if column not in text_columns and column not in exclude_metadata
        }
        records.append({
            "id": raw_id or str(row_number),
            "text": " ".join(fields),
            "metadata": metadata,
        })
    return records, empty_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Input CSV, JSONL, or Parquet path")
    parser.add_argument("--text-columns", nargs="+", required=True)
    parser.add_argument("--id-column")
    parser.add_argument("--exclude-metadata", nargs="*", default=[])
    parser.add_argument("--output", default="data/processed/products.jsonl")
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    frame = read_table(args.file)
    records, empty_count = make_records(
        frame, args.text_columns, args.id_column, args.exclude_metadata
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    ids = [record["id"] for record in records]
    duplicate_count = len(ids) - len(set(ids))
    print(f"Rows loaded: {len(frame):,}")
    print(f"Searchable records: {len(records):,}")
    print(f"Rows skipped (no text): {empty_count:,}")
    print(f"Repeated IDs (reported, not removed): {duplicate_count:,}")
    print(f"Saved normalized JSONL: {output}")
    print(f"Text columns: {', '.join(args.text_columns)}")
    print(f"Metadata columns: {', '.join(str(c) for c in frame.columns if c not in args.text_columns and c not in args.exclude_metadata)}")
    for record in records[: max(0, args.show)]:
        print(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2)[:1800])


if __name__ == "__main__":
    main()