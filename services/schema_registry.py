"""
services/schema_registry.py

Schema Registry — Per-fileId column metadata store.

Responsibilities:
  - Inspect a Pandas DataFrame and infer per-column types (numeric / date / string / boolean)
  - Compute sample values, min/max, and unique count per column
  - Store DatasetSchema under fileId
  - Provide schema to AI prompting (richer context) and QueryValidator (type-aware validation)

Why this exists:
  Sending column names alone to the AI is insufficient. With type + sample values, the AI can:
    - Choose correct operators (> for numeric, month for date, contains for string)
    - Avoid type mismatches (e.g., applying > to a string column)
    - Understand domain values (e.g., "Pending", "Paid" → knows to use == or contains)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from schemas.query_plan_schema import ColumnSchema, DatasetSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory registry
# ---------------------------------------------------------------------------

_SCHEMA_STORE: Dict[str, DatasetSchema] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register(file_id: str, df: pd.DataFrame, sheet_name: str = "Sheet1") -> DatasetSchema:
    """
    Inspect *df* and store a DatasetSchema under *file_id*.

    Parameters
    ----------
    file_id    : str            — the fileId from excel_loader
    df         : pd.DataFrame   — the loaded DataFrame
    sheet_name : str            — name of the sheet (for metadata)

    Returns
    -------
    DatasetSchema
    """
    columns: list[ColumnSchema] = []

    for col in df.columns:
        series = df[col]
        dtype = _infer_dtype(series)
        non_null = series.dropna()

        samples: list[Any]
        min_val: Any = None
        max_val: Any = None

        if dtype == "numeric":
            try:
                numeric = pd.to_numeric(non_null, errors="coerce").dropna()
                samples = [_safe_scalar(v) for v in numeric.unique()[:5]]
                min_val = _safe_scalar(numeric.min()) if not numeric.empty else None
                max_val = _safe_scalar(numeric.max()) if not numeric.empty else None
            except Exception:
                samples = [str(v) for v in non_null.unique()[:5]]
        elif dtype == "date":
            try:
                dates = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True).dropna()
                samples = [str(d.date()) for d in dates.unique()[:5]]
                min_val = str(dates.min().date()) if not dates.empty else None
                max_val = str(dates.max().date()) if not dates.empty else None
            except Exception:
                samples = [str(v) for v in non_null.unique()[:5]]
        else:
            samples = [str(v) for v in non_null.unique()[:5]]

        col_schema = ColumnSchema(
            name=col,
            dtype=dtype,
            sample_values=samples,
            nullable=bool(series.isna().any()),
            min_value=min_val,
            max_value=max_val,
            unique_count=int(series.nunique()),
        )
        columns.append(col_schema)

    schema = DatasetSchema(
        file_id=file_id,
        sheet_name=sheet_name,
        total_rows=len(df),
        columns=columns,
    )
    _SCHEMA_STORE[file_id] = schema

    logger.info(
        "Schema registered: fileId=%s | sheet=%s | rows=%d | cols=%d",
        file_id,
        sheet_name,
        len(df),
        len(columns),
    )
    return schema


def get_schema(file_id: str) -> DatasetSchema:
    """Retrieve the schema for a previously loaded file. Raises KeyError if not found."""
    schema = _SCHEMA_STORE.get(file_id)
    if schema is None:
        raise KeyError(
            f"No schema found for fileId='{file_id}'. "
            "The file may have expired or the fileId is incorrect."
        )
    return schema


def evict(file_id: str) -> None:
    """Remove schema when the DataFrame is evicted."""
    _SCHEMA_STORE.pop(file_id, None)
    logger.debug("Schema evicted for fileId=%s", file_id)


def registry_size() -> int:
    """Return number of schemas currently stored."""
    return len(_SCHEMA_STORE)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _infer_dtype(series: pd.Series) -> str:
    """Infer the semantic type of a Pandas Series."""

    # Explicit pandas types first
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return "string"

    # Try numeric
    numeric_conv = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = numeric_conv.notna().sum() / max(len(non_null), 1)
    if numeric_ratio >= 0.8:
        return "numeric"

    # Try date
    try:
        date_conv = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
        date_ratio = date_conv.notna().sum() / max(len(non_null), 1)
        if date_ratio >= 0.7:
            return "date"
    except Exception:
        pass

    return "string"


def _safe_scalar(value: Any) -> Any:
    """Convert numpy scalars to Python native types for JSON serialization."""
    try:
        if hasattr(value, "item"):  # numpy scalar
            return value.item()
        return value
    except Exception:
        return str(value)
