"""
services/excel_loader.py

Responsibilities:
  - Validate uploaded Excel files (.xlsx / .xls)
  - Read the file into a Pandas DataFrame
  - Generate a unique fileId
  - Store the DataFrame in server memory (in-process cache)
  - Expose helpers to retrieve / evict cached DataFrames

This module is completely independent from the existing RAG / PDF services.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory DataFrame store
# key  : fileId (str)
# value: dict with "df" (DataFrame) and "sheet_name" (str)
# ---------------------------------------------------------------------------
_DATAFRAME_STORE: Dict[str, dict] = {}

# Supported Excel extensions
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}

# Max file size: 10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_extension(filename: str) -> str:
    """
    Return the lowercase extension if it is allowed.
    Raises ValueError with a user-friendly message otherwise.
    """
    if not filename:
        raise ValueError("No filename provided.")
    ext = _get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are accepted."
        )
    return ext


def load_excel(file_path: str, filename: str) -> dict:
    """
    Read an Excel file from *file_path*, store its first sheet's DataFrame
    in memory, and return metadata.

    Parameters
    ----------
    file_path : str
        Absolute path to the temporary file on disk.
    filename  : str
        Original uploaded filename (used for extension detection).

    Returns
    -------
    dict with keys: fileId, sheetName, totalRows, totalColumns, headers
    """
    ext = validate_extension(filename)

    try:
        engine = "xlrd" if ext == ".xls" else "openpyxl"
        sheets: dict = pd.read_excel(file_path, sheet_name=None, engine=engine)
    except Exception as exc:
        logger.error("Failed to read Excel file '%s': %s", filename, exc)
        raise ValueError(
            f"Could not read the Excel file. "
            f"It may be corrupted or password-protected. Details: {exc}"
        ) from exc

    if not sheets:
        raise ValueError("The Excel file contains no sheets.")

    # Use the first non-empty sheet
    df, sheet_name = _pick_first_non_empty_sheet(sheets, filename)

    # Clean up
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("All sheets in the Excel file are empty.")

    file_id = str(uuid.uuid4())

    _DATAFRAME_STORE[file_id] = {
        "df": df,
        "sheet_name": sheet_name,
    }

    logger.info(
        "Loaded Excel '%s' → fileId=%s | sheet='%s' | rows=%d | cols=%d",
        filename,
        file_id,
        sheet_name,
        len(df),
        len(df.columns),
    )

    return {
        "fileId": file_id,
        "sheetName": sheet_name,
        "totalRows": len(df),
        "totalColumns": len(df.columns),
        "headers": list(df.columns),
    }


def get_dataframe(file_id: str) -> pd.DataFrame:
    """
    Retrieve a cached DataFrame by fileId.
    Raises KeyError if the fileId is unknown or has been evicted.
    """
    entry = _DATAFRAME_STORE.get(file_id)
    if entry is None:
        raise KeyError(
            f"No file found for fileId='{file_id}'. "
            "The file may have expired or the fileId is incorrect."
        )
    return entry["df"]


def evict(file_id: str) -> None:
    """Remove a cached DataFrame (call when no longer needed)."""
    _DATAFRAME_STORE.pop(file_id, None)
    logger.debug("Evicted fileId=%s from store", file_id)


def store_size() -> int:
    """Return the number of DataFrames currently cached (for diagnostics)."""
    return len(_DATAFRAME_STORE)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_extension(filename: str) -> str:
    import os
    return os.path.splitext(filename)[1].lower()


def _pick_first_non_empty_sheet(
    sheets: dict, filename: str
) -> tuple[pd.DataFrame, str]:
    """
    Iterate over sheets and return the first one that has actual data.
    Raises ValueError if every sheet is empty.
    """
    for sheet_name, df in sheets.items():
        cleaned = df.dropna(how="all").dropna(axis=1, how="all")
        if not cleaned.empty:
            return df, sheet_name

    raise ValueError(
        f"'{filename}' contains no readable data — all sheets are empty."
    )
