"""
services/mongodb_service.py

Responsibilities:
  - Connect to MongoDB (URI from environment variable MONGO_URI)
  - Save Excel query reports to the 'savedReports' collection
  - Retrieve previously saved reports (history)

Collection schema (savedReports):
    {
        "_id"       : ObjectId,
        "query"     : str,
        "createdAt" : datetime (UTC),
        "totalRows" : int,
        "rows"      : list[dict]
    }

This module is completely independent from the existing RAG / PDF services.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MongoDB connection — imported only when first used so the rest of
# the application keeps working even if pymongo is not installed.
# ---------------------------------------------------------------------------

_client = None
_db = None


def _get_db():
    """Return the MongoDB database instance, creating it on first call."""
    global _client, _db
    if _db is not None:
        return _db

    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError(
            "pymongo is not installed. "
            "Run 'pip install pymongo' to enable MongoDB support."
        ) from exc

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "smartdocs_ai")

    _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)

    # Ping to validate the connection immediately
    try:
        _client.admin.command("ping")
        logger.info("MongoDB connected: uri=%s | db=%s", mongo_uri, db_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to MongoDB at '{mongo_uri}': {exc}"
        ) from exc

    _db = _client[db_name]
    return _db


COLLECTION_NAME = "savedReports"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_report(query: str, rows: list[dict[str, Any]]) -> str:
    """
    Insert a new report document into MongoDB.

    Parameters
    ----------
    query : str            — Human-readable label for the report
    rows  : list[dict]     — The data rows to persist

    Returns
    -------
    str — The inserted document's _id as a string
    """
    db = _get_db()
    collection = db[COLLECTION_NAME]

    document = {
        "query": query,
        "createdAt": datetime.now(tz=timezone.utc),
        "totalRows": len(rows),
        "rows": rows,
    }

    result = collection.insert_one(document)
    inserted_id = str(result.inserted_id)

    logger.info(
        "Report saved: id=%s | query='%s' | totalRows=%d",
        inserted_id,
        query,
        len(rows),
    )
    return inserted_id


def get_history() -> list[dict[str, Any]]:
    """
    Return a summary list of all saved reports (newest first).
    Each entry contains: id, query, createdAt, totalRows.
    The full 'rows' payload is excluded to keep responses lightweight.

    Returns
    -------
    list[dict]
    """
    db = _get_db()
    collection = db[COLLECTION_NAME]

    cursor = collection.find(
        {},
        {"_id": 1, "query": 1, "createdAt": 1, "totalRows": 1},
    ).sort("createdAt", -1)  # newest first

    history = []
    for doc in cursor:
        history.append(
            {
                "id": str(doc["_id"]),
                "query": doc.get("query", ""),
                "createdAt": doc.get("createdAt"),
                "totalRows": doc.get("totalRows", 0),
            }
        )

    logger.info("History retrieved: %d reports", len(history))
    return history
