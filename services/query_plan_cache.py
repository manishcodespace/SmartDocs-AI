"""
services/query_plan_cache.py

Query Plan Cache — In-memory LRU store for ExecutionPlans.

Key   : (file_id, normalized_question)
Value : ExecutionPlan (Pydantic model)

Benefits:
  - Identical question on the same file skips the AI API call entirely
  - Reduces per-query latency from ~800–1200ms (Gemini round-trip) to <5ms
  - Bounded memory usage (max _MAX_SIZE entries, FIFO eviction)

Production upgrade path:
  - Swap _CACHE dict for a Redis client to support multi-worker/multi-process deployments
  - Add TTL per entry (e.g., 30 minutes) if DataFrame can be re-uploaded with new data
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

from schemas.query_plan_schema import ExecutionPlan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_SIZE: int = 256  # Max plans cached across all users/sessions

# ---------------------------------------------------------------------------
# LRU Cache (OrderedDict-based, thread-safe for single-worker deployments)
# ---------------------------------------------------------------------------

_CACHE: OrderedDict[tuple[str, str], ExecutionPlan] = OrderedDict()

# ---------------------------------------------------------------------------
# Stats (for diagnostics/health endpoint)
# ---------------------------------------------------------------------------

_hits: int = 0
_misses: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get(file_id: str, question: str) -> Optional[ExecutionPlan]:
    """
    Retrieve a cached ExecutionPlan for (file_id, question).

    Parameters
    ----------
    file_id  : str — the fileId from excel_loader
    question : str — the user's natural-language question (normalized internally)

    Returns
    -------
    ExecutionPlan if cached, else None
    """
    global _hits, _misses

    key = _make_key(file_id, question)
    plan = _CACHE.get(key)

    if plan is not None:
        # Move to end (most recently used)
        _CACHE.move_to_end(key)
        _hits += 1
        logger.debug(
            "Cache HIT [%d hits / %d misses]: fileId=%s | q='%s...'",
            _hits, _misses, file_id, question[:40],
        )
        return plan

    _misses += 1
    logger.debug(
        "Cache MISS [%d hits / %d misses]: fileId=%s | q='%s...'",
        _hits, _misses, file_id, question[:40],
    )
    return None


def put(file_id: str, question: str, plan: ExecutionPlan) -> None:
    """
    Store an ExecutionPlan in the cache.

    Parameters
    ----------
    file_id  : str           — the fileId
    question : str           — the user's question (normalized internally)
    plan     : ExecutionPlan — the plan to cache
    """
    key = _make_key(file_id, question)

    if key in _CACHE:
        # Update existing entry, move to end
        _CACHE.move_to_end(key)
        _CACHE[key] = plan
        logger.debug("Cache UPDATE: fileId=%s | q='%s...'", file_id, question[:40])
        return

    if len(_CACHE) >= _MAX_SIZE:
        # Evict least recently used (first item in OrderedDict)
        evicted_key, _ = _CACHE.popitem(last=False)
        logger.debug("Cache LRU eviction: key=%s", evicted_key)

    _CACHE[key] = plan
    logger.debug(
        "Cache SET [size=%d]: fileId=%s | q='%s...'",
        len(_CACHE), file_id, question[:40],
    )


def invalidate_file(file_id: str) -> int:
    """
    Remove all cached plans for a given fileId (e.g., when file is evicted).

    Returns
    -------
    int — number of entries evicted
    """
    keys_to_remove = [k for k in _CACHE if k[0] == file_id]
    for k in keys_to_remove:
        del _CACHE[k]
    if keys_to_remove:
        logger.info(
            "Cache invalidated for fileId=%s: %d entries removed", file_id, len(keys_to_remove)
        )
    return len(keys_to_remove)


def cache_size() -> int:
    """Return the number of entries currently in the cache."""
    return len(_CACHE)


def cache_stats() -> dict:
    """Return diagnostic stats for the health endpoint."""
    return {
        "size": len(_CACHE),
        "max_size": _MAX_SIZE,
        "hits": _hits,
        "misses": _misses,
        "hit_ratio": round(_hits / max(_hits + _misses, 1), 3),
    }


def clear() -> None:
    """Clear all cached plans (e.g., for testing)."""
    global _hits, _misses
    _CACHE.clear()
    _hits = 0
    _misses = 0
    logger.info("Query plan cache cleared")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_key(file_id: str, question: str) -> tuple[str, str]:
    """Normalize question to reduce near-duplicate cache misses."""
    normalized = question.strip().lower()
    # Remove punctuation that doesn't change semantics
    normalized = " ".join(normalized.split())  # collapse whitespace
    return (file_id, normalized)
