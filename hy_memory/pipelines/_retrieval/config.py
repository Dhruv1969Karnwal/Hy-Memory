"""
Central entry point for read pipeline environment variables and constants.

Priority for every parameter:
  process environment variable > code default constant

All environment variables use the `HY_MEMORY_READER_` prefix, aligned with the existing `HY_MEMORY_*` family.
"""

import os


# ========================================================================
# Reader dispatch
# ========================================================================

READER_LEGACY = "legacy"
READER_HYBRID = "hybrid"

ALL_READERS = (READER_LEGACY, READER_HYBRID)


def resolve_reader_name(override: str = "") -> str:
    """
    Resolve the effective reader name: explicit override > environment variable > legacy default.
    An unknown value falls back to legacy and logs a warning.
    """
    name = (override or os.environ.get("HY_MEMORY_READER", "") or READER_LEGACY).strip().lower()
    if name not in ALL_READERS:
        import logging
        logging.getLogger(__name__).warning(
            f"[reader-config] unknown HY_MEMORY_READER={name!r}, fallback to {READER_LEGACY}"
        )
        return READER_LEGACY
    return name


# ========================================================================
# Recall pool sizes
# ========================================================================

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


VEC_POOL_SIZE: int = _int_env("HY_MEMORY_READER_VEC_POOL_SIZE", 30)
"""Candidate pool limit for path A (primary vector recall)."""

TAG_POOL_SIZE: int = _int_env("HY_MEMORY_READER_TAG_POOL_SIZE", 20)
"""Candidate pool limit for path B (tag semantic bridge recall)."""

TAG_MATCH_TOPK: int = _int_env("HY_MEMORY_READER_TAG_MATCH_TOPK", 5)
"""Maximum number of tags matched per query keyword. The main truncation of path B happens here: how
many tags are actually hit is decided by topk, while min_score only soft-filters wholly unrelated noise."""

TAG_MATCH_MIN_SCORE: float = _float_env("HY_MEMORY_READER_TAG_MATCH_MIN_SCORE", 0.3)
"""Soft-filter floor for tag matching (default 0.3).

Semantics: **topk first, min_score secondary** -- take topk by cosine, then drop whatever is below min_score.
No Qdrant server-side hard filter, to avoid "empty recall when the user has no highly relevant tag".

A tag embedding is word/phrase granularity, so its signal-to-noise ratio is weak by nature (the cosine between `food` and `beef noodle`
may be only 0.45~0.55), and a high threshold wipes out every match. Lowered from 0.5 to 0.3 since post16."""

ABSTAIN_THRESHOLD: float = _float_env("HY_MEMORY_READER_ABSTAIN_THRESHOLD", 0.15)
"""Abstain / low-confidence threshold: marks is_low_confidence when the average RRF score of the top3 falls below it."""

KEYWORD_MAX_COUNT: int = _int_env("HY_MEMORY_READER_KEYWORD_MAX_COUNT", 10)
"""Maximum number of keywords extracted from a query for path B batch embed."""


# ========================================================================
# RRF fusion
# ========================================================================

RRF_K: int = _int_env("HY_MEMORY_READER_RRF_K", 60)
"""RRF smoothing constant, the classic value from Cormack et al. 2009."""


# ========================================================================
# BM25-lite parameters
# ========================================================================

BM25_K1: float = _float_env("HY_MEMORY_READER_BM25_K1", 1.5)
BM25_B: float = _float_env("HY_MEMORY_READER_BM25_B", 0.75)


# ========================================================================
# Tag index storage
# ========================================================================

def tag_index_collection_name(memory_collection: str) -> str:
    """Naming rule for the standalone tag_index collection."""
    return f"{memory_collection}_tag_index"


def entity_collection_name(memory_collection: str) -> str:
    """Naming rule for the standalone entity store collection (aligned with mem0's {collection}_entities)."""
    return f"{memory_collection}_entities"


# intent_override for debugging (forces an intention at runtime)
INTENT_OVERRIDE: str = (os.environ.get("HY_MEMORY_READER_INTENT_OVERRIDE", "") or "").strip().upper()


# Hard switch: lets the business side turn off tag_index maintenance entirely (e.g. during a storage pressure review)
# Once it is off, the writer no longer maintains tag_index.
TAG_INDEX_WRITE_ENABLED: bool = (
    (os.environ.get("HY_MEMORY_TAG_INDEX_WRITE", "true") or "true").strip().lower() not in ("0", "false", "off", "no")
)
