# -*- coding: utf-8 -*-
"""
HY Memory - Coding Memory module

An independent memory pipeline designed for productivity / coding scenarios, with a two-layer structure:
- Experience layer CodingMemory (procedural): reusable methods, how something was done
- Fact layer WorkingFact (episodic): the task, approach and outcome of one trace
- Independent physical storage (SQLite meta + two separate VDB collections)
- Independent extractor / two reconcilers / store
- Fully separated from the existing chat pipeline, zero regressions on the chat path
"""

from .types import (
    CodingMemory,
    CodingMemoryDraft,
    ReconcileOp,
    WorkingFact,
    WorkingFactDraft,
    FactReconcileOp,
    FactReconcileAction,
    FactStatus,
    FACT_STATUSES,
    MAX_FACT_CHAIN_ENTRIES,
    short_date,
)

__all__ = [
    "CodingMemory",
    "CodingMemoryDraft",
    "ReconcileOp",
    "WorkingFact",
    "WorkingFactDraft",
    "FactReconcileOp",
    "FactReconcileAction",
    "FactStatus",
    "FACT_STATUSES",
    "MAX_FACT_CHAIN_ENTRIES",
    "short_date",
]
