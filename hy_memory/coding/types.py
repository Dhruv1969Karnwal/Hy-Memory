# -*- coding: utf-8 -*-
"""
Coding Memory data structures

Design principles:
- Fully separate from the chat pipeline, with its own schema
- Multi-key retrieval via task + search_keys
- boundary_envs records the environment an experience depends on, so consumers can judge whether it still applies
- Overwriting update (no history kept), may carry a DELETE op

See docs/scene_coding_usage.md for the public scene and response structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal


# Fact-layer status values
FACT_STATUSES = ("active", "superseded")
FactStatus = Literal["active", "superseded"]

# How many generations back to walk when writing the evolution summary. Without a cap, every overwrite of a long-lived fact
# would traverse the whole fact_link chain, costing extra queries and blowing up the prompt; the excess is marked as "earlier generations omitted".
MAX_FACT_CHAIN_ENTRIES = 8


# ================================================================
# Draft (output of the LLM extract stage)
# ================================================================

@dataclass
class CodingMemoryDraft:
    """
    Draft produced by the LLM extractor, not yet reconciled or persisted.

    Fields serve extraction and reconcile inside the coding/working pipeline.
    """
    # Retrieval primary keys
    task: str
    search_keys: List[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""

    confidence: float = 0.7

    # Context (injected by the SDK)
    user_id: str = ""
    agent_id: str = "default_agent"
    workspace_id: Optional[str] = None
    branch: Optional[str] = None
    session_id: Optional[str] = None
    files: List[str] = field(default_factory=list)
    source: str = "default"
    extraction_source: str = "auto_extract"

    def __post_init__(self) -> None:
        from ..utils.source_filter import normalize_source
        self.source = normalize_source(self.source)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "search_keys": list(self.search_keys),
            "solution": self.solution,
            "boundary_envs": self.boundary_envs,
            "confidence": self.confidence,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "session_id": self.session_id,
            "files": list(self.files),
            "source": self.source,
            "extraction_source": self.extraction_source,
        }


# ================================================================
# Persisted memory
# ================================================================

@dataclass
class CodingMemory:
    """Persisted experience-layer memory (one record shared by the SQLite and VDB layers)."""
    # Primary keys
    experience_id: str
    user_id: str
    agent_id: str = "default_agent"

    # Retrieval primary keys
    task: str = ""
    search_keys: List[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""

    # Scope (exposed in results and also optional filters for search / list)
    workspace_id: Optional[str] = None
    branch: Optional[str] = None

    # Context
    session_id: Optional[str] = None
    files: List[str] = field(default_factory=list)

    # Cross-layer links: which facts this experience derives from (backfilled across the whole batch)
    source_fact_ids: List[str] = field(default_factory=list)

    # Metadata
    confidence: float = 0.7
    source: str = "default"
    extraction_source: str = "auto_extract"

    # Time
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        from ..utils.source_filter import normalize_source
        self.source = normalize_source(self.source)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "task": self.task,
            "search_keys": list(self.search_keys),
            "solution": self.solution,
            "boundary_envs": self.boundary_envs,
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "session_id": self.session_id,
            "files": list(self.files),
            "source_fact_ids": list(self.source_fact_ids),
            "confidence": self.confidence,
            "source": self.source,
            "extraction_source": self.extraction_source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ================================================================
# Fact layer (episodic): WorkingFact
#
# A second layer beside the experience layer (procedural). The experience layer stores "how to do it" (reusable methods, cross-session);
# the fact layer stores "how this one went" (the whole record of one trace/session, serving conversational continuity).
#
# Design decisions (see the discussion log):
#   - No TTL: freshness comes from embed top-K recall plus a separate FactReconciler(LLM) overwrite decision,
#             not from expiry eviction. On overwrite the old fact goes status→superseded and fact_link is backfilled.
#   - session_id is stored but not a hard isolation boundary (same as the experience layer); retrieval recalls per user.
#   - Cross-layer links: fact.derived_experience_ids ↔ experience.source_fact_ids (full linking within a batch).
#   - subject hard anchor (switchable, see fact_reconciler._SUBJECT_GATE_DEFAULT): when on, an overwrite
#     decision must pass the subject gate first, then semantic recall + LLM.
#     Without this anchor (pure semantics + LLM overwrite decisions), "several parallel subjects in one workspace"
#     goes wrong easily: when several people share one itinerary or several machines share one deployment script, the
#     subjects' statements are semantically very close, so the LLM may read them as earlier/later versions of the same task and one write
#     replaces someone else's fact. Forbidding overwrite when subject differs avoids that: near-identical wording at worst keeps one extra row,
#     while cross-subject overwrite deletes someone else's state outright.
#
#     Why it is a switch: the gate compares strings exactly (see normalize_subject below),
#     so its correctness depends entirely on subject being a stable entity name. When subject is free text or a location
#     label, one entity gets written as several subjects that block each other; if a later step depends on an earlier
#     result, blocking a real update makes it read a stale value. Hence it is off by default and enabled with
#     HY_FACT_SUBJECT_GATE=1 when needed.

# ================================================================


def normalize_subject(subject: Optional[str]) -> str:
    """Comparison form of subject: stripped, inner whitespace collapsed, lowercased.

    The anchor only decides "is this the same subject", so `Dylan` / `dylan ` / `DYLAN` must compare equal;
    but no synonym merging is done (`auth/middleware.ts` and `middleware` are two subjects) -- that is a
    semantic judgement and belongs to the LLM step. An empty subject returns an empty string; callers treat it as "no anchor".
    """
    return " ".join((subject or "").split()).lower()


@dataclass
class WorkingFactDraft:
    """Session-level fact draft produced by the LLM extractor, not yet reconciled or persisted.

    Context (user_id / session_id, etc.) is not part of the draft; FactReconciler injects it when persisting.
    At most one per trace segment; statement is a resumable state record that concisely covers the user's task, what was actually done,
    the evidence boundary and the current ending state, while keeping retrieval anchors such as file/directory paths, symbols, commands,
    error strings, versions and configuration values.
    """
    statement: str = ""
    files: List[str] = field(default_factory=list)
    # The subject of this session segment: the person / file / service / component being operated on. Hard anchor for overwrite decisions.
    subject: str = ""
    # Addressable source text the session produced: plan tables, config blocks, command output, structured results. statement is
    # the summary (the retrieval vector); details holds the concrete values that summarizing dropped -- kept verbatim so they can be
    # read for real after a retrieval hit. Summaries inherently lose addressed cells such as "what for lunch on day two and how much",
    # which is exactly the part downstream consumers need.
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "files": list(self.files),
            "subject": self.subject,
            "details": self.details,
        }


@dataclass
class WorkingFact:
    """The whole record of one trace/session (SQLite working_fact plus a separate facts collection)."""
    # Primary keys
    fact_id: str
    user_id: str
    agent_id: str = "default_agent"

    # Context (stored but not a hard isolation boundary, same as the experience layer)
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    branch: Optional[str] = None
    source: str = "default"

    # Content
    statement: str = ""                    # Session summary (task/approach/result) = retrieval vector
    files: List[str] = field(default_factory=list)  # File/directory paths actually operated on or investigated in this trace

    # Subject of this session segment (person / file / service / component). Hard gate for overwrite decisions: a different
    # subject may never overwrite, see the module header comment. An empty string means extraction did not settle on a subject, and we
    # fall back to pure semantics + LLM (rows in old databases are empty strings too, so behaviour matches the pre-change code).
    subject: str = ""

    # Addressable source text the session produced; excluded from the retrieval vector and handed to downstream readers with statement after a hit.
    # statement makes the record findable; details makes the found value actually present.
    details: str = ""

    # Overwrite / cross-links
    fact_link: Optional[str] = None        # Which newer fact overwrote this one (backfilled after overwrite, for tracing)
    derived_experience_ids: List[str] = field(default_factory=list)  # Which experiences were derived from it
    status: FactStatus = "active"          # active / superseded (soft delete)

    # LLM-written progress summary: "how the same task advanced across sessions to the current state", rewritten on every SUPERSEDE.
    # An empty string means there is no evolution to tell (never superseded) or that LLM call failed.
    #
    # The history itself is not stored here -- superseded facts stay in the database as they are, fact_link points at the successor, and
    # walking it in reverse reconstructs every past wording. Only the distilled sentence lives here.
    #
    # Kept out of the vector database: retrieval matches the current value of statement, and embedding past wordings would give
    # already voided phrasing recall power again.
    fact_chain_summary: str = ""

    # Time
    created_at: Optional[datetime] = None  # Creation time; newer records take precedence

    def __post_init__(self) -> None:
        from ..utils.source_filter import normalize_source
        self.source = normalize_source(self.source)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "source": self.source,
            "statement": self.statement,
            "files": list(self.files),
            "subject": self.subject,
            "details": self.details,
            "fact_link": self.fact_link,
            "derived_experience_ids": list(self.derived_experience_ids),
            "status": self.status,
            "fact_chain_summary": self.fact_chain_summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ================================================================
# Fact lineage
# ================================================================

def short_date(iso: str) -> str:
    """ISO timestamp truncated to the day. Returned as-is when it cannot be parsed: ugly beats losing information."""
    s = (iso or "").strip()
    return s[:10] if len(s) >= 10 else s


# ================================================================
# Reconcile op
# ================================================================

ReconcileAction = Literal["ADD", "UPDATE", "DELETE", "SKIP"]


@dataclass
class ReconcileOp:
    """
    A single operation decided by the Reconciler.

    - ADD     ← insert triggered by a draft; draft_idx required, target_experience_id unset
    - UPDATE  ← update triggered by a draft; both draft_idx and target_experience_id required
    - DELETE  ← new content explicitly denies or voids a candidate; draft_idx records "which draft triggered it"
                and may be empty (rare cases), target_experience_id required
    - SKIP    ← duplicate; draft_idx required, target_experience_id unset

    Note: one reconcile call may emit *more* ops than there are drafts (a draft can come with several DELETE ops).
    """
    action: ReconcileAction
    draft_idx: Optional[int] = None
    target_experience_id: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "draft_idx": self.draft_idx,
            "target_experience_id": self.target_experience_id,
            "reason": self.reason,
        }


# ================================================================
# Fact-layer op
#
# Session-level facts have no value threshold, so every draft always yields one INSERT; if it continues the same user task recorded by
# an older fact, it additionally yields SUPERSEDE (soft delete plus adoption of the experience chain).
# ================================================================

FactReconcileAction = Literal["INSERT", "SUPERSEDE"]


@dataclass
class FactReconcileOp:
    """
    - INSERT     ← new fact persisted as active; draft_idx + fact_id required
    - SUPERSEDE  ← the old session fact and the new fact are consecutive progress on the same user task;
                   fact_id is the old record, superseded_by is the new progress record
    """
    action: FactReconcileAction
    draft_idx: Optional[int] = None
    fact_id: Optional[str] = None
    superseded_by: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "draft_idx": self.draft_idx,
            "fact_id": self.fact_id,
            "superseded_by": self.superseded_by,
            "reason": self.reason,
        }
