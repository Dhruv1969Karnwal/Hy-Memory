"""
Agent Memory V2 - Core data models

Contents:
- Enums: MemoryLayer, MemoryStatus, SourceType, UpdateType,
            SchemaStatus, TriggerType, IntentionPriority, MetaCognitionTag, GapType
- Core model: MemoryNode (unified base class)
- Subclass models: VersionedFact, SchemaNode, IntentionNode
- Auxiliary models: KnowledgeGap, TemporalEvent, UserTimeline, LifeStage
- Output protocol: MemoryContextPackage and its sub-structures
- Compat aliases: MemoryEntry, MemoryMetadata, MemoryScore (legacy interfaces kept)
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid


# ============================================================
# Enums
# ============================================================

class MemoryLayer(str, Enum):
    """
    Seven-layer memory model (V2 Memory Hierarchy)

    L1 RAW       — Raw dialogue layer (Append-Only, vector database)
    L2 FACT      — Atomic fact layer (versioned immutable records)
    L3 SUMMARY   — Session summary layer
    L4 IDENTITY  — Identity profile layer (core profile)
    L5 KNOWLEDGE — Knowledge graph layer (entities/relations/topics, Graph layer, not implemented yet)
    L6 SCHEMA    — Mental model layer (abstract narrative templates, Graph layer)
    L7 INTENTION — Prospective intention layer (intentions awaiting a future trigger, Graph layer)

    Storage split: L0-L4 in the VDB, L5-L7 in Graph (ultra mode)
    """
    L0_BASIC_INFO = "l0_basic_info"
    L1_RAW = "l1_raw"
    L2_FACT = "l2_fact"
    L3_SUMMARY = "l3_summary"
    L4_IDENTITY = "l4_identity"
    L5_KNOWLEDGE = "l5_knowledge"
    L6_SCHEMA = "l6_schema"
    L7_INTENTION = "l7_intention"

    # --- Legacy numbering compat aliases (values found in stored data) ---
    # v2 → v3 renumbering compat
    L6_IDENTITY = "l4_identity"     # old l6_identity → new l4_identity
    L4_KNOWLEDGE = "l5_knowledge"   # old l4_knowledge → new l5_knowledge
    L5_SCHEMA = "l6_schema"         # old l5_schema → new l6_schema
    # v1 → v2 legacy compat (still required)
    L4_5_SCHEMA = "l6_schema"       # old l4_5_schema → new l6_schema
    L5_IDENTITY = "l4_identity"     # old l5_identity → new l4_identity
    L6_INTENTION = "l7_intention"   # old l6_intention → new l7_intention

    # --- V1 compat aliases ---
    PROFILE = "l4_identity"
    DIALOGUE = "l2_fact"
    SUMMARY = "l3_summary"
    KNOWLEDGE = "l5_knowledge"
    RAW = "l1_raw"

    @classmethod
    def from_string(cls, value: str) -> "MemoryLayer":
        """Build a MemoryLayer from a string; supports V1/V2/V3 names and legacy numbering."""
        value = value.lower().strip()
        # Legacy → new numbering map (compat with payloads in stored data)
        legacy_mapping = {
            # v2 → v3
            "l6_identity": cls.L4_IDENTITY,
            "l4_knowledge": cls.L5_KNOWLEDGE,
            "l5_schema": cls.L6_SCHEMA,
            # v1 → v2 → v3
            "l4_5_schema": cls.L6_SCHEMA,
            "l5_identity": cls.L4_IDENTITY,
            "l6_intention": cls.L7_INTENTION,
        }
        if value in legacy_mapping:
            return legacy_mapping[value]
        # V1 compat mapping
        v1_mapping = {
            "profile": cls.L4_IDENTITY,
            "dialogue": cls.L2_FACT,
            "summary": cls.L3_SUMMARY,
            "knowledge": cls.L5_KNOWLEDGE,
            "raw": cls.L1_RAW,
        }
        if value in v1_mapping:
            return v1_mapping[value]
        for layer in cls:
            if layer.value == value:
                return layer
        raise ValueError(f"Invalid memory layer: {value}")

    @classmethod
    def all_layers(cls) -> List["MemoryLayer"]:
        """Return all memory layers (aliases excluded)"""
        return [
            cls.L0_BASIC_INFO,
            cls.L1_RAW, cls.L2_FACT, cls.L3_SUMMARY,
            cls.L4_IDENTITY, cls.L5_KNOWLEDGE,
            cls.L6_SCHEMA, cls.L7_INTENTION,
        ]



class ContentType(str, Enum):
    """Memory content types (kept for legacy compat; MemoryNode no longer uses this field)"""
    FACT = "fact"
    SUMMARY = "summary"
    ENTITY = "entity"
    SCHEMA = "schema"
    PROFILE = "profile"
    INTENTION = "intention"
    RAW = "raw"
    HYPOTHESIS = "hypothesis"


class MemoryStatus(str, Enum):
    """Memory status."""
    ACTIVE = "active"               # Currently valid
    SUPERSEDED = "superseded"       # Replaced by a newer version (old node on an EVOLVE evolution chain)
    NEGATED = "negated"             # Negated
    CONFLICTED = "conflicted"       # In conflict
    SHADOW = "shadow"               # Shadow state: low-confidence inference or reconcile DELETE soft delete
                                    # Neither case takes part in retrieval (is_latest is usually False)
    ARCHIVED = "archived"           # Archived


class SourceType(str, Enum):
    """Information source types."""
    EXPLICIT = "explicit"           # Stated explicitly by the user
    INFERRED = "inferred"           # Inferred by the system
    COMPOSITE = "composite"         # Synthesis over multiple pieces of evidence


class UpdateType(str, Enum):
    """Update types for versioned facts."""
    OVERRIDE = "override"           # New value replaces the old one
    SUPPLEMENT = "supplement"       # Supplementary information
    TEMPORAL = "temporal"           # Time window changed
    NEGATE = "negate"               # Negates the old information
    CONFLICT = "conflict"           # Conflict between old and new


class SchemaStatus(str, Enum):
    """Schema status."""
    FORMING = "forming"             # Forming (insufficient evidence)
    STABLE = "stable"               # Stable (supported by multiple pieces of evidence)
    EVOLVING = "evolving"           # Evolving (new evidence modifies it)
    DEPRECATED = "deprecated"       # Deprecated


class TriggerType(str, Enum):
    """Intention trigger types."""
    TIME_BASED = "time_based"       # Time trigger
    EVENT_BASED = "event_based"     # Event trigger


class IntentionPriority(str, Enum):
    """Intention priority."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MetaCognitionTag(str, Enum):
    """
    Meta-cognition signal tags

    Attached to memory entries so the calling LLM can sense how reliable a memory is and what is special about it.
    """
    # Confidence
    HIGH_CONFIDENCE = "high_confidence"
    LOW_CONFIDENCE = "low_confidence"
    INFERRED_NOT_STATED = "inferred_not_stated"
    STALE = "stale"

    # Conflict and uncertainty
    NEEDS_CLARIFICATION = "needs_clarification"
    CONFLICTED = "conflicted"
    CONTEXT_DEPENDENT = "context_dependent"

    # Emotion and sensitivity
    EMOTIONALLY_SENSITIVE = "emotionally_sensitive"
    POSITIVE_MEMORY = "positive_memory"
    NEGATIVE_MEMORY = "negative_memory"
    GOTCHA = "gotcha"

    # Temporal specifics
    RECENTLY_CHANGED = "recently_changed"
    HISTORICAL = "historical"
    TEMPORAL_CONTEXT = "temporal_context"

    # Knowledge gaps
    KNOWLEDGE_GAP = "knowledge_gap"
    INCOMPLETE = "incomplete"

    # Value tags
    LONG_TAIL = "long_tail"
    SCHEMA_ACTIVATED = "schema_activated"

    # Prospective tags
    PROACTIVE_CARE = "proactive_care"
    FOLLOW_UP = "follow_up"


class GapType(str, Enum):
    """The four kinds of knowledge gaps."""
    PROFILE_INCOMPLETE = "profile_incomplete"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    INFERRED_UNVERIFIED = "inferred_unverified"
    STALE_UNVERIFIED = "stale_unverified"


# ============================================================
# Core model: MemoryNode (unified base class)
# ============================================================

@dataclass
class MemoryNode:
    """
    Unified base class for memory at every layer (V2)

    Field groups:
    - Identity (node_id, user_id, layer, content)
    - Business isolation (agent_id, session_id)
    - Time and space (memory_at, temporal_anchor, valid_from/until, gmt_created/modified)
    - Evolution graph (supersedes, superseded_by, is_latest) -- EVOLVE semantics
    - Inference annotation (speculate) -- excluded from embedding
    - Summary anchor (source_raw_memory_id) -- specific to L3_SUMMARY/L4_IDENTITY
    - Status (status)
    - Confidence and calibration (confidence, evidence_count, source_type)
    - Emotion tags (emotional_valence, emotional_arousal)
    - Retrieval metadata (access_count, last_accessed_at, embedding)
    - Long-tail tags (specificity_score, rarity_score, longtail_flag)
    - Meta-cognition tags (meta_tags, meta_hints)
    - Provenance (source_session_id, source_turn_index, evidence_chain)
    - Extensions (custom, tags)
    """

    # === Identity ===
    node_id: str = ""
    user_id: str = ""
    layer: MemoryLayer = MemoryLayer.L1_RAW
    content: str = ""                        # Core semantic content; embedding uses only this field

    # === Business isolation ===
    agent_id: str = ""
    session_id: str = ""
    # Call source that produced this memory (openclaw / cursor / codex, etc.). Do not mix it with
    # source_type (explicit/inferred/composite) or with retrieval channels.
    source: str = "default"

    # === Ownership (owner) ===
    # Who this memory belongs to: 'user' (about the user / stated by the user) | 'agent' (assistant input / what the user asks the agent to do).
    # Only written for L2_FACT / L7_INTENTION; other layers (L0/L1/L3/L5/L6) stay empty (None).
    owner: Optional[str] = None

    # === Time and space ===
    memory_at: Optional[datetime] = None     # When the remembered event happened (caller-supplied, may be empty)
    temporal_anchor: Optional[str] = None    # Kept for compatibility with historical data; not written since v0.3.45_v0 post8
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # === Data management timestamps ===
    gmt_created: Optional[datetime] = None   # Row write time (set automatically by the system)
    gmt_modified: Optional[datetime] = None  # Last modification time (set automatically by the system)

    # === Evolution graph (EVOLVE semantics) ===
    supersedes: Optional[List[str]] = None       # Old node IDs this node supersedes; None for brand-new information
    superseded_by: Optional[List[str]] = None    # Which newer nodes superseded it (written by the system, not exposed to callers)
    is_latest: bool = True                       # Whether this is the chain tip; the system sets it to False when superseded

    # === Inference annotation (excluded from embedding) ===
    speculate: Optional[str] = None              # LLM inference annotation for complex or ambiguous signals

    # === Summary anchor (L3_SUMMARY / L4_IDENTITY) ===
    source_raw_memory_id: Optional[str] = None   # Corresponding L1_RAW node ID

    # === Status ===
    status: MemoryStatus = MemoryStatus.ACTIVE

    # === Confidence and calibration ===
    confidence: float = 1.0
    evidence_count: int = 1
    source_type: SourceType = SourceType.EXPLICIT

    # === Emotion tags ===
    emotional_valence: float = 0.0      # [-1, 1] emotional valence
    emotional_arousal: float = 0.0      # [0, 1] emotional arousal

    # === Retrieval metadata ===
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    embedding: Optional[List[float]] = None

    # === Long-tail tags ===
    specificity_score: float = 0.0      # [0, 1] specificity
    rarity_score: float = 0.0           # [0, 1] rarity
    longtail_flag: bool = False

    # === Meta-cognition tags ===
    meta_tags: List[MetaCognitionTag] = field(default_factory=list)
    meta_hints: Dict[str, str] = field(default_factory=dict)

    # === Episode (internal session) ===
    # One matter usually spans several write calls (a dozen or twenty turns); memories extracted from them belong to one discussion.
    # Tagging them with the same episode_id lets the read side **bound a candidate set** for weighting -- far cheaper
    # than building O(n²) temporal edges, and more accurate than "same day".
    #
    # Note the difference from session_id: session_id is a **caller-supplied business isolation key** (part of isolation_key),
    # possibly very long (a whole conversation) or simply default; episode_id is one the SDK cuts **by time interval**
    # into a "single discussion", so it is finer-grained and takes no part in isolation.
    episode_id: str = ""

    # === Provenance ===
    source_session_id: str = ""
    source_turn_index: Optional[int] = None
    evidence_chain: List[str] = field(default_factory=list)

    # === Extension fields ===
    custom: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    # NER labels (GPE/PERSON/...) aligned with tags; empty string for topic words. Tagged at write time, used for read-side boosts.
    tag_ners: List[str] = field(default_factory=list)

    def __post_init__(self):
        from ..utils.source_filter import normalize_source
        self.source = normalize_source(self.source)
        if not self.node_id:
            self.node_id = str(uuid.uuid4())
        now = datetime.now()
        # gmt_created/gmt_modified are system timestamps, filled automatically
        if self.gmt_created is None:
            self.gmt_created = now
        if self.gmt_modified is None:
            self.gmt_modified = now
        # memory_at is when the event happened, supplied by the caller, never filled automatically
        # valid_from defaults to gmt_created (data is valid from the moment it is written)
        if self.valid_from is None:
            self.valid_from = now

    @property
    def memory_id(self) -> str:
        """V1 compat: memory_id == node_id"""
        return self.node_id

    @property
    def uid(self) -> str:
        """V1 compat: uid == user_id"""
        return self.user_id

    def get_isolation_key(self) -> str:
        """Isolation key: {user_id}::{agent_id}::{session_id}

        An empty session_id falls back to 'default_session', aligned with build_isolation_key.
        """
        sid = self.session_id or "default_session"
        return f"{self.user_id}::{self.agent_id}::{sid}"

    def get_topic_scope_key(self) -> str:
        """Topic isolation key: {user_id}::{agent_id} -- **excludes session_id**.

        Why a Topic is wider than a memory: topics are **accumulated across sessions**. "What I know about your
        health" should not be cut apart per session, while `get_isolation_key()` carries a session, and using it as
        the Topic isolation key makes every session build its own unrelated Topic node per tag name.

        Isolating by session splits same-name Topics into several nodes, so the temporal spine cannot build edges,
        topic summaries keep re-summarizing same-name fragments, and cross-session tag recall stays disconnected.
        """
        return f"{self.user_id}::{self.agent_id}"

    @staticmethod
    def build_topic_scope_key(uid: str, agent_id: str) -> str:
        """Build the Topic isolation key: {uid}::{agent_id} (no session, see get_topic_scope_key)"""
        return f"{uid}::{agent_id}"

    @staticmethod
    def topic_scope_of(isolation_key: str) -> str:
        """Trim a session-bearing isolation_key down to the Topic scope (first two segments).

        For read paths that only have an isolation_key -- creating and looking up a Topic must use the same key,
        otherwise the lookup misses (both backends match `isolation_key` exactly).
        """
        if not isolation_key:
            return isolation_key
        parts = isolation_key.split("::")
        return "::".join(parts[:2]) if len(parts) >= 2 else isolation_key

    @staticmethod
    def build_isolation_key(uid: str, agent_id: str, session_id: str = "default_session") -> str:
        """Build the isolation key: {uid}::{agent_id}::{session_id}"""
        sid = session_id or "default_session"
        return f"{uid}::{agent_id}::{sid}"

    @staticmethod
    def parse_isolation_key(key: str) -> tuple:
        """
        Parse an isolation key into (user_id, agent_id, session_id).

        Accepts several formats:
        - 4 parts: (legacy format with appid) skip the first part → uid::agent_id::session_id
        - 3 parts: uid::agent_id::session_id (standard format)
        - 2 parts: uid::agent_id (session_id defaults to "default_session")
        """
        parts = key.split("::")
        if len(parts) >= 4:
            # Legacy format compat: skip the appid
            return parts[1], parts[2], parts[3]
        elif len(parts) == 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return parts[0], parts[1], "default_session"
        return "", "", ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict"""
        return {
            "node_id": self.node_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "source": self.source,
            "episode_id": self.episode_id,
            "owner": self.owner,
            "layer": self.layer.value,
            "content": self.content,
            "memory_at": int(self.memory_at.timestamp()) if self.memory_at else None,
            "temporal_anchor": self.temporal_anchor,
            "gmt_created": int(self.gmt_created.timestamp()) if self.gmt_created else None,
            "gmt_modified": int(self.gmt_modified.timestamp()) if self.gmt_modified else None,
            "valid_from": int(self.valid_from.timestamp()) if self.valid_from else None,
            "valid_until": int(self.valid_until.timestamp()) if self.valid_until else None,
            # Evolution graph
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "is_latest": self.is_latest,
            # Inference annotation
            "speculate": self.speculate,
            # Summary anchor
            "source_raw_memory_id": self.source_raw_memory_id,
            # Status
            "status": self.status.value,
            # Confidence and calibration
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "source_type": self.source_type.value,
            # Emotion
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            # Retrieval metadata
            "access_count": self.access_count,
            "last_accessed_at": int(self.last_accessed_at.timestamp()) if self.last_accessed_at else None,
            # Long-tail tags
            "specificity_score": self.specificity_score,
            "rarity_score": self.rarity_score,
            "longtail_flag": self.longtail_flag,
            # Meta-cognition
            "meta_tags": [t.value for t in self.meta_tags],
            "meta_hints": self.meta_hints,
            # Provenance
            "source_session_id": self.source_session_id,
            "source_turn_index": self.source_turn_index,
            "evidence_chain": self.evidence_chain,
            # Extensions
            "custom": self.custom,
            "tags": self.tags,
            "tag_ners": list(getattr(self, "tag_ners", None) or []),
            # Vector (optional, must be carried for clone / serialization)
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """Deserialize from a dict (deprecated fields in old payloads are ignored)"""
        def parse_dt(v):
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v)
            if isinstance(v, str):
                return datetime.fromisoformat(v)
            return None

        def parse_enum(enum_cls, v, default):
            if v is None:
                return default
            if isinstance(v, enum_cls):
                return v
            try:
                return enum_cls(v)
            except (ValueError, KeyError):
                return default

        meta_tags_raw = data.get("meta_tags", [])
        meta_tags = []
        for t in meta_tags_raw:
            if isinstance(t, MetaCognitionTag):
                meta_tags.append(t)
            else:
                try:
                    meta_tags.append(MetaCognitionTag(t))
                except (ValueError, KeyError):
                    pass

        return cls(
            node_id=data.get("node_id", data.get("memory_id", "")),
            user_id=data.get("user_id", data.get("uid", "")),
            agent_id=data.get("agent_id", ""),
            session_id=data.get("session_id", ""),
            source=data.get("source", "default") or "default",
            episode_id=data.get("episode_id", "") or "",
            owner=data.get("owner") or None,
            layer=parse_enum(MemoryLayer, data.get("layer"), MemoryLayer.L1_RAW),
            content=data.get("content", ""),
            # memory_at: prefer memory_at, fall back to the legacy key created_at
            memory_at=parse_dt(data.get("memory_at") or data.get("created_at")),
            temporal_anchor=data.get("temporal_anchor"),
            gmt_created=parse_dt(data.get("gmt_created")),
            gmt_modified=parse_dt(data.get("gmt_modified")),
            valid_from=parse_dt(data.get("valid_from")),
            valid_until=parse_dt(data.get("valid_until")),
            # Evolution graph
            supersedes=data.get("supersedes"),
            superseded_by=data.get("superseded_by"),
            is_latest=data.get("is_latest", True),
            # Inference annotation
            speculate=data.get("speculate"),
            # Summary anchor
            source_raw_memory_id=data.get("source_raw_memory_id"),
            # Status
            status=parse_enum(MemoryStatus, data.get("status"), MemoryStatus.ACTIVE),
            # Confidence
            confidence=data.get("confidence", 1.0),
            evidence_count=data.get("evidence_count", 1),
            source_type=parse_enum(SourceType, data.get("source_type"), SourceType.EXPLICIT),
            # Emotion
            emotional_valence=data.get("emotional_valence", 0.0),
            emotional_arousal=data.get("emotional_arousal", 0.0),
            # Retrieval metadata
            access_count=data.get("access_count", 0),
            last_accessed_at=parse_dt(data.get("last_accessed_at")),
            embedding=data.get("embedding"),
            # Long tail
            specificity_score=data.get("specificity_score", 0.0),
            rarity_score=data.get("rarity_score", 0.0),
            longtail_flag=data.get("longtail_flag", False),
            # Meta-cognition
            meta_tags=meta_tags,
            meta_hints=data.get("meta_hints", {}),
            # Provenance
            source_session_id=data.get("source_session_id", ""),
            source_turn_index=data.get("source_turn_index"),
            evidence_chain=data.get("evidence_chain", []),
            # Extensions
            custom=data.get("custom", {}),
            tags=data.get("tags", []) or [],
            tag_ners=list(data.get("tag_ners") or []),
        )


# ============================================================
# Subclass models
# ============================================================

@dataclass
class VersionedFact(MemoryNode):
    """
    Atomic fact at the L2 layer -- versioned immutable record

    V2 §3.2: fact content is never edited in place; changes are tracked via status transitions + SUPERSEDED_BY edges.
    """
    update_type: Optional[UpdateType] = None

    def __post_init__(self):
        super().__post_init__()
        if self.layer == MemoryLayer.L1_RAW:
            self.layer = MemoryLayer.L2_FACT

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["update_type"] = self.update_type.value if self.update_type else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionedFact":
        base = MemoryNode.from_dict(data)
        update_type = None
        if data.get("update_type"):
            try:
                update_type = UpdateType(data["update_type"])
            except (ValueError, KeyError):
                pass
        return cls(**{**base.__dict__, "update_type": update_type})


@dataclass
class SchemaNode(MemoryNode):
    """
    L6 layer: mental model -- abstract narrative template induced automatically from related facts (Graph layer)

    V3 §2.2
    """
    central_proposition: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    expected_inferences: List[str] = field(default_factory=list)
    activation_threshold: float = 0.7
    activation_count: int = 0
    last_activated_at: Optional[datetime] = None
    schema_status: SchemaStatus = SchemaStatus.FORMING

    def __post_init__(self):
        super().__post_init__()
        self.layer = MemoryLayer.L6_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "central_proposition": self.central_proposition,
            "supporting_evidence": self.supporting_evidence,
            "expected_inferences": self.expected_inferences,
            "activation_threshold": self.activation_threshold,
            "activation_count": self.activation_count,
            "last_activated_at": int(self.last_activated_at.timestamp()) if self.last_activated_at else None,
            "schema_status": self.schema_status.value,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaNode":
        base = MemoryNode.from_dict(data)
        def parse_dt(v):
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v)
            if isinstance(v, str):
                return datetime.fromisoformat(v)
            return None
        schema_status = SchemaStatus.FORMING
        if data.get("schema_status"):
            try:
                schema_status = SchemaStatus(data["schema_status"])
            except (ValueError, KeyError):
                pass
        return cls(
            **{**base.__dict__,
               "central_proposition": data.get("central_proposition", ""),
               "supporting_evidence": data.get("supporting_evidence", []),
               "expected_inferences": data.get("expected_inferences", []),
               "activation_threshold": data.get("activation_threshold", 0.7),
               "activation_count": data.get("activation_count", 0),
               "last_activated_at": parse_dt(data.get("last_activated_at")),
               "schema_status": schema_status,
               })


@dataclass
class IntentionNode(MemoryNode):
    """
    L7 layer: prospective memory -- remembering "what to do later" (Graph layer)

    V3 §2.2
    """
    trigger_type: TriggerType = TriggerType.EVENT_BASED
    trigger_condition: str = ""

    # Time trigger
    trigger_time: Optional[datetime] = None

    # Event trigger
    trigger_event_pattern: Optional[str] = None
    trigger_event_embedding: Optional[List[float]] = None

    # Intention content
    intention_content: str = ""
    priority: IntentionPriority = IntentionPriority.MEDIUM
    expiry: Optional[datetime] = None
    triggered: bool = False
    triggered_at: Optional[datetime] = None

    def __post_init__(self):
        super().__post_init__()
        self.layer = MemoryLayer.L7_INTENTION

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "trigger_type": self.trigger_type.value,
            "trigger_condition": self.trigger_condition,
            "trigger_time": int(self.trigger_time.timestamp()) if self.trigger_time else None,
            "trigger_event_pattern": self.trigger_event_pattern,
            "intention_content": self.intention_content,
            "priority": self.priority.value,
            "expiry": int(self.expiry.timestamp()) if self.expiry else None,
            "triggered": self.triggered,
            "triggered_at": int(self.triggered_at.timestamp()) if self.triggered_at else None,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentionNode":
        base = MemoryNode.from_dict(data)
        def parse_dt(v):
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v)
            if isinstance(v, str):
                return datetime.fromisoformat(v)
            return None
        trigger_type = TriggerType.EVENT_BASED
        if data.get("trigger_type"):
            try:
                trigger_type = TriggerType(data["trigger_type"])
            except (ValueError, KeyError):
                pass
        priority = IntentionPriority.MEDIUM
        if data.get("priority"):
            try:
                priority = IntentionPriority(data["priority"])
            except (ValueError, KeyError):
                pass
        return cls(
            **{**base.__dict__,
               "trigger_type": trigger_type,
               "trigger_condition": data.get("trigger_condition", ""),
               "trigger_time": parse_dt(data.get("trigger_time")),
               "trigger_event_pattern": data.get("trigger_event_pattern"),
               "trigger_event_embedding": data.get("trigger_event_embedding"),
               "intention_content": data.get("intention_content", ""),
               "priority": priority,
               "expiry": parse_dt(data.get("expiry")),
               "triggered": data.get("triggered", False),
               "triggered_at": parse_dt(data.get("triggered_at")),
               })


# ============================================================
# Auxiliary model: KnowledgeGap
# ============================================================

@dataclass
class KnowledgeGap:
    """
    Knowledge gap record -- "information missing or unreliable on some dimension"

    V2 §5.6.2
    """
    gap_id: str = ""
    user_id: str = ""
    domain: str = ""
    gap_type: GapType = GapType.PROFILE_INCOMPLETE
    importance: str = "medium"          # high / medium / low
    description: str = ""
    related_node_ids: List[str] = field(default_factory=list)
    hint: str = ""
    created_at: Optional[datetime] = None
    last_scanned_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolve_trigger: Optional[str] = None

    def __post_init__(self):
        if not self.gap_id:
            self.gap_id = f"gap_{uuid.uuid4().hex[:12]}"
        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        if self.last_scanned_at is None:
            self.last_scanned_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "user_id": self.user_id,
            "domain": self.domain,
            "gap_type": self.gap_type.value,
            "importance": self.importance,
            "description": self.description,
            "related_node_ids": self.related_node_ids,
            "hint": self.hint,
            "created_at": int(self.created_at.timestamp()) if self.created_at else None,
            "last_scanned_at": int(self.last_scanned_at.timestamp()) if self.last_scanned_at else None,
            "resolved": self.resolved,
            "resolved_at": int(self.resolved_at.timestamp()) if self.resolved_at else None,
            "resolve_trigger": self.resolve_trigger,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGap":
        def parse_dt(v):
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v)
            if isinstance(v, str):
                return datetime.fromisoformat(v)
            return None
        gap_type = GapType.PROFILE_INCOMPLETE
        if data.get("gap_type"):
            try:
                gap_type = GapType(data["gap_type"])
            except (ValueError, KeyError):
                pass
        return cls(
            gap_id=data.get("gap_id", ""),
            user_id=data.get("user_id", ""),
            domain=data.get("domain", ""),
            gap_type=gap_type,
            importance=data.get("importance", "medium"),
            description=data.get("description", ""),
            related_node_ids=data.get("related_node_ids", []),
            hint=data.get("hint", ""),
            created_at=parse_dt(data.get("created_at")),
            last_scanned_at=parse_dt(data.get("last_scanned_at")),
            resolved=data.get("resolved", False),
            resolved_at=parse_dt(data.get("resolved_at")),
            resolve_trigger=data.get("resolve_trigger"),
        )


# ============================================================
# Auxiliary models: temporal system (Temporal)
# ============================================================

@dataclass
class LifeStage:
    """One life stage of the user."""
    name: str = ""                      # "college years", "working in Shanghai"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_time": int(self.start_time.timestamp()) if self.start_time else None,
            "end_time": int(self.end_time.timestamp()) if self.end_time else None,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LifeStage":
        def parse_dt(v):
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v)
            if isinstance(v, str):
                return datetime.fromisoformat(v)
            return None
        return cls(
            name=data.get("name", ""),
            start_time=parse_dt(data.get("start_time")),
            end_time=parse_dt(data.get("end_time")),
            description=data.get("description", ""),
        )


@dataclass
class TemporalEvent:
    """
    One event on the timeline

    V2 §6.1
    """
    event_id: str = ""                  # Associated fact_id
    event_description: str = ""
    event_time: Optional[datetime] = None
    time_precision: str = "approximate"  # exact / month / quarter / year / approximate
    time_description: str = ""          # Original time expression
    life_stage: Optional[str] = None
    is_milestone: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_description": self.event_description,
            "event_time": int(self.event_time.timestamp()) if self.event_time else None,
            "time_precision": self.time_precision,
            "time_description": self.time_description,
            "life_stage": self.life_stage,
            "is_milestone": self.is_milestone,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalEvent":
        def parse_dt(v):
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            if isinstance(v, (int, float)):
                return datetime.fromtimestamp(v)
            if isinstance(v, str):
                return datetime.fromisoformat(v)
            return None
        return cls(
            event_id=data.get("event_id", ""),
            event_description=data.get("event_description", ""),
            event_time=parse_dt(data.get("event_time")),
            time_precision=data.get("time_precision", "approximate"),
            time_description=data.get("time_description", ""),
            life_stage=data.get("life_stage"),
            is_milestone=data.get("is_milestone", False),
        )


@dataclass
class UserTimeline:
    """
    The user's complete timeline

    V2 §6.1
    """
    user_id: str = ""
    events: List[TemporalEvent] = field(default_factory=list)
    life_stages: List[LifeStage] = field(default_factory=list)
    milestone_events: Dict[str, TemporalEvent] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "events": [e.to_dict() for e in self.events],
            "life_stages": [s.to_dict() for s in self.life_stages],
            "milestone_events": {k: v.to_dict() for k, v in self.milestone_events.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserTimeline":
        return cls(
            user_id=data.get("user_id", ""),
            events=[TemporalEvent.from_dict(e) for e in data.get("events", [])],
            life_stages=[LifeStage.from_dict(s) for s in data.get("life_stages", [])],
            milestone_events={
                k: TemporalEvent.from_dict(v)
                for k, v in data.get("milestone_events", {}).items()
            },
        )


# ============================================================
# Output protocol: MemoryContextPackage
# ============================================================

@dataclass
class ProfileSummary:
    """User profile summary (Layer 0, always returned)"""
    core: str = ""                      # "Alex, 30, Berlin, backend engineer"
    personality: str = ""               # "pragmatic, tech-driven"
    active_schemas: List[str] = field(default_factory=list)
    gotchas: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core": self.core,
            "personality": self.personality,
            "active_schemas": self.active_schemas,
            "gotchas": self.gotchas,
            "note": self.note,
        }


@dataclass
class MemoryIndexEntry:
    """Index view entry (Layer 1, ~100 tokens)"""
    node_id: str = ""
    date: str = ""
    content_type: str = ""
    title: str = ""
    relevance: float = 0.0
    meta_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "date": self.date,
            "type": self.content_type,
            "title": self.title,
            "relevance": self.relevance,
            "meta_tags": self.meta_tags,
        }


@dataclass
class MemorySummaryEntry:
    """Summary view entry (Layer 2, ~500 tokens)"""
    node_id: str = ""
    summary: str = ""
    confidence: float = 0.0
    meta_tags: List[str] = field(default_factory=list)
    meta_hints: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "summary": self.summary,
            "confidence": self.confidence,
            "meta_tags": self.meta_tags,
            "meta_hints": self.meta_hints,
        }


@dataclass
class MetaCognitionReport:
    """Meta-cognition report."""
    confidence_summary: str = ""
    knowledge_gaps: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    stale_memories: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_summary": self.confidence_summary,
            "knowledge_gaps": self.knowledge_gaps,
            "unresolved_conflicts": self.unresolved_conflicts,
            "stale_memories": self.stale_memories,
        }


@dataclass
class TriggeredIntention:
    """A prospective intention that was triggered."""
    trigger_reason: str = ""
    intention: str = ""
    priority: str = "medium"
    meta_tags: List[str] = field(default_factory=list)
    source_intention_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_reason": self.trigger_reason,
            "intention": self.intention,
            "priority": self.priority,
            "meta_tags": self.meta_tags,
            "source_intention_id": self.source_intention_id,
        }


@dataclass
class ActivatedSchema:
    """An activated Schema."""
    schema: str = ""
    evidence_summary: str = ""
    expected_inferences: List[str] = field(default_factory=list)
    meta_tags: List[str] = field(default_factory=list)
    source_schema_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_summary": self.evidence_summary,
            "expected_inferences": self.expected_inferences,
            "meta_tags": self.meta_tags,
            "source_schema_id": self.source_schema_id,
        }


@dataclass
class TemporalContext:
    """Timeline context (returned when the query involves time)"""
    relevant_timeline: List[Dict[str, str]] = field(default_factory=list)
    temporal_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevant_timeline": self.relevant_timeline,
            "temporal_note": self.temporal_note,
        }


@dataclass
class MemoryContextPackage:
    """
    Complete memory context package returned to the calling LLM

    V2 §7.1: three-tier progressive disclosure + meta-cognition signals + prospective intentions + Schema + timeline
    """
    # Layer 0: user profile summary (always returned)
    profile_summary: ProfileSummary = field(default_factory=ProfileSummary)

    # Layer 1: index view (~100 tokens)
    index_view: List[MemoryIndexEntry] = field(default_factory=list)

    # Layer 2: summary view (~500 tokens, expanded on demand)
    summary_view: List[MemorySummaryEntry] = field(default_factory=list)

    # Layer 3: full content (lazily loaded on demand)
    full_content_ids: List[str] = field(default_factory=list)

    # Meta-cognition signals
    meta_cognition: MetaCognitionReport = field(default_factory=MetaCognitionReport)

    # Prospective intentions
    triggered_intentions: List[TriggeredIntention] = field(default_factory=list)

    # Activated Schemas
    activated_schemas: List[ActivatedSchema] = field(default_factory=list)

    # Timeline context
    temporal_context: Optional[TemporalContext] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_summary": self.profile_summary.to_dict(),
            "index_view": [e.to_dict() for e in self.index_view],
            "summary_view": [e.to_dict() for e in self.summary_view],
            "full_content_ids": self.full_content_ids,
            "meta_cognition": self.meta_cognition.to_dict(),
            "triggered_intentions": [t.to_dict() for t in self.triggered_intentions],
            "activated_schemas": [s.to_dict() for s in self.activated_schemas],
            "temporal_context": self.temporal_context.to_dict() if self.temporal_context else None,
        }


# ============================================================
# V1 compat type aliases
# ============================================================

# MemoryEntry aliases MemoryNode so legacy code keeps importing it without errors
MemoryEntry = MemoryNode


@dataclass
class MemoryMetadata:
    """
    V1 compat: memory metadata

    New code should use MemoryNode fields directly; this class is kept for compatibility with legacy code.
    """
    uid: str = ""
    agent_id: str = ""
    layer: MemoryLayer = MemoryLayer.L1_RAW

    event_time: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    importance: float = 1.0
    recency_weight: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    source: Optional[str] = None
    parent_id: Optional[str] = None
    related_ids: List[str] = field(default_factory=list)

    custom: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        if self.event_time is None:
            self.event_time = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "agent_id": self.agent_id,
            "layer": self.layer.value,
            "event_time": int(self.event_time.timestamp()) if self.event_time else None,
            "created_at": int(self.created_at.timestamp()) if self.created_at else None,
            "updated_at": int(self.updated_at.timestamp()) if self.updated_at else None,
            "expires_at": int(self.expires_at.timestamp()) if self.expires_at else None,
            "importance": self.importance,
            "recency_weight": self.recency_weight,
            "access_count": self.access_count,
            "last_accessed": int(self.last_accessed.timestamp()) if self.last_accessed else None,
            "source": self.source,
            "parent_id": self.parent_id,
            "related_ids": self.related_ids,
            "custom": self.custom,
            "tags": self.tags,
            "tag_ners": list(self.tag_ners or []),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMetadata":
        def parse_datetime(value):
            if value is None:
                return None
            if isinstance(value, datetime):
                return value
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value)
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            return None

        layer = data.get("layer", MemoryLayer.L1_RAW)
        if isinstance(layer, str):
            layer = MemoryLayer.from_string(layer)

        return cls(
            uid=data.get("uid", ""),
            agent_id=data.get("agent_id", ""),
            layer=layer,
            event_time=parse_datetime(data.get("event_time")),
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
            expires_at=parse_datetime(data.get("expires_at")),
            importance=data.get("importance", 1.0),
            recency_weight=data.get("recency_weight", 1.0),
            access_count=data.get("access_count", 0),
            last_accessed=parse_datetime(data.get("last_accessed")),
            source=data.get("source"),
            parent_id=data.get("parent_id"),
            related_ids=data.get("related_ids", []),
            custom=data.get("custom", {}),
            tags=data.get("tags", []),
        )


@dataclass
class MemoryScore:
    """
    V1 compat: memory scoring

    V2 uses a more complex scoring formula (see core/scorer.py); this class is kept for compatibility.
    """
    semantic_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    access_score: float = 0.0

    semantic_weight: float = 0.5
    recency_weight: float = 0.3
    importance_weight: float = 0.15
    access_weight: float = 0.05

    @property
    def final_score(self) -> float:
        return (
            self.semantic_score * self.semantic_weight
            + self.recency_score * self.recency_weight
            + self.importance_score * self.importance_weight
            + self.access_score * self.access_weight
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_score": self.semantic_score,
            "recency_score": self.recency_score,
            "importance_score": self.importance_score,
            "access_score": self.access_score,
            "final_score": self.final_score,
            "weights": {
                "semantic": self.semantic_weight,
                "recency": self.recency_weight,
                "importance": self.importance_weight,
                "access": self.access_weight,
            },
        }
