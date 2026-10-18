"""
Basic profile (L0_BASIC_INFO) upsert module.

Merges the user's stable structured attributes (name / age / location / occupation / employer, configurable)
into the L0_BASIC_INFO memory evolution chain.

Properties:
- Each update stores only the diff fields and creates a new version node.
- Old versions are marked superseded and linked through the supersedes/superseded_by chain.
- Users read L0_BASIC_INFO separately via get_user_profile(); Search no longer returns L0.
- The field table (fields schema) is configured by MemoryConfig.basic_profile.fields.

Design change (2026-06):
- Removed the LLM function-calling tool `update_basic_user_profile` (weak models filled it in arbitrarily).
- Instead, the extractor prompt asks the LLM to return basic_info directly in its JSON output, and the
  writer calls this module's upsert_basic_profile() once it receives basic_info.
- The field table is configurable, so behaviour is not pinned to 5 fixed fields.

Public API:
- upsert_basic_profile(...)        — main entry, called directly by the writer
- basic_info_node_id(...)          — deterministic id (still kept, for fallback lookups)
- render_l0_evolution_chain(...)   — renders the L0 evolution chain
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import weakref
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ================================================================
# Default field order (controls how _render_content concatenates fields)
# Note: which fields are actually accepted is decided by the allowed_fields the caller passes in.
# ================================================================

BASIC_FIELDS: List[str] = ["name", "age", "location", "occupation", "employer"]


# ================================================================
# Deterministic node id (kept so historical data can still be looked up as a fallback)
# ================================================================

_BASIC_INFO_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000babb1e")  # stable sentinel
USER_LEVEL_AGENT_ID = ""
USER_LEVEL_SESSION_ID = ""

# Serializes L0 updates for one user inside a process and event loop, so concurrent requests cannot create
# several latest chain heads at once. Locks are bucketed per event loop, so several clients/loops never share one Lock.
# The VDB has no cross-process transaction or CAS, so the read side still defensively folds history and cross-replica forks.
_LOOP_USER_LOCKS = weakref.WeakKeyDictionary()


def _user_lock(user_id: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _LOOP_USER_LOCKS.get(loop)
    if locks is None:
        locks = weakref.WeakValueDictionary()
        _LOOP_USER_LOCKS[loop] = locks
    lock = locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[user_id] = lock
    return lock


def basic_info_node_id(user_id: str) -> str:
    """Return the user-level L0 deterministic id derived from user_id alone."""
    key = f"basic_info::{user_id or 'default'}"
    return str(uuid.uuid5(_BASIC_INFO_NAMESPACE, key))


# ================================================================
# Content rendering
# ================================================================

def _render_content(kv: Dict[str, Any], field_order: Optional[List[str]] = None) -> str:
    """
    Render a KV mapping into a human-readable natural-language string.
    Stays consistent with the "The user's ..." style of other memories.

    field_order: optional, controls the concatenation order; defaults to kv's lexicographic order.
    """
    order = field_order or sorted(kv.keys())
    parts: List[str] = []
    for key in order:
        if key not in kv:
            continue
        value = kv[key]
        if value is None:
            continue
        s = str(value).strip()
        if not s:
            continue
        parts.append(f"{key} is {s}")
    if not parts:
        return ""
    return "The user's " + ", ".join(parts) + "."


def render_l0_evolution_chain(l0_nodes: list) -> str:
    """
    Show all historical L0 versions grouped by attribute, in chronological order.

    Input: a list of L0_BASIC_INFO nodes (including superseded older versions)
    Output: a formatted string, for example:
        User name: Alice
        User age: (2026-01-15 user said age 25) → (2026-05-10 user said age 26)
        User occupation: (2026-01-15 user said they are a programmer) → (2026-03-20 user said they are an architect)
    """
    from datetime import datetime as _dt

    # Sort chronologically ascending
    sorted_nodes = sorted(
        l0_nodes,
        key=lambda n: (getattr(n, 'gmt_created', None) or getattr(n, 'valid_from', None) or _dt.min),
    )

    # Collect every historical version per attribute (keys discovered dynamically, not fixed to BASIC_FIELDS)
    field_history: Dict[str, List[tuple]] = {}
    for node in sorted_nodes:
        kv = {}
        if hasattr(node, 'custom') and isinstance(node.custom, dict):
            kv = node.custom.get("basic_info_kv") or {}
        ts = getattr(node, 'gmt_created', None) or getattr(node, 'valid_from', None)
        ts_str = ts.strftime("%Y-%m-%d") if ts else "unknown"
        for field_name, val in kv.items():
            field_history.setdefault(field_name, []).append((ts_str, val))

    # Rendering: the 5 built-in fields use friendly labels, everything else uses the raw key
    field_labels = {
        "name": "User name",
        "age": "User age",
        "location": "User location",
        "occupation": "User occupation",
        "employer": "User employer",
    }
    # Order: known BASIC_FIELDS first, then the remainder alphabetically
    ordered_keys: List[str] = [k for k in BASIC_FIELDS if k in field_history]
    for k in sorted(field_history.keys()):
        if k not in ordered_keys:
            ordered_keys.append(k)

    lines = []
    for field_name in ordered_keys:
        history = field_history[field_name]
        if not history:
            continue
        label = field_labels.get(field_name, field_name)
        if len(history) == 1:
            lines.append(f"{label}: {history[0][1]}")
        else:
            chain = " → ".join(f"({ts} {v})" for ts, v in history)
            lines.append(f"{label}: {chain}")

    return "\n".join(lines) if lines else ""


# ================================================================
# Input sanitization
# ================================================================

def _sanitize_kv(
    kv: Dict[str, Any],
    allowed_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Keeps only keys present in allowed_fields; drops null / empty strings / "null" / "none".
    The age field is coerced to int.
    """
    if allowed_fields is None:
        allowed_fields = BASIC_FIELDS
    allowed_set = set(allowed_fields)

    result: Dict[str, Any] = {}
    for k, v in kv.items():
        if k not in allowed_set:
            continue
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() in ("null", "none"):
                continue
            result[k] = s
        elif isinstance(v, bool):
            # bool is a subclass of int, so it is special-cased first
            result[k] = "yes" if v else "no"
        elif isinstance(v, int):
            result[k] = int(v)
        elif isinstance(v, float):
            try:
                if k == "age":
                    result[k] = int(v)
                else:
                    result[k] = float(v)
            except (TypeError, ValueError):
                continue
        else:
            # Other unexpected types (list/dict) are ignored
            continue
    return result


# ================================================================
# Public entry point: upsert_basic_profile
# ================================================================

@dataclass
class BasicProfileUpsertResult:
    """Upsert result. On failure node_id="" and error is not None."""
    success: bool
    node_id: str = ""
    content: str = ""
    diff_kv: Dict[str, Any] = None  # type: ignore[assignment]
    changed: bool = False
    created: bool = False
    supersedes: Optional[str] = None
    reason: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "node_id": self.node_id,
            "content": self.content,
            "diff_kv": self.diff_kv or {},
            "changed": self.changed,
            "created": self.created,
            "supersedes": self.supersedes,
            "reason": self.reason,
            "error": self.error,
        }


async def upsert_basic_profile(
    *,
    user_id: str,
    kv: Dict[str, Any],
    vector_store,
    embed_service,
    allowed_fields: Optional[List[str]] = None,
    client_meta: Optional[Dict[str, Any]] = None,
) -> BasicProfileUpsertResult:
    """Serially update the single global L0 Basic Info chain for a user_id."""
    normalized_user_id = (user_id or "").strip()
    if not normalized_user_id:
        return BasicProfileUpsertResult(success=False, error="user_id required")
    async with _user_lock(normalized_user_id):
        return await _upsert_basic_profile_unlocked(
            user_id=normalized_user_id,
            kv=kv,
            vector_store=vector_store,
            embed_service=embed_service,
            allowed_fields=allowed_fields,
            client_meta=client_meta,
        )


async def _upsert_basic_profile_unlocked(
    *,
    user_id: str,
    kv: Dict[str, Any],
    vector_store,
    embed_service,
    allowed_fields: Optional[List[str]] = None,
    client_meta: Optional[Dict[str, Any]] = None,
) -> BasicProfileUpsertResult:
    """
    Persist the LLM-extracted basic_info kv into the user-level L0_BASIC_INFO evolution chain.

    kv fields outside allowed_fields are dropped (so the LLM cannot invent extra fields).
    A new version node: deterministic node_id plus a fresh UUID.
    Old versions: is_latest=False, status=SUPERSEDED, superseded_by pointing at the new node.

    client_meta is the business meta the caller passes through on write. Every L0 change creates a new node and turns
    the old one SUPERSEDED, structurally the same as the writer's SUPERSEDE, so it likewise carries only this write's
    meta instead of inheriting the old chain, avoiding unbounded accumulation on long chains. Historical meta stays queryable on the old nodes.
    """
    user_id = (user_id or "").strip()
    # L0 Basic Info belongs to user_id alone, from input through uniqueness to persistence.

    new_kv = _sanitize_kv(kv or {}, allowed_fields=allowed_fields)
    if not new_kv:
        return BasicProfileUpsertResult(
            success=True, changed=False, reason="no valid fields after sanitize"
        )

    # Fetch all L0 nodes for the user once, then compute both the chain head and the full KV. The old implementation
    # ran list_by_user(limit=100) followed by list_by_user(limit=1000), so whenever remote Qdrant was congested a single
    # basic_info update waited on one extra network round trip for nothing.
    all_l0_nodes = await _list_l0_nodes(vector_store, user_id)
    latest_nodes = _latest_l0_heads_from_nodes(all_l0_nodes)
    existing_full_kv = _assemble_full_kv_from_nodes(
        all_l0_nodes, allowed_fields=allowed_fields,
    )

    diff_kv: Dict[str, Any] = {}
    for k, v in new_kv.items():
        old_v = existing_full_kv.get(k)
        if old_v is None or str(v) != str(old_v):
            diff_kv[k] = v

    # Legacy agent-level L0 or several chain heads create one canonical user-level node and supersede every old head
    # even with no field change, completing the lazy migration and deduplication.
    needs_migration = (
        (not latest_nodes and bool(existing_full_kv))
        or len(latest_nodes) > 1
        or any(
            (getattr(n, "agent_id", "") or "") != USER_LEVEL_AGENT_ID
            for n in latest_nodes
        )
    )
    if not diff_kv and not needs_migration:
        return BasicProfileUpsertResult(
            success=True, changed=False, reason="no diff from existing"
        )

    stored_kv = diff_kv if diff_kv else existing_full_kv
    content = _render_content(stored_kv, field_order=allowed_fields)
    if not content:
        return BasicProfileUpsertResult(success=False, error="rendered content is empty")

    # Recompute the embedding
    try:
        embedding = await embed_service.embed_queued(content)
    except Exception as e:
        return BasicProfileUpsertResult(success=False, error=f"embed failed: {e}")

    from ...models.memory import MemoryNode, MemoryLayer, MemoryStatus, SourceType

    new_node_id = str(uuid.uuid4())
    supersedes_ids = [n.node_id for n in latest_nodes if getattr(n, "node_id", "")]
    supersedes_id = supersedes_ids[0] if supersedes_ids else None

    node_custom: Dict[str, Any] = {"basic_info_kv": stored_kv}
    if needs_migration:
        node_custom["migrated_from_agent_level"] = True
    if client_meta:
        node_custom["meta"] = [dict(client_meta)]

    new_node = MemoryNode(
        node_id=new_node_id,
        user_id=user_id,
        agent_id=USER_LEVEL_AGENT_ID,
        session_id=USER_LEVEL_SESSION_ID,
        layer=MemoryLayer.L0_BASIC_INFO,
        content=content,
        # L0 is written out of band (not through MemoryWriter), so the event time is filled in explicitly to keep
        # created_after on identical field semantics across all backends.
        memory_at=datetime.now(),
        source_type=SourceType.EXPLICIT,
        status=MemoryStatus.ACTIVE,
        is_latest=True,
        supersedes=supersedes_ids or None,
        embedding=embedding,
        custom=node_custom,
        tags=["basic_info"],
    )

    try:
        await vector_store.upsert(new_node)
    except Exception as e:
        return BasicProfileUpsertResult(success=False, error=f"upsert failed: {e}")

    # Mark every historical active/latest chain head superseded, converging onto one user-level head.
    supersede_failures: List[str] = []
    for old_head in latest_nodes:
        try:
            await vector_store.update_payload(
                old_head.node_id,
                {
                    "is_latest": False,
                    "superseded_by": [new_node_id],
                    "status": MemoryStatus.SUPERSEDED.value,
                },
            )
        except Exception as e:
            supersede_failures.append(old_head.node_id)
            logger.warning(
                f"[basic_profile] failed to supersede old L0 node "
                f"{old_head.node_id}: {e}"
            )

    logger.info(
        f"[basic_profile] user={user_id} scope=user "
        f"new L0 version={new_node_id} "
        f"diff={diff_kv} supersedes={supersedes_ids} failures={supersede_failures}"
    )

    return BasicProfileUpsertResult(
        success=True,
        node_id=new_node_id,
        content=content,
        diff_kv=diff_kv,
        changed=True,
        created=True,
        supersedes=supersedes_id,
    )


# ================================================================
# Internal: locating L0 nodes / assembling the full kv
# ================================================================

def _is_valid_l0_head(node) -> bool:
    """A chain head must be a valid L0 node: layer==L0_BASIC_INFO carrying a basic_info_kv structure.

    Defence in depth: even if an upstream query (list_by_user / get_by_id) returns a non-L0 node because the backend's
    layer filter failed (for example l1_raw), it is never treated as a chain head to supersede, which would produce a
    dirty cross-layer chain (raw stitched into the L0 evolution chain).
    """
    from ...models.memory import MemoryLayer
    if node is None:
        return False
    if getattr(node, "layer", None) != MemoryLayer.L0_BASIC_INFO:
        return False
    custom = getattr(node, "custom", None)
    return isinstance(custom, dict) and "basic_info_kv" in custom


def _node_timestamp(node) -> float:
    value = getattr(node, "gmt_created", None) or getattr(node, "valid_from", None)
    if isinstance(value, datetime):
        try:
            return value.timestamp()
        except (OSError, ValueError):
            return 0.0
    return 0.0


async def _find_latest_l0_heads(vector_store, user_id: str) -> List[Any]:
    """Return all valid active L0 chain heads for a user_id, canonical user-level nodes first."""
    nodes = await _list_l0_nodes(vector_store, user_id)
    return _latest_l0_heads_from_nodes(nodes)


async def _list_l0_nodes(vector_store, user_id: str) -> List[Any]:
    """Fetch all of a user's L0 nodes once, reused by both head selection and KV assembly."""
    from ...models.memory import MemoryLayer
    return await vector_store.list_by_user(
        user_id=user_id,
        layers=[MemoryLayer.L0_BASIC_INFO],
        limit=1000,
        with_vectors=False,
    )


def _latest_l0_heads_from_nodes(nodes: List[Any]) -> List[Any]:
    from ...models.memory import MemoryStatus
    active = [
        n for n in nodes
        if getattr(n, "status", None) == MemoryStatus.ACTIVE
    ]
    valid = [n for n in active if _is_valid_l0_head(n)]
    heads = [n for n in valid if getattr(n, "is_latest", False)] or valid
    heads.sort(
        key=lambda n: (
            (getattr(n, "agent_id", "") or "") == USER_LEVEL_AGENT_ID,
            _node_timestamp(n),
        ),
        reverse=True,
    )
    return heads


async def _find_latest_l0(vector_store, user_id: str):
    """Return this user's single or highest-priority L0 chain head."""
    heads = await _find_latest_l0_heads(vector_store, user_id)
    if heads:
        return heads[0]

    # Very old backends may fail to find the canonical deterministic node through list_by_user.
    legacy = await vector_store.get_by_id(basic_info_node_id(user_id))
    if _is_valid_l0_head(legacy):
        return legacy
    return None



async def _assemble_full_kv(
    vector_store,
    user_id: str,
    allowed_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the full KV for a user_id across historical agent chains."""
    all_nodes = await _list_l0_nodes(vector_store, user_id)
    return _assemble_full_kv_from_nodes(all_nodes, allowed_fields=allowed_fields)


def _assemble_full_kv_from_nodes(
    all_nodes: List[Any],
    *,
    allowed_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the full KV from the L0 nodes already read, without triggering a second remote query."""
    all_nodes.sort(key=_node_timestamp)

    allowed_set = set(allowed_fields) if allowed_fields else None
    full_kv: Dict[str, Any] = {}
    for n in all_nodes:
        if not isinstance(n.custom, dict):
            continue
        kv = n.custom.get("basic_info_kv") or {}
        if not isinstance(kv, dict):
            continue
        for k, v in kv.items():
            if allowed_set is not None and k not in allowed_set:
                continue
            full_kv[k] = v
    return full_kv
