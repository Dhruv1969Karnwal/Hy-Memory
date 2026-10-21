# -*- coding: utf-8 -*-
"""
Core memory deduplication logic (pure functions, no I/O, unit-testable).

Production L2_FACT / L4_IDENTITY often accumulate duplicates (the extractor extracts twice, reconcile misses a merge).
This module identifies duplicate groups by embedding cosine similarity (above a threshold, default 0.95) and
decides keep/delete by deterministic rules, shared by the extractor / reconcile / search pipelines.

Scope of duplicate detection (the caller supplies it):
  Only plain memories and chain heads (is_latest=True) are compared; chain bodies (SUPERSEDED,
  is_latest=False) never take part -- they are history, represented by their chain head.

The isolation dimension follows **owner** (see group_items_by_owner):
  owner='user' (incl. empty) -> batch by (user_id,), **merged across agents**: the same thing one
    user said under different agents is one and the same fact.
  owner='agent'              -> batch by (user_id, agent_id), **isolated per agent**: what
    different assistants said are independent memories.

Deletion decisions (per duplicate group, deterministic, never random):
  - all non-chain nodes     -> keep the earliest gmt_created, delete the rest
  - 1 chain head + non-chain -> keep the chain head, delete the non-chain ones
  - n chain heads (n≥2)      -> keep the head with the earliest gmt_created, delete the other heads plus non-chain;
                            each deleted chain head **takes its whole SUPERSEDED history chain with it**
                            (chain_node_ids is filled in ahead of time by the caller via _trace_full_chain)

"Delete" here is a **logical delete**: status=SHADOW and is_latest=False (never recalled by retrieval),
not a hard delete, and it is recoverable. See _shadow_nodes.

Similarity: the vectors of participating nodes are stacked into one matrix scored once as M @ M.T (rows L2-normalized first),
about 0.3ms for 30 items at 4096 dims; no embedding API call and no pre-filtering required.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

logger = logging.getLogger(__name__)

DEDUP_LOG_STEP = "DEDUP"

DEFAULT_DEDUP_THRESHOLD = 0.95


def get_dedup_threshold() -> float:
    """Dedup cosine threshold, overridable by env MEMORY_DEDUP_THRESHOLD, default 0.95."""
    try:
        return float(os.environ.get("MEMORY_DEDUP_THRESHOLD", DEFAULT_DEDUP_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_DEDUP_THRESHOLD


@dataclass
class DedupItem:
    """One dedup input item (assembled by the caller).

    is_chain_head / chain_node_ids are pre-determined and filled in by the caller:
      - plain node: is_chain_head=False, chain_node_ids=[node_id]
      - chain head: is_chain_head=True,  chain_node_ids=[head, ...SUPERSEDED bodies]
    """
    node_id: str
    embedding: List[float]
    content: str = ""
    is_latest: bool = True
    is_chain_head: bool = False
    gmt_created: Optional[float] = None  # Unix ts; used for deterministic retention (smaller is earlier)
    chain_node_ids: List[str] = field(default_factory=list)
    # Used for batching (see group_items_by_owner): owner decides the isolation dimension, user_id/agent_id
    # decide the batch key. When all three are missing the caller's default_* values take over, degrading
    # behaviour to "everything in one batch", which equals the behaviour before batching existed.
    owner: Optional[str] = None
    user_id: str = ""
    agent_id: str = ""
    source: str = "default"


def _gmt_key(item: DedupItem) -> float:
    """Sort key for gmt_created: a missing value counts as +inf (latest), so timestamped items are kept first."""
    return item.gmt_created if item.gmt_created is not None else float("inf")


# owner normalization: an empty owner is always treated as 'user'.
#
# The owner field was only introduced on 2026-06-16 (commit a240189); L2 nodes written before it lack the field,
# and the LLM may omit it too. The extractor/reconcile prompts both stipulate "When unsure,
# default to user", and `reconciler.py` falls back the same way when assembling existing_mem_list with
# `owner or "user"` -- this module keeps the same convention so the whole pipeline stays consistent.
OWNER_USER = "user"
OWNER_AGENT = "agent"


def _owner_bucket(owner: Optional[str]) -> str:
    """Normalize owner into the 'user' / 'agent' buckets; empty or unknown values go to 'user'."""
    if not owner:
        return OWNER_USER
    v = str(owner).strip().lower()
    return OWNER_AGENT if v == OWNER_AGENT else OWNER_USER


def group_items_by_owner(
    items: List[DedupItem],
    *,
    default_user_id: str = "",
    default_agent_id: str = "",
) -> List[Dict[str, Any]]:
    """Split candidates into batches that never dedup against each other, following owner semantics.

    The isolation dimension of dedup should follow **owner**, not a blanket split by agent_id:

      - owner='user' (incl. empty): the memory belongs to the **user** side. The same thing one user
        says under agent_a and agent_b is one fact and must be judged in the same batch, otherwise
        "The user is a vegetarian." would be stored once per agent and could never merge.
        → batch key = (user_id,), **merged across agents**
      - owner='agent': the memory is something an assistant provided or was asked to do. What agent_a
        said and what agent_b said are two independent memories that must not delete each other, however similar.
        → batch key = (user_id, agent_id), **isolated per agent**

    L4_IDENTITY always has an empty owner (only L2_FACT / L7_INTENTION write owner), so by the fallback above
    it lands in the user bucket -- identity is the user profile anyway, and merging across agents is the correct semantics.

    This also fixes wrong audit attribution: callers previously passed a single user_id/agent_id (taking
    `effective_user_ids[0]`), so the DEDUP log recorded the wrong owner for cross-user or cross-agent retrieval.
    After batching, each batch has a definite user_id/agent_id and the log is accurate by construction.

    Args:
        items: candidates, carrying user_id / agent_id / owner (default_* covers anything missing).
        default_user_id / default_agent_id: fallback values when the item carries no such field.

    Returns:
        [{"user_id": ..., "agent_id": ..., "owner": "user"|"agent",
          "items": [DedupItem, ...]}, ...]
        Only batches with len(items) >= 2 are returned (a single item cannot be compared against anything).
    """
    buckets: Dict[tuple, Dict[str, Any]] = {}
    for it in items:
        uid = it.user_id or default_user_id
        aid = it.agent_id or default_agent_id
        from ...utils.source_filter import normalize_source
        source = normalize_source(it.source)
        bucket = _owner_bucket(it.owner)
        # The user bucket carries no agent_id -> all agents of one user collapse into a single batch
        key = (
            (uid, "", OWNER_USER, source)
            if bucket == OWNER_USER else (uid, aid, OWNER_AGENT, source)
        )
        entry = buckets.get(key)
        if entry is None:
            entry = {
                "user_id": uid,
                # The user bucket spans agents and has no single owning agent_id: use the caller's default for
                # the audit log (meaning "the agent that triggered this dedup"); it plays no part in batching.
                "agent_id": default_agent_id if bucket == OWNER_USER else aid,
                "owner": bucket,
                "source": source,
                "items": [],
            }
            buckets[key] = entry
        entry["items"].append(it)

    return [b for b in buckets.values() if len(b["items"]) >= 2]


def _pairwise_cosine(vectors: List[List[float]]):
    """Rows L2-normalized then M @ M.T, returning the similarity matrix (numpy)."""
    M = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = M / norms
    return M @ M.T


def _union_find_groups(n: int, sim, threshold: float) -> List[List[int]]:
    """Union-find over index pairs whose sim > threshold; returns the index list of each group (singletons included)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i][j]) > threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _decide_group(members: List[DedupItem]) -> Dict[str, Any]:
    """Decide keep/delete for one duplicate group and return the group detail (with audit fields)."""
    heads = [m for m in members if m.is_chain_head]

    if heads:
        # Chain head present: keep the head with the earliest gmt; delete the rest (other heads plus non-chain)
        kept = min(heads, key=_gmt_key)
        reason = "earliest_chain_head" if len(heads) > 1 else "chain_head"
    else:
        # All non-chain: keep the one with the earliest gmt
        kept = min(members, key=_gmt_key)
        reason = "earliest_gmt"

    deleted = []
    delete_ids: List[str] = []
    for m in members:
        if m.node_id == kept.node_id:
            continue
        # A deleted chain head takes its whole history chain with it (chain_node_ids holds the head plus bodies)
        cascade = [cid for cid in (m.chain_node_ids or [m.node_id]) if cid != kept.node_id]
        if not cascade:
            cascade = [m.node_id]
        also = [cid for cid in cascade if cid != m.node_id]
        deleted.append({
            "node_id": m.node_id,
            "content": m.content,
            "is_chain_head": m.is_chain_head,
            "also_deleted_chain": also,
        })
        delete_ids.extend(cascade)

    return {
        "members": [
            {
                "node_id": m.node_id,
                "content": m.content,
                "is_latest": m.is_latest,
                "is_chain_head": m.is_chain_head,
            }
            for m in members
        ],
        "kept": {"node_id": kept.node_id, "content": kept.content, "reason": reason},
        "deleted": deleted,
        "delete_ids": delete_ids,
    }


def plan_dedup(
    items: List[DedupItem],
    *,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Plan deduplication for a set of candidates (pure function, performs no deletion).

    Only is_latest=True items are compared (plain nodes plus chain heads); is_latest=False chain bodies are ignored,
    but when a chain head is deleted, the bodies in its chain_node_ids join delete_ids and go with it.

    Returns:
        {
          "threshold": float,
          "groups": [ {members, kept, deleted, delete_ids}, ... ]  # only groups that delete something
          "delete_ids": [every node_id to delete (including cascaded chains), deduplicated],
          "kept_ids":   [every retained representative node_id],
        }
    """
    thr = threshold if threshold is not None else get_dedup_threshold()
    empty = {"threshold": thr, "groups": [], "delete_ids": [], "kept_ids": [],
             "compared_count": 0, "max_cosine": None, "avg_cosine": None}

    if np is None or not items:
        return empty

    # Only is_latest=True items that carry an embedding take part
    parts = [it for it in items if it.is_latest and it.embedding]
    if len(parts) < 2:
        return {**empty, "compared_count": len(parts)}

    sim = _pairwise_cosine([it.embedding for it in parts])
    idx_groups = _union_find_groups(len(parts), sim, thr)

    # Upper-triangle cosine statistics (diagonal 1.0 excluded) for audit: max / mean
    n = len(parts)
    iu = np.triu_indices(n, k=1)
    pair_cos = sim[iu]
    max_cos = float(pair_cos.max()) if pair_cos.size else None
    avg_cos = float(pair_cos.mean()) if pair_cos.size else None

    groups_out: List[Dict[str, Any]] = []
    all_delete: List[str] = []
    all_kept: List[str] = []
    seen_del = set()

    for idxs in idx_groups:
        if len(idxs) < 2:
            continue  # not a duplicate, skip
        members = [parts[i] for i in idxs]
        decided = _decide_group(members)
        all_kept.append(decided["kept"]["node_id"])
        for did in decided["delete_ids"]:
            if did not in seen_del:
                seen_del.add(did)
                all_delete.append(did)
        groups_out.append({
            "members": decided["members"],
            "kept": decided["kept"],
            "deleted": decided["deleted"],
        })

    return {
        "threshold": thr,
        "groups": groups_out,
        "delete_ids": all_delete,
        "kept_ids": all_kept,
        "compared_count": len(parts),
        "max_cosine": max_cos,
        "avg_cosine": avg_cos,
    }


def _build_dedup_content(trigger: str, plan: Dict[str, Any]) -> str:
    """Build the human-readable content of a DEDUP step.

    Nothing deleted: trigger + comparison count + max/avg cosine + threshold.
    Something deleted: all of the above plus, per duplicate group, which ones duplicated / what was deleted / which one was kept.
    """
    thr = plan.get("threshold")
    compared = plan.get("compared_count", 0)
    max_c = plan.get("max_cosine")
    avg_c = plan.get("avg_cosine")
    delete_ids = plan.get("delete_ids", [])

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

    lines = [
        f"[DEDUP] trigger={trigger} threshold={thr}",
        f"compared_count={compared} max_cosine={_fmt(max_c)} avg_cosine={_fmt(avg_c)}",
        f"deleted_count={len(delete_ids)}",
    ]

    groups = plan.get("groups", [])
    if groups:
        for gi, g in enumerate(groups):
            members = g.get("members", [])
            kept = g.get("kept", {})
            deleted = g.get("deleted", [])
            lines.append(f"-- group#{gi} ({len(members)} duplicates) --")
            for m in members:
                lines.append(
                    f"   dup: id={m.get('node_id')} head={m.get('is_chain_head')} "
                    f"content={(m.get('content') or '')[:120]}"
                )
            lines.append(
                f"   KEEP: id={kept.get('node_id')} reason={kept.get('reason')} "
                f"content={(kept.get('content') or '')[:120]}"
            )
            for d in deleted:
                also = d.get("also_deleted_chain") or []
                also_s = f" also_chain={also}" if also else ""
                lines.append(
                    f"   DELETE: id={d.get('node_id')} head={d.get('is_chain_head')}{also_s} "
                    f"content={(d.get('content') or '')[:120]}"
                )
        lines.append(f"all_deleted_ids={delete_ids}")

    return "\n".join(lines)


async def _emit_dedup_log(
    cache: Any,
    *,
    request_id: str,
    user_id: str,
    agent_id: str,
    trigger: str,
    plan: Dict[str, Any],
    extra_parsed: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the DEDUP pipeline log (deletion audit detail). Best-effort."""
    if cache is None or not getattr(cache, "store_pipeline_log", None):
        return
    try:
        parsed = {
            "trigger": trigger,
            "threshold": plan.get("threshold"),
            "groups": plan.get("groups", []),
            "deleted_count": len(plan.get("delete_ids", [])),
            "kept_count": len(plan.get("kept_ids", [])),
            "compared_count": plan.get("compared_count", 0),
            "max_cosine": plan.get("max_cosine"),
            "avg_cosine": plan.get("avg_cosine"),
        }
        if extra_parsed:
            parsed.update(extra_parsed)
        await cache.store_pipeline_log(
            request_id=request_id or "",
            user_id=user_id or "",
            agent_id=agent_id or "default_agent",
            step=DEDUP_LOG_STEP,
            prompt="",
            response=_build_dedup_content(trigger, plan),
            parsed=json.dumps(parsed, ensure_ascii=False, default=str),
            memory_ids=plan.get("delete_ids", []) or None,
        )
    except Exception as e:
        logger.debug(f"[dedup] emit DEDUP log failed: {e}")


async def _shadow_nodes(
    vector_store: Any,
    node_ids: List[str],
    *,
    kept_id: str,
    trigger: str,
) -> int:
    """Mark duplicate nodes SHADOW (logical delete) instead of hard-deleting them.

    This used to call `vector_store.delete(nid)`, an unrecoverable hard delete -- yet what gets deleted can be a whole
    evolution chain (a chain head judged duplicate takes its chain_node_ids with it), so a single misjudged threshold
    would permanently erase a memory's whole evolution trail. Dedup on the search path is also a fire-and-forget background task,
    entirely invisible to the user, so a mistake could not be recovered.

    SHADOW is semantically equivalent (no retrieval recalls it, so it is a logical delete) but recoverable,
    and uses the same status bits as reconciler UPDATE and L1_RAW degradation.

    update_payload is used instead of upsert: only status bits change and vectors are never rewritten (a DedupItem holds
    only an embedding, not a full node, and upserting a whole node would damage the payload).
    """
    from ...models.memory import MemoryStatus

    ok = 0
    for nid in node_ids:
        try:
            await vector_store.update_payload(
                nid,
                {
                    "is_latest": False,
                    "status": MemoryStatus.SHADOW.value,
                    # Audit: why this one disappeared and which one replaced it
                    "shadow_reason": f"dedup:{trigger}",
                    "dedup_kept_id": kept_id,
                },
            )
            ok += 1
        except Exception as e:
            logger.warning(f"[dedup] shadow {nid} failed (trigger={trigger}): {e}")
    return ok


async def execute_dedup(
    items: List[DedupItem],
    *,
    vector_store: Any,
    cache: Any = None,
    trigger: str,
    request_id: str = "",
    user_id: str = "",
    agent_id: str = "",
    delete_from_store: bool = True,
    threshold: Optional[float] = None,
    extra_parsed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Dedup orchestration: plan -> (optionally) delete from store -> write the DEDUP log. Best-effort throughout; never raises into the main flow.

    Args:
        items: candidates within one isolation scope (the caller guarantees no cross user/agent mixing).
        delete_from_store: when True, plan.delete_ids become SHADOW (logical delete, recoverable, not a
            hard delete -- see _shadow_nodes). Pass False during the extractor stage (nothing stored yet) to
            only record the dropped items.
        trigger: "extractor" / "reconcile" / "search", recorded in the DEDUP log.
        extra_parsed: extra fields written into DEDUP log.parsed (for example embed latency or provider meta).

    Returns:
        The plan dict (delete_ids / kept_ids / groups). delete_ids is empty when nothing is duplicated.
    """
    try:
        plan = plan_dedup(items, threshold=threshold)
    except Exception as e:
        logger.warning(f"[dedup] plan_dedup failed (trigger={trigger}): {e}")
        return {"threshold": threshold, "groups": [], "delete_ids": [], "kept_ids": []}

    delete_ids = plan.get("delete_ids", [])
    if not delete_ids:
        # No duplicates: still persist one DEDUP step (recording compared_count / max_cosine / avg_cosine) so
        # it is provable that dedup ran and how far the closest pair stayed from the threshold.
        await _emit_dedup_log(
            cache,
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            trigger=trigger,
            plan=plan,
            extra_parsed=extra_parsed,
        )
        return plan

    if delete_from_store and vector_store is not None:
        # Logical delete (SHADOW), never a hard delete. Each group's kept_id is recorded separately so a later
        # trace can answer which memory replaced which.
        for g in plan.get("groups", []):
            _kept = (g.get("kept") or {}).get("node_id", "")
            _gids: List[str] = []
            for d in (g.get("deleted") or []):
                if d.get("node_id"):
                    _gids.append(d["node_id"])
                _gids.extend(d.get("also_deleted_chain") or [])
            if _gids:
                await _shadow_nodes(
                    vector_store, _gids, kept_id=_kept, trigger=trigger,
                )

    await _emit_dedup_log(
        cache,
        request_id=request_id,
        user_id=user_id,
        agent_id=agent_id,
        trigger=trigger,
        plan=plan,
        extra_parsed=extra_parsed,
    )

    logger.info(
        f"[dedup] trigger={trigger} groups={len(plan.get('groups', []))} "
        f"shadowed={len(delete_ids)} (apply={delete_from_store})"
    )
    return plan
