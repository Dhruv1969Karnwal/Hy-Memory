# -*- coding: utf-8 -*-
"""
HY Memory - Intention (L7) recall + lazy expiry conversion + duplicate detection

Intentions (L7_INTENTION) are written straight into the VDB by the extractor (System 1) with an optional `valid_until`
deadline. The reader calls this module on the proactive channel to recall them:

  - Recall L7 nodes with status=ACTIVE (**oversampled**, see below);
  - Check `valid_until` on every hit: once expired (now > valid_until) the node's layer is **lazily** rewritten
    to l2_fact (persisted through update_payload) and it is dropped from the intention results --
    an expired intention becomes an ordinary historical fact and goes through normal recall next time;
  - **Dedup (the key part)**: surviving intentions are grouped by the rule below, and only one per group is returned:
        same user_id + same memory_at day + identical valid_until (both None or equal)
        + cosine similarity >= MEMORY_DEDUP_THRESHOLD (default 0.95)
    -> treated as duplicates of one intention (an echoing extractor / the same intention restated across sessions).
    The member with the earliest `gmt_created` survives (first come first served, stable id);
  - **Async delete**: the remaining duplicates are removed from the VDB fire-and-forget, without affecting this call's RT.

Q: Why look at memory_at and valid_until at all?
A: An intention is a point-in-time signal. The same sentence "I have a meeting on Friday" can refer to
   two distinct events: one with memory_at on this Monday (valid_until=this Friday) and another with
   memory_at on next Monday (valid_until=next Friday), those are **two different meetings** that must never be merged; cosine alone would cause false merges.

Best-effort: any exception in expiry conversion / dedup / async deletion leaves the main search flow untouched.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from ...models.memory import MemoryLayer, MemoryStatus

logger = logging.getLogger(__name__)


def _get_dedup_threshold() -> float:
    """Dedup cosine threshold: reuses MEMORY_DEDUP_THRESHOLD from the dedup core, default 0.95."""
    try:
        return float(os.environ.get("MEMORY_DEDUP_THRESHOLD", 0.95))
    except (TypeError, ValueError):
        return 0.95


def _memory_at_day(node: Any) -> Optional[date]:
    """Date part of node.memory_at; returns None for None."""
    if node is None:
        return None
    ma = getattr(node, "memory_at", None)
    if ma is None:
        return None
    # Both datetime and date are accepted
    try:
        return ma.date() if hasattr(ma, "date") else ma
    except Exception:
        return None


def _valid_until_key(node: Any) -> Any:
    """Normalize valid_until into a "comparison key": datetime→date, None→None.

    Only day-level agreement is required (the writer pins valid_until to the end of that day, so the same day counts as equal).
    """
    if node is None:
        return None
    vu = getattr(node, "valid_until", None)
    if vu is None:
        return None
    try:
        return vu.date() if hasattr(vu, "date") else vu
    except Exception:
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    """Plain cosine (agreed: "take the ready-made embeddings from the VDB, do not recompute"; this just compares the vectors we received once).

    Call volume: fetch_limit <= 30 per search, so the O(n²) pairwise comparison stays under ~500 float
    dot products and numpy is not needed. Returns [-1, 1], with -inf for empty vectors to avoid false matches.
    """
    if not a or not b:
        return float("-inf")
    la = sum(x * x for x in a) ** 0.5
    lb = sum(x * x for x in b) ** 0.5
    if la == 0.0 or lb == 0.0:
        return float("-inf")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (la * lb)


def _partition_duplicates(
    hits: List[Dict[str, Any]],
    threshold: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split hits into a "kept list" and a "node_ids to delete" list.

    Grouping key: (memory_at.day, valid_until.day); inside a group cosine is compared pairwise,
    and pairs >= threshold are duplicates. A duplicate group keeps the earliest gmt_created (a missing
    gmt counts as +inf, i.e. latest); the rest go into delete_ids.

    Args:
        hits: L7 hits after recall and the not-expired filter
        threshold: cosine duplicate threshold

    Returns:
        (kept_hits, delete_ids) -- kept_hits preserves the original order of the retained items
    """
    if len(hits) < 2:
        return list(hits), []

    # 1. Bucket by (memory_at day, valid_until day)
    buckets: Dict[Tuple[Any, Any], List[int]] = {}
    for idx, h in enumerate(hits):
        node = h.get("node")
        key = (_memory_at_day(node), _valid_until_key(node))
        buckets.setdefault(key, []).append(idx)

    def _gmt(h: Dict[str, Any]) -> float:
        node = h.get("node")
        gc = getattr(node, "gmt_created", None) if node else None
        if gc is None:
            return float("inf")
        try:
            return gc.timestamp() if hasattr(gc, "timestamp") else float(gc)
        except Exception:
            return float("inf")

    dropped_idx: set = set()

    # 2. Within each bucket compare cosine pairwise → union-find
    for key, idx_list in buckets.items():
        if len(idx_list) < 2:
            continue

        parent = {i: i for i in idx_list}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Collect embeddings
        embs: Dict[int, List[float]] = {}
        for i in idx_list:
            node = hits[i].get("node")
            emb = getattr(node, "embedding", None) if node else None
            if emb:
                embs[i] = emb

        # Pairwise comparison
        candidates = [i for i in idx_list if i in embs]
        for a_pos in range(len(candidates)):
            for b_pos in range(a_pos + 1, len(candidates)):
                ia, ib = candidates[a_pos], candidates[b_pos]
                if _cosine(embs[ia], embs[ib]) >= threshold:
                    union(ia, ib)

        # 3. Keep the earliest gmt in each connected component
        groups: Dict[int, List[int]] = {}
        for i in idx_list:
            groups.setdefault(find(i), []).append(i)

        for members in groups.values():
            if len(members) < 2:
                continue  # single node, not a duplicate
            # Keep the earliest gmt
            keeper = min(members, key=lambda i: _gmt(hits[i]))
            for i in members:
                if i != keeper:
                    dropped_idx.add(i)

    kept = [h for i, h in enumerate(hits) if i not in dropped_idx]
    delete_ids = [
        hits[i].get("node_id") for i in sorted(dropped_idx)
        if hits[i].get("node_id")
    ]
    return kept, delete_ids


def _spawn_async_delete(
    vector_store: Any,
    delete_ids: List[str],
    *,
    user_id_for_log: str = "",
) -> None:
    """Delete duplicates from the VDB fire-and-forget, without affecting main search RT.

    - Attached to the current loop with asyncio.create_task (recall already runs in an async context);
    - Per-item failures are swallowed, with one summary log line at the end;
    - If no loop can be obtained (very rare), it silently skips.
    """
    if not delete_ids or vector_store is None:
        return

    async def _run():
        ok, fail = 0, 0
        for nid in delete_ids:
            try:
                await vector_store.delete(nid)
                ok += 1
            except Exception as e:
                fail += 1
                logger.debug(f"[intention] async delete {nid} failed: {e}")
        logger.info(
            f"[intention] async dedup delete done: ok={ok} fail={fail} "
            f"total={len(delete_ids)} user_id={user_id_for_log or '-'}"
        )

    try:
        asyncio.create_task(_run())
    except RuntimeError:
        # No running loop: skip silently (recall itself runs in async, so this is normally unreachable)
        logger.debug("[intention] no running loop for async delete; skipped")


async def recall_intentions(
    vector_store: Any,
    query_embedding: List[float],
    *,
    user_ids: Optional[List[str]] = None,
    agent_ids: Optional[List[str]] = None,
    limit: int = 10,
    now: Optional[datetime] = None,
    stats_out: Optional[Dict[str, Any]] = None,
    include_sources: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Recall non-expired L7_INTENTION nodes (VDB), lazily convert expired ones to L2_FACT, dedup and delete duplicates asynchronously.

    Args:
        vector_store: VectorStoreBase instance
        query_embedding: query vector
        user_ids: user isolation (single or multiple users)
        agent_ids: optional agent filter
        limit: recall cap (how many rows are returned upstream)
        now: current time (defaults to datetime.now(), injectable for tests)
        stats_out: optional dict this function **fills in place** with per-stage timings and counts, so the caller
            can persist the READ_INTENTION pipeline log. Passing None disables measurement entirely (zero overhead).

    Returns:
        [{"node_id", "score", "node", "layer": "l7_intention", "source": "vdb_intention"}, ...]

    Implementation details:
      - **Oversampling**: fetch more from the VDB to absorb echoing L7 writes; trimmed to limit after dedup
      - **Dedup rule**: same memory_at day + same valid_until day + cosine>=0.95.
      - **Async delete**: duplicates to remove are deleted from the VDB fire-and-forget, without affecting response RT.

    Instrumentation fields (filled into stats_out):
      Timings (ms): vdb_search_ms / expire_ms / backfill_embed_ms / dedup_ms / total_ms
      Counts: fetched / expired / survivors / dedup_dropped / returned
      Config: fetch_limit / out_limit / dedup_threshold
      Notes: expire_ms and backfill_embed_ms are non-zero only when the corresponding work really happened,
            which shows whether the slowness is in VDB search, per-node expiry conversion, embedding backfill or dedup compute.
    """
    if limit <= 0:
        return []
    _now = now or datetime.now()
    _t_total = time.perf_counter()

    def _stat(**kv: Any) -> None:
        """Fill instrumentation fields; a no-op when stats_out is None."""
        if stats_out is not None:
            stats_out.update(kv)

    # Oversampling: fetch more from the VDB to absorb echoing L7 writes; trimmed to limit after dedup
    fetch_limit = max(limit * 6, limit + 15)
    _stat(fetch_limit=fetch_limit, out_limit=limit)

    _t = time.perf_counter()
    try:
        hits = await vector_store.search(
            query_embedding=query_embedding,
            user_ids=user_ids or None,
            agent_ids=agent_ids,
            layers=[MemoryLayer.L7_INTENTION],
            limit=fetch_limit,
            score_threshold=0.0,  # intentions set no semantic threshold
            status_filter=[MemoryStatus.ACTIVE],
            only_latest=True,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
        ) or []
    except Exception as e:
        logger.debug(f"[intention] recall search failed: {e}")
        _stat(
            vdb_search_ms=round((time.perf_counter() - _t) * 1000, 2),
            total_ms=round((time.perf_counter() - _t_total) * 1000, 2),
            fetched=0, returned=0, search_failed=True,
        )
        return []
    _stat(vdb_search_ms=round((time.perf_counter() - _t) * 1000, 2), fetched=len(hits))

    # 1. Lazy expiry conversion: past valid_until → convert to L2_FACT + drop from results
    _t = time.perf_counter()
    survivors: List[Dict[str, Any]] = []
    expired_count = 0
    for h in hits:
        node = h.get("node")
        node_id = h.get("node_id", "")
        valid_until = getattr(node, "valid_until", None) if node else None

        if valid_until and _now > valid_until:
            try:
                await vector_store.update_payload(node_id, {"layer": MemoryLayer.L2_FACT.value})
                logger.debug(
                    f"[intention] expired L7 → L2_FACT: node_id={node_id} "
                    f"valid_until={valid_until}"
                )
                expired_count += 1
            except Exception as e:
                logger.debug(f"[intention] expire conversion failed for {node_id}: {e}")
            continue

        survivors.append({
            **h,
            "layer": MemoryLayer.L7_INTENTION.value,
            "retrieval_source": "vdb_intention",
        })
    # With no expired node this is a pure in-memory loop costing ~0; any non-zero value is per-node update_payload overhead
    _stat(
        expire_ms=round((time.perf_counter() - _t) * 1000, 2),
        expired=expired_count,
        survivors=len(survivors),
    )

    # 1.5. Key patch: backends disagree on whether search() returns embeddings
    #    - Qdrant/Faiss return embeddings on hits in some situations
    #    - Chroma search() does not include embeddings by default → node.embedding=None
    # If no embedding is available at dedup time, _partition_duplicates silently degrades to "no dedup".
    # Nodes missing an embedding are backfilled in bulk with get_embeddings (the vectors already live in the VDB,
    # no embedding service recomputation). On failure they stay None → dedup degrades naturally, main flow unaffected.
    _t = time.perf_counter()
    _backfilled = 0
    _missing_cnt = 0
    if survivors:
        missing_ids = [
            h.get("node_id") for h in survivors
            if h.get("node_id") and not getattr(h.get("node"), "embedding", None)
        ]
        _missing_cnt = len(missing_ids)
        if missing_ids and hasattr(vector_store, "get_embeddings"):
            try:
                emb_map = await vector_store.get_embeddings(missing_ids)
                if emb_map:
                    for h in survivors:
                        node = h.get("node")
                        nid = h.get("node_id")
                        if node is not None and nid in emb_map and not getattr(node, "embedding", None):
                            try:
                                node.embedding = list(emb_map[nid])
                            except Exception:
                                pass
                    _backfilled = len(emb_map)
                    logger.debug(
                        f"[intention] backfilled embeddings for {len(emb_map)}/"
                        f"{len(missing_ids)} nodes before dedup"
                    )
            except Exception as e:
                logger.debug(f"[intention] get_embeddings failed (skip backfill): {e}")
    # Non-zero only when a backfill really happened; a large value means get_embeddings is the bottleneck
    _stat(
        backfill_embed_ms=round((time.perf_counter() - _t) * 1000, 2),
        embed_missing=_missing_cnt,
        embed_backfilled=_backfilled,
    )

    # 2. Dedup: same user + same memory_at day + same valid_until day + cosine>=threshold
    _t = time.perf_counter()
    threshold = _get_dedup_threshold()
    try:
        kept, delete_ids = _partition_duplicates(survivors, threshold)
    except Exception as e:
        # Any dedup exception → degrade to no dedup (all survivors, truncated to limit)
        logger.warning(f"[intention] dedup partition failed, degrade to raw: {e}")
        kept, delete_ids = survivors, []
    # Pure CPU (pairwise cosine); can become significant with many survivors
    _stat(
        dedup_ms=round((time.perf_counter() - _t) * 1000, 2),
        dedup_dropped=len(delete_ids),
        dedup_threshold=threshold,
    )

    # 3. Truncate to the limit the caller asked for
    result = kept[:limit]
    _stat(
        returned=len(result),
        total_ms=round((time.perf_counter() - _t_total) * 1000, 2),
    )

    # 4. Async delete (fire-and-forget)
    if delete_ids:
        user_id_for_log = (user_ids[0] if user_ids else "") or ""
        _spawn_async_delete(vector_store, delete_ids, user_id_for_log=user_id_for_log)

    if expired_count or delete_ids or len(hits) != len(result):
        logger.info(
            f"[intention] recall: fetched={len(hits)} expired={expired_count} "
            f"survivors={len(survivors)} dedup_dropped={len(delete_ids)} "
            f"kept_after_dedup={len(kept)} returned={len(result)} "
            f"(fetch_limit={fetch_limit}, out_limit={limit}, thr={threshold})"
        )
    return result
