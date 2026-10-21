# -*- coding: utf-8 -*-
"""
Profile evidence reverse lookup -- find the L6 schemas that support a batch of VDB hit nodes.

The forward path (graph vector_search) looks for L6 nodes semantically close to the query; the reverse path (this module)
starts from the VDB nodes hit by normal recall and walks DERIVED_FROM edges backwards to the L6 schemas referencing them, ordered by support
(how many hit nodes back each one). Forward and reverse feed the reader's RRF fusion into profile output.

graph_store.find_referencing_memories returns one row per (Memory, VdbRef) edge, carrying
evidence_vdb_id. An L6 supported by N input nodes shows up in N rows; support = the number of distinct
evidence_vdb_id values on that L6.

Free async function with graph_store as the first argument, matching evolution.recall / intention.recall;
other readers can import and reuse it. Best-effort: unavailable graph, missing capability or exceptions all return [].
"""

from typing import Any, Dict, List
import json

import logging

from ..specific_supplement import is_internal_profile_chain_custom

logger = logging.getLogger(__name__)

_L6_LAYER = "l6_schema"


async def reverse_lookup_l6(
    graph_store: Any,
    vdb_node_ids: List[str],
    *,
    limit: int = 50,
    include_sources: List[str] = None,
    exclude_sources: List[str] = None,
) -> List[Dict[str, Any]]:
    """Reverse-lookup the L6 schemas supporting a list of VDB node_ids, sorted by support descending.

    Args:
        graph_store: graph store (returns [] when None or when find_referencing_memories is absent).
        vdb_node_ids: VDB node ids hit by the normal channel (the reverse-lookup anchors).
        limit: upper bound on the number of edges to reverse-lookup (passed to find_referencing_memories).

    Returns:
        Hit list compatible with the reader output layer, sorted by support descending:
          [{node_id, content, layer:"l6_schema", score:<support/max_support>,
            source:"schema_reverse", confidence, _support_count, node:None}, ...]
        Support = number of distinct evidence_vdb_id values. Returns [] when nothing is found.
    """
    if graph_store is None or not vdb_node_ids:
        return []
    if not hasattr(graph_store, "find_referencing_memories"):
        return []

    try:
        if include_sources or exclude_sources:
            rows = await graph_store.find_referencing_memories(
                vdb_node_ids, limit=limit,
                include_sources=include_sources,
                exclude_sources=exclude_sources,
            )
        else:
            rows = await graph_store.find_referencing_memories(
                vdb_node_ids, limit=limit,
            )
    except Exception as e:
        logger.debug(f"[profile-evidence] reverse_lookup_l6 failed: {e}")
        return []

    # Aggregate by L6 node_id: support = number of distinct evidence_vdb_id values
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if row.get("layer") != _L6_LAYER:
            continue
        raw_custom = row.get("custom_json") or {}
        if isinstance(raw_custom, str):
            try:
                custom = json.loads(raw_custom or "{}")
            except json.JSONDecodeError:
                custom = {}
        elif isinstance(raw_custom, dict):
            custom = raw_custom
        else:
            custom = {}
        if is_internal_profile_chain_custom(custom):
            continue
        nid = row.get("node_id")
        if not nid:
            continue
        g = grouped.get(nid)
        if g is None:
            g = {
                "node_id": nid,
                "content": row.get("content", ""),
                "confidence": row.get("confidence"),
                "memory_source": row.get("memory_source") or "default",
                "_schema_kind": str(custom.get("schema_kind") or "behavior_pattern"),
                "_evidence_ids": set(),
            }
            grouped[nid] = g
        ev = row.get("evidence_vdb_id")
        if ev:
            g["_evidence_ids"].add(ev)

    if not grouped:
        return []

    max_support = max(len(g["_evidence_ids"]) or 1 for g in grouped.values())

    hits: List[Dict[str, Any]] = []
    for g in grouped.values():
        support = len(g["_evidence_ids"]) or 1
        hits.append({
            "node_id": g["node_id"],
            "content": g["content"],
            "layer": _L6_LAYER,
            # RRF only looks at rank; the score is for observation only, normalized to [0,1]
            "score": support / max_support,
            "retrieval_source": "schema_reverse",
            "memory_source": g.get("memory_source") or "default",
            "confidence": g.get("confidence"),
            "_schema_kind": g.get("_schema_kind", "behavior_pattern"),
            "_support_count": support,
            "node": None,  # graph nodes carry no MemoryNode object; the output layer renders from content/layer
        })

    hits.sort(key=lambda h: h["_support_count"], reverse=True)
    return hits
