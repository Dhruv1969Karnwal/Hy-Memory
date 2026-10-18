# -*- coding: utf-8 -*-
"""
HY Memory - Memory Strength (time-decay ranking based on "idle duration")

The first strength ranking signal reflects the "activity" of a memory -- from recency of use plus usage frequency,
rather than its absolute age since creation:

    idle_days = (now - last_accessed_at) / 1 day        # not age (time since creation)
    strength  = (1 + log(access_count)) * exp(-idle_days / tau)   # tau defaults to 180

Rationale: "the user likes Kobe" created in 2023 still matters if it was hit yesterday; age_days would decay it
harshly, while idle_days (since the last access) keeps its strength. Frequent hits (high access_count)
add a further log-frequency boost.

Cold start: when last_accessed_at is empty, idle_days falls back to gmt_created (a new memory has idle≈0
→ full strength). With access_count=0, freq=1.0 (neither punished nor rewarded, avoiding log(0)).

Final ranking: final_score = relevance_score × strength (multiplied in), applied only to the normal
channel (L2/L3/L4); profile (L0/L6) and intention (L7) do not take part.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

DEFAULT_TAU = 180.0


def compute_strength(node: Any, *, now: Optional[datetime] = None, tau: float = DEFAULT_TAU) -> float:
    """
    Compute the strength of a single node.

    strength = (1 + log(access_count)) * exp(-idle_days / tau)
      - access_count < 1 → frequency factor = 1.0 (avoids log(0), neither punished nor rewarded)
      - last_accessed_at empty → idle_days computed from gmt_created (cold start)
      - both empty → idle_days = 0 (full recency)
    """
    if tau <= 0:
        tau = DEFAULT_TAU
    _now = now or datetime.now()

    ac = getattr(node, "access_count", 0) or 0
    if ac < 0:
        ac = 0
    freq = 1.0 + math.log(ac) if ac >= 1 else 1.0

    last = getattr(node, "last_accessed_at", None) or getattr(node, "gmt_created", None)
    idle_days = 0.0
    if last is not None:
        try:
            idle_days = max(0.0, (_now - last).total_seconds() / 86400.0)
        except (TypeError, ValueError):
            idle_days = 0.0

    return freq * math.exp(-idle_days / tau)


def apply_strength_to_normal(
    hits: List[Dict[str, Any]],
    *,
    profile_layers: Optional[Set[str]] = None,
    intention_layers: Optional[Set[str]] = None,
    score_key: str = "score",
    now: Optional[datetime] = None,
    tau: float = DEFAULT_TAU,
) -> List[Dict[str, Any]]:
    """
    Multiply strength into the scores of normal-channel hits in place; profile / intention layers pass through unchanged.

    hits: [{"node": MemoryNode, "score": float, "node_id": ...}, ...]
    profile_layers / intention_layers: sets of layer values excluded from decay (e.g.
        {"l0_basic_info","l6_schema"} / {"l7_intention"}）。
    Returns the same list (already modified in place) so calls can be chained.
    """
    _now = now or datetime.now()
    _profile = profile_layers or set()
    _intention = intention_layers or set()

    for h in hits:
        node = h.get("node")
        if node is None:
            continue
        layer_val = getattr(getattr(node, "layer", None), "value", None) or h.get("layer", "")
        if layer_val in _profile or layer_val in _intention:
            continue
        s = compute_strength(node, now=_now, tau=tau)
        h[score_key] = float(h.get(score_key, 0.0)) * s
    return hits


async def bump_access(
    vector_store: Any,
    items: Iterable[Any],
    *,
    now: Optional[datetime] = None,
) -> int:
    """
    Best-effort write-back of access_count+1 and last_accessed_at=now for hit nodes into the VDB.

    items: an iterable of (node_id, current_access_count) pairs.
    Any write failure is only logged at debug level and never raised (the search response is unaffected). Returns the number of successful writes.
    """
    _now = now or datetime.now()
    ts = int(_now.timestamp())
    ok = 0
    for node_id, current in items:
        if not node_id:
            continue
        try:
            new_count = int(current or 0) + 1
            success = await vector_store.update_payload(
                node_id,
                {"access_count": new_count, "last_accessed_at": ts},
            )
            if success:
                ok += 1
        except Exception as e:
            logger.debug(f"[strength] bump_access failed for {node_id}: {e}")
    return ok
