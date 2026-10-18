"""One agreed reading of read-side time filtering (created_after).

`created_after` means "the event described by the memory happened after that time", so filtering runs on the **event time**
`memory_at` and only falls back to the persist time `gmt_created` when `memory_at` is empty.
Filtering by gmt_created is effectively void in bulk import scenarios (memories from one import share almost the same gmt_created),
see `data/vector_store_base.node_effective_time`.

Division of work:
- **Push down what can be pushed down**: `vector_store.search()` accepts `created_after` so the VDB filters first, saving recall volume.
- **Uniform fallback at the end**: `keyword_search()` (BM25), the graph channel and the others do not all support that parameter,
  so every reader passes its final results through the gate in this module once more, ensuring no channel can bypass the filter.
"""

from typing import Any, Dict, List, Optional


def hit_effective_ts(hit: Dict[str, Any]) -> Optional[float]:
    """Return the effective timestamp of one recall hit, in unix seconds.

    Prefers `node.memory_at` and falls back to `node.gmt_created`; when node is None (graph channel result)
    the fields with the same names on the hit dict are used instead. Returns None when neither is available.
    """
    node = hit.get("node")

    for attr in ("memory_at", "gmt_created"):
        v = getattr(node, attr, None) if node is not None else hit.get(attr)
        if v is None or v == "":
            continue
        ts = getattr(v, "timestamp", None)
        if callable(ts):
            try:
                return float(ts())
            except (TypeError, ValueError, OSError):
                continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def filter_hits_by_created_after(
    hits: List[Dict[str, Any]],
    created_after: Optional[float],
) -> List[Dict[str, Any]]:
    """Filter recall hits by created_after.

    Hits whose time field cannot be resolved at all are **kept** -- recalling too much is better than silently
    dropping data over a missing field (graph channel results and legacy data may carry no time field).
    """
    if not created_after or not hits:
        return hits
    threshold = float(created_after)
    return [
        h for h in hits
        if (ts := hit_effective_ts(h)) is None or ts >= threshold
    ]
