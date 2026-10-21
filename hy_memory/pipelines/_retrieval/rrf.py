"""
Reciprocal Rank Fusion -- rank-level fusion of multiple retrieval channels.

Formula (Cormack et al. 2009):

    score(d) = Σ_c [ w_c / (k + rank_c(d)) ]

Key properties:
  - Only rank matters, never absolute scores → sidesteps "vec cosine [0,1] is not comparable with BM25 [0,+∞)"
  - Every channel ranks on its own; a doc present in several channels accumulates score
  - Intention weight w_c amplifies a given channel's contribution per scenario
"""

from typing import Any, Dict, List, Optional

from . import config


def rrf_fuse(
    channels: Dict[str, List[Dict[str, Any]]],
    weights: Optional[Dict[str, float]] = None,
    k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fuse multi-channel retrieval results with RRF.

    Args:
        channels: {"vec": [hit, ...], "tag": [hit, ...], "bm25": [hit, ...]}
                  Hits of each channel **must already be sorted by their own score, descending**. Each hit must carry
                  a `node_id` field; a `node` field is strongly recommended (used to hydrate the output layer).
        weights:  Channel weights; absent ones default to 1.0 per channel. Unknown channel names contribute nothing.
        k:        RRF smoothing constant, defaults to config.RRF_K (60).

    Returns:
        Merged list sorted by RRF score descending, each record containing:
          {
            "node_id": ...,
            "node": ...,                # Kept when any channel's original hit had a node field
            "rrf_score": float,         # Fused score (not normalized)
            "rrf_rank_by_channel": {channel: rank, ...},  # For observability
            "per_channel_score": {channel: raw_score, ...},
          }
    """
    weights = weights or {}
    k = k if k is not None else config.RRF_K

    acc: Dict[str, Dict[str, Any]] = {}

    for channel_name, hits in channels.items():
        w = float(weights.get(channel_name, 1.0))
        if w == 0 or not hits:
            continue
        for rank, hit in enumerate(hits, start=1):
            nid = hit.get("node_id")
            if not nid:
                continue
            contrib = w / (k + rank)
            entry = acc.get(nid)
            if entry is None:
                entry = {
                    "node_id": nid,
                    "node": hit.get("node"),
                    "rrf_score": 0.0,
                    "rrf_rank_by_channel": {},
                    "per_channel_score": {},
                }
                acc[nid] = entry
            else:
                # Backfill the node object that earlier channels did not carry
                if entry.get("node") is None and hit.get("node") is not None:
                    entry["node"] = hit.get("node")
            entry["rrf_score"] += contrib
            entry["rrf_rank_by_channel"][channel_name] = rank
            entry["per_channel_score"][channel_name] = float(hit.get("score", 0.0))

    fused = list(acc.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def compute_confidence(fused: List[Dict[str, Any]], top_n: int = 3) -> float:
    """
    Proxy confidence based on the mean top-N RRF score, used for abstain decisions.

    Returns a score in [0, ~1] -- the theoretical RRF ceiling depends on the number of channels and their weights, so
    this is only a relative confidence and callers must not treat it as an exact probability.
    """
    if not fused:
        return 0.0
    n = min(top_n, len(fused))
    avg = sum(h.get("rrf_score", 0.0) for h in fused[:n]) / max(n, 1)
    # Empirical normalization: with single-channel weight 1.0 the top contribution ≈ 1/(60+1) ≈ 0.0164; all three channels at top-1
    # sum to roughly 0.05. Mapping 0.05 onto 1.0 gives a rough scaling
    return min(1.0, avg / 0.05)
