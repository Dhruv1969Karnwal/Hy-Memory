"""Read side: how the memory relation graph takes part in scoring.

Three uses need three different pieces of maths; the most common mistake is one weighting scheme doing all three:

| Usage | What it does | Which failure it fixes | Adds new candidates |
|---|---|---|---|
| **A** `propagate_support` | Reranks **inside the widened pre-truncation pool** | Ordering is noise when scores are close | **No** |
| **B** `relax_admission` | Lets candidates backed by strong hits **lower the admission bar** | Related but low similarity, blocked by the threshold | Yes (bounded) |
| **C** `find_relation_paths` | Emits the **path** joining two hits | Relation semantics are lost in a flat list | A few |

Key trade-offs (details in `docs/pro_graph_design.md` §3):

1. **A must act on the widened pool before truncation.** Reordering an already chosen top-10 buys little;
   the value is re-deciding which 10 of several dozen candidates make the final result.
2. **B never invents graph scores for expanded candidates.** Graph scores and semantic scores have different units, and hard fusion is just slapping coefficients onto incomparable things.
   The right move is for the graph to only relax the admission threshold dynamically; the final score remains the
   candidate's own similarity to the query -- the graph fixes **missed recall**, not relevance, so it can never lift genuinely irrelevant content into the result.
3. **Shared tags are not traversal edges** (τ=0). A tag is a classification relation, not an association: 200 memories under one `work` tag
   are all "equally related" -- weak and uniform, and frequent tags turn into hubs. The right use of tags on the read side
   is a **scoring channel** (query BM25 terms ∩ mem.tags, weighted above co_occur) plus filter/boost.
4. **Hard gate for usage B**: `sim(cand) ≥ seed_sim − δ` (default δ=0.2), replacing the old
   `min_score*(1−γ*support)` slack threshold. Once an expansion passes the gate it is merged into the seed chronologically (`\\n\\n`),
   and takes no top-k slot of its own.
"""

import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Per-edge-type prior τ. The test is **whether this edge can add information vector search cannot obtain**.
TYPE_PRIOR: Dict[str, float] = {
    # LLM-labelled: cosine at both ends is often low and vector search would never find it -> most informative
    "causes": 0.9,
    "contrasts": 0.7,
    "elaborates": 0.9,
    # Independent dated occurrences of the same concrete activity/event type.
    "same_series": 0.85,
    # Temporal spine: ordered progression inside one Topic, directed. Answers "after/before X"
    "temporal_next": 0.7,
    # Written together: still part of neighbourhood expansion, but the prior is below typed edges, and the co_occur
    # channel is weighted lower in the final score too (versus tag∩keyword). Noise is held by degree normalisation + the seed_sim gate.
    "co_occurs": 0.55,
    # Deterministic: it records exactly "semantically close", which is what the vector index already knows -> highly redundant
    "semantic_neighbor": 0.35,
    # Same episode: one discussion (across write batches). Not carried by edges, framed by the episode_id field;
    # essentially co_occurs stretched across batches, so the prior matches it
    "episode": 0.55,
    # Fallback (untyped RELATED_TO written by System 2 in old data)
    "related": 0.5,
}
_DEFAULT_PRIOR = 0.5

# Graph node statuses always filtered on the read side: memories logically deleted by dedup must not loop back in through the graph
_EXCLUDED_STATUS = {"shadow"}


def edge_prior(relation_type: Optional[str]) -> float:
    return TYPE_PRIOR.get((relation_type or "").lower(), _DEFAULT_PRIOR)


def filter_relation_edges(
    edges: Iterable[Dict[str, Any]], *, include_same_series: bool,
) -> List[Dict[str, Any]]:
    """Apply feature gates before graph edges affect ranking or admission."""
    return [
        edge for edge in (edges or [])
        if include_same_series
        or str(edge.get("relation_type") or "").lower() != "same_series"
    ]


def build_adjacency(
    edges: Iterable[Dict[str, Any]],
    *,
    exclude_status: Optional[Set[str]] = None,
) -> Dict[str, List[Tuple[str, str, float]]]:
    """Build an undirected adjacency table from the result of `graph_store.get_related_edges()`.

    One node pair can carry several relations (both `co_occurs` and `causes`). These are **folded into a single
    edge keeping the highest effective weight**: degree should count distinct neighbours, not parallel edges,
    otherwise hub down-weighting (`sqrt(deg)`) gets diluted by parallel edges and support is counted twice.

    Returns:
        {node_id: [(neighbor_id, relation_type, effective_weight), ...]}
        `effective_weight` = edge weight × type prior τ; the formulas below use it directly.
    """
    excluded = exclude_status if exclude_status is not None else _EXCLUDED_STATUS
    # (node, neighbor) -> (relation_type, effective_weight), keeping only the strongest
    best: Dict[Tuple[str, str], Tuple[str, float]] = {}

    for e in edges or []:
        src, tgt = e.get("source_id"), e.get("target_id")
        if not src or not tgt or src == tgt:
            continue
        if (e.get("target_status") or "").lower() in excluded:
            continue
        # Directed edges (TEMPORAL_NEXT) can be hit from either end, so shadow must be filtered on the source side too
        if (e.get("source_status") or "").lower() in excluded:
            continue
        rtype = e.get("relation_type") or "related"
        try:
            w = float(e.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        eff = max(0.0, min(w, 1.0)) * edge_prior(rtype)
        # The adjacency table itself is undirected (A/B only care about "being next to each other"); direction is kept separately in forward_dirs
        for a, b in ((src, tgt), (tgt, src)):
            prev = best.get((a, b))
            if prev is None or eff > prev[1]:
                best[(a, b)] = (rtype, eff)

    adj: Dict[str, List[Tuple[str, str, float]]] = {}
    for (a, b), (rtype, eff) in best.items():
        adj.setdefault(a, []).append((b, rtype, eff))
    return adj


def collect_directed_edges(edges: Iterable[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """Collect the forward (source, target) set of directed edges (currently only `temporal_next`).

    The adjacency table is deliberately undirected because usages A/B only care about adjacency. Usage C emits
    paths and has to state direction, otherwise "what happened after X" and "before X" cannot be told apart.
    """
    out: Set[Tuple[str, str]] = set()
    for e in edges or []:
        if not e.get("directed"):
            continue
        src, tgt = e.get("source_id"), e.get("target_id")
        if src and tgt and src != tgt:
            out.add((src, tgt))
    return out


# ================================================================
# Usage A: support propagation inside the pool
# ================================================================

def propagate_support(
    scores: Dict[str, float],
    adjacency: Dict[str, List[Tuple[str, str, float]]],
    *,
    beta: float = 0.15,
    iterations: int = 1,
    support_cap: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Graph-regularized reranking **inside the already recalled pool**.

        support(v) = Σ_{u ∈ N(v) ∩ P, u≠v}  eff_w(u,v) · s(u)  /  sqrt(deg_pool(v))
        s'(v)      = s0(v) + beta · min(support(v), support_cap)

    All three guardrails sit in the formula:
      - `∩ P` (propagates only inside the pool) -> no new candidates, no hub fan-out;
      - `sqrt(deg_pool(v))` degree normalisation -> the direct antidote to tag hubs (the graph analogue of IDF);
      - small `beta` + `support_cap` -> this is a **nudge**, not a rewrite of the ranking. A beta large enough
        to overturn the order would mean the graph replaces retrieval, and precision would certainly drop.

    Args:
        scores: {node_id: s0}, the direct recall scores (**the widened pool before truncation**)
        adjacency: the product of build_adjacency()
        iterations: propagation rounds, 1~2 is enough; more has diminishing returns and starts to blur

    Returns:
        (new scores dict, support dict). support is returned separately for observability and reuse by usage B.
    """
    if not scores or not adjacency or beta <= 0:
        return dict(scores), {}

    pool = set(scores)
    # Degree inside the pool (only edges whose both ends are in the pool)
    deg: Dict[str, int] = {}
    for v in pool:
        deg[v] = sum(1 for (nb, _t, _w) in adjacency.get(v, []) if nb in pool)

    s0 = dict(scores)
    cur = dict(scores)
    support: Dict[str, float] = {}

    for _ in range(max(1, iterations)):
        nxt: Dict[str, float] = {}
        for v in pool:
            raw = 0.0
            for nb, _rtype, eff_w in adjacency.get(v, []):
                if nb == v or nb not in pool:
                    continue
                raw += eff_w * cur.get(nb, 0.0)
            d = deg.get(v, 0)
            sup = raw / (d ** 0.5) if d > 0 else 0.0
            support[v] = sup
            nxt[v] = s0[v] + beta * min(sup, support_cap)
        cur = nxt

    return cur, support


# ================================================================
# Usage B: threshold relaxation (the graph only lowers the bar, it never invents relevance)
# ================================================================

# Edge type groups, used to hand out expansion quotas per type.
#
# Temporal edges are the most numerous and least informative (same-day writes need not be related); relational
# edges are the fewest but most informative (`causes` endpoints can be vector-dissimilar, exactly what vector
# search can never find). Without quotas temporal edges crowd relational edges out of the top-N, the same class of problem as a tag hub drowning the result.
_RELATIONAL_TYPES = {"causes", "elaborates", "contrasts", "same_series",
                     "co_occurs", "related",
                     "episode"}
_TEMPORAL_TYPES = {"temporal_next"}


def edge_group(relation_type: Optional[str]) -> str:
    rt = (relation_type or "").lower()
    if rt in _TEMPORAL_TYPES:
        return "temporal"
    if rt == "semantic_neighbor":
        return "semantic"
    return "relational"


def normalize_by_degree(
    adjacency: Dict[str, List[Tuple[str, str, float]]],
) -> Dict[str, List[Tuple[str, str, float]]]:
    """Devalue edge weights by degree: `eff_w /= sqrt(deg(neighbor))`.

    Every edge of a highly connected node (many memories written the same day, or a topic hub) is less discriminative.
    This is the graph analogue of IDF and the precondition for temporal edges joining expansion safely;
    otherwise one productive day pulls in every memory of that day with equal weight.
    """
    deg = {n: len(nbs) for n, nbs in (adjacency or {}).items()}
    out: Dict[str, List[Tuple[str, str, float]]] = {}
    for node, nbs in (adjacency or {}).items():
        scaled = []
        for nb, rtype, eff in nbs:
            d = deg.get(nb, 1) or 1
            scaled.append((nb, rtype, eff / (d ** 0.5)))
        out[node] = scaled
    return out


def reachable_within(
    seed_scores: Dict[str, float],
    adjacency: Dict[str, List[Tuple[str, str, float]]],
    *,
    hops: int = 2,
    seed_floor: float = 0.6,
    hop_decay: float = 0.5,
    max_nodes: int = 200,
    multi_seed_bonus: float = 0.5,
) -> Dict[str, Tuple[float, str, str, int]]:
    """Breadth-first search from high-scoring seeds, returning structural support for nodes reachable outside the pool.

    This is the **relational multi-hop** implementation: `causes` / `co_occurs` / `temporal_next` edges can be
    chained, and the two ends of a question like "which restaurant did the friend recommend" often sit one intermediate node apart.

    Support decays along the path: `support = Π(eff_w) * hop_decay^(hop-1) * s0(seed)`.
    Decay plus a node budget are mandatory -- 2 hops without constraints pulls in the whole connected component.

    **Multi-seed corroboration**: one seed pointing at a candidate may be coincidence; **two different seeds pointing
    at it at once** is real corroboration (the same mutual-corroboration logic usage A applies inside the pool).
    So support is not a plain max but "strongest path + remaining seeds' contribution * multi_seed_bonus".
    Example: a query hits both "eating" and "cycling", both point at "park" -> park rises noticeably.

    Returns:
        {node_id: (support, via_seed, relation_type_of_first_edge, hop)}
    """
    if hops < 1 or not seed_scores or not adjacency:
        return {}

    in_pool = set(seed_scores)
    seeds = sorted(
        [(nid, s) for nid, s in seed_scores.items() if s >= seed_floor],
        key=lambda x: x[1], reverse=True,
    )
    if not seeds:
        return {}

    # cand -> {seed: (best_support_from_that_seed, first_rel, hop)}
    per_seed: Dict[str, Dict[str, Tuple[float, str, int]]] = {}
    # frontier: (node, accumulated weight, via_seed, first-hop relation, hop)
    frontier: List[Tuple[str, float, str, str, int]] = [
        (nid, s, nid, "", 0) for nid, s in seeds
    ]
    visited = set(in_pool)

    for hop in range(1, hops + 1):
        nxt: List[Tuple[str, float, str, str, int]] = []
        for cur, acc, via, first_rel, _h in frontier:
            for nb, rtype, eff_w in adjacency.get(cur, []):
                if nb == cur:
                    continue
                # Hop decay: the first hop does not decay, each later hop multiplies by hop_decay
                sup = acc * eff_w * (hop_decay ** (hop - 1))
                rel = first_rel or rtype
                if nb not in in_pool:
                    slot = per_seed.setdefault(nb, {})
                    prev = slot.get(via)
                    if prev is None or sup > prev[0]:
                        slot[via] = (sup, rel, hop)
                if nb not in visited and hop < hops:
                    visited.add(nb)
                    nxt.append((nb, acc * eff_w, via, rel, hop))
            if len(per_seed) >= max_nodes:
                break
        if len(per_seed) >= max_nodes:
            break
        frontier = nxt
        if not frontier:
            break

    # Merge: strongest path in full + discounted contributions of the other seeds (bounded, so seeds cannot stack forever)
    best: Dict[str, Tuple[float, str, str, int]] = {}
    for cand, slots in per_seed.items():
        ranked = sorted(slots.items(), key=lambda kv: kv[1][0], reverse=True)
        top_seed, (top_sup, top_rel, top_hop) = ranked[0]
        extra = sum(s[0] for _sid, s in ranked[1:])
        best[cand] = (
            top_sup + multi_seed_bonus * extra, top_seed, top_rel, top_hop,
        )
    return best


def apply_type_quota(
    reach: Dict[str, Tuple[float, str, str, int]],
    *,
    top_n: int,
    quotas: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[float, str, str, int]]:
    """Allocate the top-N quota per edge type, then truncate by support.

    Why grouping is mandatory: temporal edges are the most numerous and least informative, relational edges the
    fewest but most informative (`causes` endpoints may have very low cosine and vector search would never find them).
    Without a quota, temporal edges would push relational edges out of the top-N.

    This step is **free** (pure local ordering) and must come before "batch-fetch vectors and compute similarity":
    similarity is the only step in the whole pipeline that carries an I/O cost.
    """
    if top_n <= 0 or not reach:
        return {}
    q = quotas or {"relational": 0.5, "temporal": 0.3, "semantic": 0.2}

    grouped: Dict[str, List[Tuple[str, Tuple[float, str, str, int]]]] = {}
    for cand, val in reach.items():
        grouped.setdefault(edge_group(val[2]), []).append((cand, val))

    out: Dict[str, Tuple[float, str, str, int]] = {}
    for group, items in grouped.items():
        items.sort(key=lambda kv: kv[1][0], reverse=True)
        budget = max(1, int(round(top_n * q.get(group, 0.0))))
        for cand, val in items[:budget]:
            out[cand] = val

    # When the quotas are not filled, top up with the globally strongest candidates
    if len(out) < top_n:
        rest = sorted(
            ((c, v) for c, v in reach.items() if c not in out),
            key=lambda kv: kv[1][0], reverse=True,
        )
        for cand, val in rest[: top_n - len(out)]:
            out[cand] = val
    return out


def episode_support(
    hits: List[Dict[str, Any]],
    episode_members: Dict[str, List[str]],
    *,
    seed_floor: float = 0.6,
    weight: float = 0.7,
    multi_seed_bonus: float = 0.5,
) -> Dict[str, Tuple[float, str, str, int]]:
    """Turn episode membership into structural support, shaped like `reachable_within` so it enters the same funnel.

    An episode is one discussion (spanning write batches, split by time gaps inside the SDK). Memories in the same
    episode are genuinely related, yet their vectors may be dissimilar -- exactly the part vector search cannot find.

    Why a field rather than edges: an episode holds 10-30 memories, pairwise linking would mean 45-435 edges;
    one `episode_id` plus one filtered query suffices, and it **does not depend on the graph** (works with graph disabled).

    Difference from tag bridging (τ=0, rejected): a `work` tag can hold 200 memories across years, a **classification
    relation**; an episode holds 10-30 memories from one discussion, an **episodic relation** -- essentially co_occurs stretched across batches.

        support(c) = weight / sqrt(episode_size) · s0(seed)

    `sqrt(episode_size)` normalisation: the bigger the episode, the less discriminative one membership is (same reasoning as tag hubs).
    Multi-seed corroboration applies as well: when two high-scoring hits belong to one episode, that episode's members are more credible.

    Args:
        hits: direct hits (used to pick seeds and to exclude what is already in the pool)
        episode_members: {episode_id: [member_node_id, ...]}, framed by the caller out of the VDB

    Returns:
        {node_id: (support, via_seed, "episode", 1)}
    """
    if not hits or not episode_members:
        return {}

    in_pool = {h.get("node_id") for h in hits if h.get("node_id")}
    # episode_id -> the highest-scoring seeds inside that episode
    seeds_by_ep: Dict[str, List[Tuple[str, float]]] = {}
    for h in hits:
        score = float(h.get("score", 0.0))
        if score < seed_floor:
            continue
        node = h.get("node")
        ep = (getattr(node, "episode_id", "") or "") if node is not None else ""
        if not ep or ep not in episode_members:
            continue
        seeds_by_ep.setdefault(ep, []).append((h.get("node_id") or "", score))

    out: Dict[str, Tuple[float, str, str, int]] = {}
    for ep, seeds in seeds_by_ep.items():
        members = [m for m in episode_members.get(ep, []) if m and m not in in_pool]
        if not members:
            continue
        size = max(2, len(episode_members.get(ep, [])))
        base = weight / (size ** 0.5)
        seeds.sort(key=lambda x: x[1], reverse=True)
        top_seed, top_score = seeds[0]
        extra = sum(s for _sid, s in seeds[1:])
        sup = base * (top_score + multi_seed_bonus * extra)
        for m in members:
            prev = out.get(m)
            if prev is None or sup > prev[0]:
                out[m] = (sup, top_seed, "episode", 1)
    return out


def blend_support(
    graph_reach: Dict[str, Tuple[float, str, str, int]],
    episode_reach: Dict[str, Tuple[float, str, str, int]],
    *,
    lam: float = 0.3,
    support_cap: float = 1.0,
) -> Dict[str, Tuple[float, str, str, int]]:
    """**Boundedly stack** the support from graph reachability and episode membership instead of taking a max.

    Why not max: these are **two independent observations**. `co_occurs` says the two came from the same passage,
    episode membership says they belong to the same affair even across several writes. When both hold, the
    candidate is more credible than when only one holds; max would declare that an extra independent piece of evidence adds nothing.

    Max also **systematically favours the coarser signal**: episode support is pressed down by `sqrt(episode_size)`
    normalisation, so it is naturally below `co_occurs` and max almost always picks the graph edge -- the episode
    channel then does nothing for candidates that already have co-occurrence edges, working only on cross-batch candidates without them.

        support = min(support_cap, s_graph + lam · s_episode)

    A discount factor rather than a plain sum stops two weak pieces of evidence stacking into a strong conclusion;
    the `support_cap` ceiling guarantees the admission bar drops at most to `(1 - gamma) * min_score`, so the guardrail holds.
    This is the same logic as multi-seed corroboration in `reachable_within` (strongest in full, the rest discounted).

    Attribution (via_seed / relation_type / hop) is taken from the side with **stronger support**, for observability.
    """
    if not episode_reach:
        return dict(graph_reach or {})
    out: Dict[str, Tuple[float, str, str, int]] = dict(graph_reach or {})
    for cand, (ep_sup, ep_seed, ep_type, ep_hop) in episode_reach.items():
        prev = out.get(cand)
        if prev is None:
            out[cand] = (min(ep_sup, support_cap), ep_seed, ep_type, ep_hop)
            continue
        g_sup, g_seed, g_type, g_hop = prev
        merged = min(g_sup + lam * ep_sup, support_cap)
        if ep_sup > g_sup:
            out[cand] = (merged, ep_seed, ep_type, ep_hop)
        else:
            out[cand] = (merged, g_seed, g_type, g_hop)
    return out


def apply_temporal_affinity(
    reach: Dict[str, Tuple[float, str, str, int]],
    *,
    times: Dict[str, float],
    tau_days: float = 7.0,
    floor: float = 0.3,
) -> Dict[str, Tuple[float, str, str, int]]:
    """Scale structural support by a **temporal affinity** factor: the closer to the anchor in time, the likelier the link is real.

    This is the second scoring dimension (the first is content: edge weight times the candidate's own similarity to the query).

    The motivation is a write like "I went to the park today and cycled there, and I also got something to eat today":
    park / cycling / eating were recorded **together**. Later, asking "where was the place I ate at last week",
    vector search may retrieve only "eating" and miss "park", yet the very fact that they were written at the same moment
    is a strong signal. Same-batch `co_occurs` edges naturally have Δt≈0, so the factor is 1.

        factor = clamp(exp(-Δdays / tau_days), floor, 1.0)

    `floor` is not 0: edges that are temporally distant but relationally strong (such as an LLM-labelled `causes`) should not be flattened by time.
    Nodes whose time cannot be read are treated as 1.0 -- a missing field is not punished.
    """
    if not reach or not times:
        return reach
    out: Dict[str, Tuple[float, str, str, int]] = {}
    tau_sec = max(1.0, tau_days * 86400.0)
    for cand, (sup, via, rel, hop) in reach.items():
        t_c, t_s = times.get(cand), times.get(via)
        if t_c is None or t_s is None:
            out[cand] = (sup, via, rel, hop)
            continue
        factor = math.exp(-abs(t_c - t_s) / tau_sec)
        factor = max(floor, min(1.0, factor))
        out[cand] = (sup * factor, via, rel, hop)
    return out


def temporal_affinity_factor(
    t_cand: Optional[float],
    t_seed: Optional[float],
    *,
    tau_days: float = 7.0,
    floor: float = 0.3,
) -> float:
    """Temporal affinity for a single (cand, seed) pair ∈ [floor, 1]; a missing time yields 1.0."""
    if t_cand is None or t_seed is None:
        return 1.0
    tau_sec = max(1.0, tau_days * 86400.0)
    factor = math.exp(-abs(t_cand - t_seed) / tau_sec)
    return max(floor, min(1.0, factor))


_TERM_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+", re.I)


def tokenize_query_terms(query: str) -> List[str]:
    """Lightweight tokenization for the BM25/tag channel (no LLM; usable on the single-query eval path)."""
    if not query:
        return []
    return [t.lower() for t in _TERM_RE.findall(query) if len(t) > 1]


def keyword_content_overlap(terms: List[str], content: str) -> float:
    """Share of query terms present in content ∈ [0, 1]."""
    if not terms or not content:
        return 0.0
    blob = content.lower()
    hit = sum(1 for t in terms if t in blob)
    return hit / len(terms)


def tag_keyword_overlap(terms: List[str], tags: Optional[Iterable[str]]) -> float:
    """Coverage of query BM25 terms ∩ mem.tags (case-insensitive) ∈ [0, 1].

    The tag channel must outweigh co_occur: proper nouns and topic words align better than weak co-occurrence edges.
    """
    if not terms or not tags:
        return 0.0
    tag_set = {str(t).lower().strip() for t in tags if t}
    if not tag_set:
        return 0.0
    # Exact intersection plus tag-contained-in-term / term-contained-in-tag (travel ↔ traveling)
    hit = 0
    for t in terms:
        if t in tag_set:
            hit += 1
            continue
        if any(t in tag or tag in t for tag in tag_set if len(tag) > 2):
            hit += 1
    return hit / len(terms)


def score_expanded_candidate(
    *,
    sim: float,
    bm25: float = 0.0,
    tag_overlap: float = 0.0,
    temporal: float = 1.0,
    support: float = 0.0,
    relation_type: Optional[str] = None,
    w_sem: float = 1.0,
    w_bm25: float = 0.3,
    w_tag: float = 0.5,
    w_temporal: float = 0.2,
    w_co_occur: float = 0.15,
    w_struct: float = 0.25,
) -> float:
    """Multi-channel fusion score for an expanded candidate (no LLM).

        score = w_sem·sim + w_bm25·bm25 + w_tag·tag
              + w_temporal·temporal + w_edge·min(support,1)

    `co_occurs` / `episode` use the lower `w_co_occur`; typed edges use `w_struct`.
    """
    rtype = (relation_type or "").lower()
    if rtype in ("co_occurs", "episode"):
        struct = w_co_occur * min(max(support, 0.0), 1.0)
    else:
        struct = w_struct * min(max(support, 0.0), 1.0)
    return (
        w_sem * float(sim)
        + w_bm25 * float(bm25)
        + w_tag * float(tag_overlap)
        + w_temporal * float(temporal)
        + struct
    )


def relax_admission(
    *,
    seed_scores: Dict[str, float],
    adjacency: Dict[str, List[Tuple[str, str, float]]],
    candidate_sims: Dict[str, float],
    min_score: float = 0.0,
    gamma: float = 0.3,
    seed_floor: float = 0.6,
    max_admit: int = 20,
    hops: int = 1,
    hop_decay: float = 0.5,
    boost_beta: float = 0.0,
    boost_cap: float = 1.0,
    reach: Optional[Dict[str, Tuple[float, str, str, int]]] = None,
    seed_sim_delta: float = 0.2,
    candidate_bm25: Optional[Dict[str, float]] = None,
    candidate_tag: Optional[Dict[str, float]] = None,
    candidate_temporal: Optional[Dict[str, float]] = None,
    channel_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Recover missed candidates from the k-hop neighbourhood of high-scoring seeds and **rescore** them.

    **Hard gate (usage B admission)**:

        admit(c) iff sim(c) ≥ seed_sim(via) − δ     (default δ=0.2)

    `seed_sim` is the score against the query of the via_seed that pulled c in. More intuitive than the old
    `min_score*(1−γ*support)`: an expansion must not fall far below its own seed.

    `gamma ≤ 0` still means usage B is **off** (compatible with `MEMORY_GRAPH_RELAX_GAMMA=0`).

    **Multi-channel scoring** (enabled once the optional feature dicts are supplied; otherwise falls back to `sim + boost_beta*support`):

        semantic + BM25(content) + tag∩keyword (high) + temporal + co_occur/structure (low)

    Args:
        candidate_sims: {candidate_id: sim(query, candidate)}
        seed_sim_delta: the hard-gate δ, env `MEMORY_GRAPH_SEED_SIM_DELTA`
        candidate_bm25 / candidate_tag / candidate_temporal: optional channel features ∈ [0,1]
        channel_weights: {sem,bm25,tag,temporal,co_occur,struct} overriding the default weights
        reach: optional precomputed reachability set (already scaled by temporal affinity)

    Returns:
        [{"node_id", "score", "sim", "support", "threshold", "via_seed",
          "relation_type", "hop", ...}], sorted by score descending, at most max_admit entries.
    """
    if gamma <= 0 or not seed_scores or not candidate_sims:
        return []
    # Also allowed with no adjacency and no precomputed reach (for instance episode-only)
    if reach is None and not adjacency:
        return []

    if reach is None:
        reach = reachable_within(
            seed_scores, adjacency,
            hops=max(1, hops), seed_floor=seed_floor, hop_decay=hop_decay,
        )

    delta = max(0.0, float(seed_sim_delta))
    use_channels = bool(
        candidate_bm25 is not None
        or candidate_tag is not None
        or candidate_temporal is not None
        or channel_weights
    )
    cw = channel_weights or {}

    admitted: List[Dict[str, Any]] = []
    for cand_id, (sup, seed_id, rtype, hop) in reach.items():
        if cand_id not in candidate_sims:
            continue
        sim = float(candidate_sims.get(cand_id, 0.0))
        seed_sim = float(seed_scores.get(seed_id, 0.0))
        threshold = seed_sim - delta
        if sim < threshold:
            continue

        bm25 = float((candidate_bm25 or {}).get(cand_id, 0.0))
        tag_ov = float((candidate_tag or {}).get(cand_id, 0.0))
        temporal = float((candidate_temporal or {}).get(cand_id, 1.0))

        if use_channels:
            score = score_expanded_candidate(
                sim=sim, bm25=bm25, tag_overlap=tag_ov, temporal=temporal,
                support=sup, relation_type=rtype,
                w_sem=float(cw.get("sem", 1.0)),
                w_bm25=float(cw.get("bm25", 0.3)),
                w_tag=float(cw.get("tag", 0.5)),
                w_temporal=float(cw.get("temporal", 0.2)),
                w_co_occur=float(cw.get("co_occur", 0.15)),
                w_struct=float(cw.get("struct", 0.25)),
            )
        else:
            score = sim + boost_beta * min(sup, boost_cap)

        admitted.append({
            "node_id": cand_id,
            "score": score,
            "sim": sim,
            "support": sup,
            "threshold": threshold,
            "seed_sim": seed_sim,
            "via_seed": seed_id,
            "relation_type": rtype,
            "hop": hop,
            "bm25": bm25,
            "tag_overlap": tag_ov,
            "temporal": temporal,
        })

    admitted.sort(key=lambda x: x["score"], reverse=True)
    return admitted[:max_admit]


def _item_time_seconds(item: Dict[str, Any]) -> float:
    node = item.get("node")
    if node is None:
        return 0.0
    value = getattr(node, "memory_at", None) or getattr(node, "gmt_created", None)
    if value is None:
        return 0.0
    try:
        return float(value.timestamp())
    except Exception:
        return 0.0


def _canonical_content(item: Dict[str, Any]) -> str:
    node = item.get("node")
    if node is not None and getattr(node, "content", None):
        return str(node.content)
    return str(item.get("content") or "")


def merge_expansions_into_seeds(
    hits: List[Dict[str, Any]],
    admitted_hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge gate-passing expansions **into their seed chronologically**, separated by ``\\n\\n``.

    - The synthesis still occupies exactly **1** top-k slot (the seed keeps its original score and rank)
    - A candidate is merged only into the seed whose via is strongest (admitted entries are already single-via)
    - Block order: seed plus expansions, all sorted ascending by `memory_at` (`gmt_created` when missing)
    - Nothing is written back to the VDB; only the content returned on the read side changes
    """
    if not admitted_hits:
        return hits

    by_seed: Dict[str, List[Dict[str, Any]]] = {}
    for a in admitted_hits:
        via = a.get("graph_via") or a.get("via_seed")
        if not via:
            continue
        by_seed.setdefault(via, []).append(a)

    if not by_seed:
        return hits

    hit_by_id = {h.get("node_id"): h for h in hits if h.get("node_id")}

    for seed_id, exps in by_seed.items():
        seed_hit = hit_by_id.get(seed_id)
        if seed_hit is None:
            continue
        existing = list(seed_hit.get("_graph_merged_items") or [])
        if not existing:
            existing = [seed_hit]
        elif not any(piece.get("node_id") == seed_id for piece in existing):
            existing.insert(0, seed_hit)
        pieces = existing + list(exps)
        deduped = []
        seen_ids: Set[str] = set()
        seen_contents: Set[str] = set()
        for piece in pieces:
            piece_id = str(piece.get("node_id") or "")
            piece_content = " ".join(_canonical_content(piece).split()).casefold()
            if ((piece_id and piece_id in seen_ids)
                    or (piece_content and piece_content in seen_contents)):
                continue
            if piece_id:
                seen_ids.add(piece_id)
            if piece_content:
                seen_contents.add(piece_content)
            deduped.append(piece)
        pieces = deduped
        pieces.sort(key=_item_time_seconds)
        merged_items = [
            {**piece, "content": _canonical_content(piece)}
            for piece in pieces
        ]
        merged = "\n\n".join(
            c for c in (_canonical_content(p) for p in pieces) if c
        )
        if not merged:
            continue
        # Written onto the hit: downstream response assembly prefers item["content"] over node.content
        seed_hit["content"] = merged
        seed_hit["graph_merged"] = True
        seed_hit["_graph_merged_items"] = merged_items
        seed_hit["graph_merged_ids"] = [
            e.get("node_id") for e in exps if e.get("node_id")
        ]
        # Keep the strongest expansion's observability fields for tracing
        best = max(exps, key=lambda x: float(x.get("score", 0.0)))
        seed_hit["graph_support"] = best.get("graph_support") or best.get("support")
        seed_hit["graph_relation"] = best.get("graph_relation") or best.get("relation_type")
        if all(
            (e.get("graph_relation") or e.get("relation_type")) == "same_series"
            for e in exps
        ):
            seed_hit["_timeline_member_ids"] = [
                piece.get("node_id") for piece in merged_items if piece.get("node_id")
            ]
            seed_hit["_timeline"] = [
                {
                    "memory_id": piece.get("node_id"),
                    "memory_at": _item_time_seconds(piece),
                }
                for piece in merged_items if piece.get("node_id")
            ]

    # Expansions are no longer returned as separate hits
    return hits


def annotate_same_series_in_pool(
    hits: List[Dict[str, Any]],
    adjacency: Dict[str, List[Tuple[str, str, float]]],
) -> List[Dict[str, Any]]:
    """Mark explicit same-series neighbors that are already query candidates."""
    pool_ids = {item.get("node_id") for item in hits if item.get("node_id")}
    for item in hits:
        node_id = item.get("node_id")
        if not node_id:
            continue
        members = sorted({
            neighbor for neighbor, relation, _weight in adjacency.get(node_id, [])
            if relation == "same_series" and neighbor in pool_ids
        })
        if members:
            item["_same_series_pool_ids"] = members
    return hits


def merge_same_series_in_pool(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold connected same-series candidates into one chronological L2 slot."""
    by_id = {item.get("node_id"): item for item in hits if item.get("node_id")}
    adjacency = {
        node_id: {
            member for member in item.get("_same_series_pool_ids") or []
            if member in by_id
        }
        for node_id, item in by_id.items()
    }
    visited: Set[str] = set()
    remove: Set[str] = set()
    for start in list(adjacency):
        if start in visited or not adjacency[start]:
            continue
        component: Set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency.get(current, set()) - visited)
        if len(component) < 2:
            continue
        members = [by_id[node_id] for node_id in component]
        if any(member.get("graph_merged") for member in members):
            continue
        seed = max(
            members,
            key=lambda item: (float(item.get("score", 0.0)), -hits.index(item)),
        )
        pieces = sorted(members, key=_item_time_seconds)
        seed["content"] = "\n\n".join(
            text for text in (_canonical_content(piece) for piece in pieces) if text
        )
        seed["graph_merged"] = True
        seed["graph_relation"] = "same_series"
        seed["_graph_merged_items"] = [
            {**piece, "content": _canonical_content(piece)} for piece in pieces
        ]
        seed["graph_merged_ids"] = [
            piece.get("node_id") for piece in pieces
            if piece.get("node_id") and piece is not seed
        ]
        seed["_timeline_member_ids"] = [
            piece.get("node_id") for piece in pieces if piece.get("node_id")
        ]
        seed["_timeline"] = [
            {
                "memory_id": piece.get("node_id"),
                "memory_at": _item_time_seconds(piece),
            }
            for piece in pieces if piece.get("node_id")
        ]
        remove.update(component - {seed.get("node_id")})
    return [item for item in hits if item.get("node_id") not in remove]


# ================================================================
# Usage C: relation paths (an independent output channel, kept out of the main ranking)
# ================================================================

def find_relation_paths(
    *,
    hit_scores: Dict[str, float],
    adjacency: Dict[str, List[Tuple[str, str, float]]],
    high_floor: float = 0.6,
    max_paths: int = 5,
    typed_only: bool = True,
    forward_dirs: Optional[Set[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Find paths of at most 2 hops connecting two high-scoring hits.

    A relational question ("who is the friend that recommended that restaurant") wants a **path**, not a sorted list.
    Mixing path nodes into a flat recall list drops the relational semantics: the answering LLM sees a few unrelated memories.
    That is why this is returned as its own field and **takes no part in the main ranking**.

    Intermediate nodes (none on a 1-hop path; the bridge on a 2-hop path) are included even when their own score is low -- they are part of the path.

    Args:
        hit_scores: the direct hits {node_id: score}
        high_floor: only look for paths between hits scoring ≥ this value
        typed_only: only high-information edges (semantic_neighbor excluded -- it merely states "semantically close",
            and a path between two semantically close memories has no explanatory value for the answer)
        forward_dirs: the product of `collect_directed_edges()`. Directed edges (`temporal_next`)
            get their direction from it: `direction="forward"` means A happened **before** B.
            Without direction, "after X" and "before X" cannot be separated in the output.

    Returns:
        [{"score", "hops", "nodes": [id...], "relations": [{type, direction}...]}]
        Sorted by score descending
    """
    fwd = forward_dirs or set()

    def _rel(a: str, b: str, rtype: str) -> Dict[str, str]:
        """Describe the relation of the a→b step. Directed edges get forward / backward."""
        item = {"type": rtype}
        if (a, b) in fwd:
            item["direction"] = "forward"    # a is earlier than b
        elif (b, a) in fwd:
            item["direction"] = "backward"   # a is later than b
        return item
    anchors = sorted(
        [(nid, s) for nid, s in (hit_scores or {}).items() if s >= high_floor],
        key=lambda x: x[1], reverse=True,
    )
    if len(anchors) < 2 or not adjacency:
        return []

    anchor_ids = {nid for nid, _ in anchors}
    anchor_score = dict(anchors)

    def _usable(rtype: str) -> bool:
        if not typed_only:
            return True
        return (rtype or "").lower() != "semantic_neighbor"

    paths: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, ...]] = set()

    def _emit(nodes: List[str], relations: List[Dict[str, str]], score: float) -> None:
        key = tuple(sorted((nodes[0], nodes[-1]))) + tuple(
            sorted(r["type"] for r in relations)
        )
        if key in seen:
            return
        seen.add(key)
        paths.append({
            "score": round(score, 6),
            "hops": len(nodes) - 1,
            "nodes": nodes,
            "relations": relations,
        })

    for a_id, a_s in anchors:
        # 1 hop: the two anchors are directly connected
        for nb, rtype, eff_w in adjacency.get(a_id, []):
            if not _usable(rtype):
                continue
            if nb in anchor_ids and nb != a_id:
                _emit(
                    [a_id, nb], [_rel(a_id, nb, rtype)],
                    min(a_s, anchor_score[nb]) * eff_w,
                )

        # 2 hops: reach the other anchor through one intermediate node
        for mid, r1, w1 in adjacency.get(a_id, []):
            if mid in anchor_ids or not _usable(r1):
                continue
            for nb, r2, w2 in adjacency.get(mid, []):
                if nb == a_id or nb not in anchor_ids or not _usable(r2):
                    continue
                _emit(
                    [a_id, mid, nb],
                    [_rel(a_id, mid, r1), _rel(mid, nb, r2)],
                    min(a_s, anchor_score[nb]) * w1 * w2,
                )

    paths.sort(key=lambda p: p["score"], reverse=True)
    return paths[:max_paths]
