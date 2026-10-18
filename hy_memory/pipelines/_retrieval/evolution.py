"""
Evolution chain traversal.

The original logic was extracted out of `reader.py` and is shared by the legacy and hybrid readers.

Responsibilities:
  For a batch of search hits, identify the nodes that sit on an evolution chain, trace the full chain in both
  directions, return the chain head (is_latest=True) as representative, and merge the hits belonging to one chain.

Return format:
  Every evolved hit carries an `evolution_chain` field (list, ordered latest→oldest),
  whose elements are {node_id, content, memory_at, gmt_created, speculate, layer}.
  Time fields are normalized to Unix timestamps (float).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set
import asyncio
import logging

from ...models.memory import MemoryNode

logger = logging.getLogger(__name__)


def _fmt_time(node: MemoryNode) -> Optional[float]:
    """Unix timestamp of node.memory_at; None when missing."""
    t = node.memory_at
    if not t:
        return None
    try:
        return int(t.timestamp())
    except Exception:
        return None


def _time_key(node: MemoryNode):
    """Tiebreaker for ordering evolution-chain internals: memory_at first, falling back to gmt_created when missing."""
    return node.memory_at or node.gmt_created or datetime.min


def _node_to_chain_item(node: MemoryNode) -> Dict[str, Any]:
    """Convert a MemoryNode into a chain entry (a serialization-friendly dict)."""
    return {
        "node_id": node.node_id,
        "content": node.content,
        "memory_at": _fmt_time(node),
        "gmt_created": int(node.gmt_created.timestamp()) if node.gmt_created else None,
        "speculate": getattr(node, "speculate", None),
        "layer": node.layer.value if node.layer else "",
    }


def _linked_ids(node: MemoryNode) -> List[str]:
    return list(dict.fromkeys(
        list(getattr(node, "supersedes", None) or [])
        + list(getattr(node, "superseded_by", None) or [])
    ))


def _order_chain(nodes: Sequence[MemoryNode]) -> List[MemoryNode]:
    """Use the historical head/timestamp ordering for one connected chain."""
    if not nodes:
        return []
    if len(nodes) == 1:
        return [nodes[0]]
    heads = [node for node in nodes if node.is_latest]
    head = max(heads or list(nodes), key=_time_key)
    rest = sorted(
        [node for node in nodes if node.node_id != head.node_id],
        key=_time_key,
        reverse=True,
    )
    return [head] + rest


async def trace_full_chains(
    vector_store,
    start_nodes: Sequence[MemoryNode],
) -> Dict[str, List[MemoryNode]]:
    """Expand many evolution chains with one batched fetch per graph depth.

    The scalar implementation performs a separate traversal for every search
    hit.  When several hits belong to the same chain this repeats identical
    ``get_by_ids`` calls.  This routine resolves the union frontier, then maps
    every start node back to its connected component.  Ordering is identical
    to ``_trace_full_chain``.
    """
    starts = [node for node in start_nodes if node is not None]
    if not starts:
        return {}

    known: Dict[str, MemoryNode] = {node.node_id: node for node in starts}
    frontier = list(dict.fromkeys(
        linked_id
        for node in starts
        for linked_id in _linked_ids(node)
        if linked_id not in known
    ))
    while frontier:
        try:
            fetched = await vector_store.get_by_ids(frontier)
        except Exception as exc:
            logger.warning("[evolution] batch get_by_ids failed: %s", exc)
            break
        next_frontier: List[str] = []
        for node in fetched or []:
            if node is None or node.node_id in known:
                continue
            known[node.node_id] = node
            for linked_id in _linked_ids(node):
                if linked_id not in known:
                    next_frontier.append(linked_id)
        frontier = list(dict.fromkeys(next_frontier))

    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in known}
    for node_id, node in known.items():
        for linked_id in _linked_ids(node):
            if linked_id in known:
                adjacency[node_id].add(linked_id)
                adjacency[linked_id].add(node_id)

    result: Dict[str, List[MemoryNode]] = {}
    component_cache: Dict[str, List[MemoryNode]] = {}
    for start in starts:
        cached = component_cache.get(start.node_id)
        if cached is not None:
            result[start.node_id] = cached
            continue
        stack = [start.node_id]
        component_ids: Set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in component_ids:
                continue
            component_ids.add(node_id)
            stack.extend(adjacency.get(node_id, ()))
        ordered = _order_chain([known[node_id] for node_id in component_ids])
        for node_id in component_ids:
            component_cache[node_id] = ordered
        result[start.node_id] = ordered
    return result


async def _trace_full_chain(
    vector_store,
    start_node: MemoryNode,
) -> List[MemoryNode]:
    """
    Trace an entire evolution chain **in both directions**, starting from any node.

    - Backwards through supersedes to the ancestors
    - Forwards through superseded_by to the successors, locating the real chain head (is_latest=True)

    Returns the whole chain with the head at [0]. Returns [start_node] when it stands alone.
    """
    chains = await trace_full_chains(vector_store, [start_node])
    return chains.get(start_node.node_id, [start_node])


async def _expand_one_chain(
    vector_store, head_item: Dict[str, Any],
    include_sources: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Trace the evolution chain of a single hit (both directions). Returned unchanged when there is no chain or tracing fails.

    Returns:
      - No evolution: the original head_item
      - Evolved: a new dict represented by the chain head (is_latest=True), carrying:
          evolution_chain: [{node_id, content, memory_at, gmt_created, speculate, layer}, ...]
                           ordering: latest → oldest
          is_evolved: True
          chain_node_ids: set — every node_id on the chain (for cross-hit deduplication)
      - **None**: the whole chain is invisible (all SHADOWed by dedup, or all L1_RAW) → the caller
        must drop the hit and must not fall back to the original head_item (that would return SHADOW nodes).
    """
    hit_node: MemoryNode = head_item.get("node")
    if not hit_node:
        return head_item

    # A node with neither supersedes nor superseded_by is not on a chain
    if not hit_node.supersedes and not hit_node.superseded_by:
        return head_item

    chain = await _trace_full_chain(vector_store, hit_node)

    if len(chain) <= 1:
        return head_item

    # Defence: never expose L1_RAW conversation nodes through an evolution chain.
    # search must not return raw-layer content on any path; when raw leaks onto the chain (legacy or abnormal
    # cross-layer supersede linkage), filter it here so it cannot escape through evolution_chain.
    #
    # SHADOW is filtered for the same reason: dedup marks duplicates SHADOW (logical delete) instead of hard-deleting, but
    # _trace_full_chain fetches by node_id through get_by_ids and **ignores status**, and the links are still there --
    # without this filter, content just "deleted" by dedup would surface again through evolution_chain.
    # The reconciler's UPDATE also uses SHADOW for logical deletion, so it is blocked here too.
    from ...models.memory import MemoryLayer, MemoryStatus
    from ...utils.source_filter import source_allowed
    chain_visible = [
        n for n in chain
        if n.layer != MemoryLayer.L1_RAW and n.status != MemoryStatus.SHADOW
        and source_allowed(n.source, include_sources, exclude_sources)
    ]

    # After filtering the whole chain may be invisible (typical case: dedup judged it duplicate and SHADOWed the entire chain,
    # while the hit is a stale node left over on another chain) → this chain must not be returned as a result any more.
    # The check has to be here: _trace_full_chain only looks at is_latest when picking the head, and once the whole chain is
    # SHADOWed is_latest is False everywhere, so it degrades to `max(all_nodes, key=_time_key)` and hands back a
    # SHADOW node as the representative -- which would return logically deleted content straight to the user.
    if not chain_visible:
        return None

    # The representative is the head of the "visible" chain, not the original chain[0]
    head_node = chain_visible[0]

    # Original chain data
    evolution_chain = [_node_to_chain_item(n) for n in chain_visible]
    chain_node_ids = {n.node_id for n in chain}

    return {
        "node_id": head_node.node_id,
        "score": head_item.get("score", 0.0),
        "node": head_node,
        "evolution_chain": evolution_chain,
        "is_evolved": True,
        "chain_node_ids": chain_node_ids,
        **{k: v for k, v in head_item.items()
           if k not in ("node_id", "score", "node",
                         "is_evolved", "evolution_chain", "chain_node_ids")},
    }


async def expand_evolution_chains(
    vector_store,
    hits: List[Dict[str, Any]],
    include_sources: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Concurrently expand the hits that carry evolution chains, and deduplicate them.

    Key behaviour (with only_latest=False several nodes of one chain may be recalled at once):
    1. For every hit on a chain, trace the complete chain in both directions
    2. Multiple hits on the same chain collapse into one result (chain head as representative, highest score kept)
    3. Hits that are not on a chain stay unchanged

    Elements of the input hits look like {"node_id": ..., "score": ..., "node": MemoryNode, ...}.
    The output list can be shorter than the input because of chain deduplication.
    """
    # Find the hits that need expansion (nodes on a chain)
    needs: List[Dict[str, Any]] = []
    idx_of_needs: Dict[int, int] = {}  # hits_idx → needs_idx
    for i, item in enumerate(hits):
        node = item.get("node")
        if node and (getattr(node, "supersedes", None) or getattr(node, "superseded_by", None)):
            idx_of_needs[i] = len(needs)
            needs.append(item)

    if not needs:
        return list(hits)

    expanded = await asyncio.gather(
        *[
            _expand_one_chain(
                vector_store, item,
                include_sources=include_sources,
                exclude_sources=exclude_sources,
            )
            for item in needs
        ],
        return_exceptions=True,
    )

    # Chain dedup: several hits of one chain collapse into a single entry (highest score kept)
    # chain_head_id → best expanded result
    chain_dedup: Dict[str, Dict[str, Any]] = {}
    expanded_by_idx: Dict[int, Dict[str, Any]] = {}  # hits_idx → expanded result
    dropped_idx: set = set()  # hits_idx whose whole chain is invisible and must be dropped outright

    for i, item in enumerate(hits):
        if i not in idx_of_needs:
            continue
        exp = expanded[idx_of_needs[i]]
        if isinstance(exp, Exception):
            logger.warning(f"[evolution] expand failed for {item.get('node_id')}: {exp}")
            expanded_by_idx[i] = item  # Fallback to the original hit
            continue
        if exp is None:
            # The whole chain is invisible (entirely SHADOWed by dedup / all raw) → drop the hit.
            # It must be recorded in dropped_idx explicitly: it is on a chain, and merely leaving it out of expanded_by_idx would
            # make the assembly below hit the else branch and emit it as an ordinary hit --
            # handing the SHADOW node we just filtered back to the user.
            logger.debug(
                f"[evolution] chain fully invisible, dropping hit {item.get('node_id')}"
            )
            dropped_idx.add(i)
            continue

        chain_head_id = exp.get("node_id", "")
        if chain_head_id in chain_dedup:
            # Another hit of the same chain — keep the higher score
            existing = chain_dedup[chain_head_id]
            if exp.get("score", 0.0) > existing.get("score", 0.0):
                chain_dedup[chain_head_id] = exp
        else:
            chain_dedup[chain_head_id] = exp

        expanded_by_idx[i] = exp

    # Collect every node_id covered by the expanded chains (used to filter duplicate hits)
    all_chain_node_ids: set = set()
    for exp in chain_dedup.values():
        chain_ids = exp.get("chain_node_ids", set())
        all_chain_node_ids.update(chain_ids)

    # Assemble the final result: deduplicated and order-preserving
    result: List[Dict[str, Any]] = []
    seen_chain_heads: set = set()
    seen_node_ids: set = set()

    for i, item in enumerate(hits):
        nid = item.get("node_id", "")

        if i in dropped_idx:
            continue

        if i in expanded_by_idx:
            exp = expanded_by_idx[i]
            chain_head_id = exp.get("node_id", "")
            if chain_head_id in seen_chain_heads:
                continue  # Another hit of the same chain, already emitted
            # Use the best entry from chain_dedup (highest score)
            best = chain_dedup.get(chain_head_id, exp)
            seen_chain_heads.add(chain_head_id)
            seen_node_ids.add(chain_head_id)
            result.append(best)
        else:
            # Ordinary hit (not on a chain) -- but skip it when its node_id lies on some chain
            if nid in all_chain_node_ids or nid in seen_node_ids:
                continue
            seen_node_ids.add(nid)
            result.append(item)

    return result
