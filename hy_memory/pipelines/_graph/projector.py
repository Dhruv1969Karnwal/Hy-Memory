"""Write side: project the L2_FACT entries of the VDB into the graph and create the relation edges between memories.

## Why project into `Memory(layer=l2_fact)` instead of extending `VdbRef`

`VdbRef` only has `node_id` + `layer`: it can carry neither tags nor related edges, and Kuzu REL TABLEs are
**strongly typed** (`TAGGED_WITH` is declared as `FROM Memory TO Topic`, `RELATED_TO` as `FROM Memory
TO Memory`), so adding edges to `VdbRef` would mean new rel tables. Projecting into `Memory` needs **zero schema changes** --
which is valuable given that Kuzu has no schema migration capability (new DDL only takes effect on a brand-new DB).

There is no collision with System 2: S2 reads and writes both filter on `layer=l6_schema`.

**Projection does not write embeddings**: the graph side performs no vector retrieval (that is the VDB's job), and Kuzu columns
under an HNSW index cannot be SET after CREATE, so writing one would force a delete-and-recreate on UPDATE.

## The four sources of edges

| Type | Source | Extra cost |
|---|---|---|
| `co_occurs` | facts inside one write batch linked pairwise | 0 (purely local) |
| `semantic_neighbor` | candidates in the reconcile pool scoring >= the threshold and not merged/superseded | 0 (the pool is already in memory) |
| `causes` / `elaborates` / `contrasts` / `same_series` | the `related` field of the reconcile output (LLM annotated) | output tokens only |
| `TAGGED_WITH` → Topic | the tags each fact already carries | 1 embed per new tag |
| `MENTIONS` → Entity | body-text spaCy NER (BOOST whitelist) | 0 embed (exact-name normalization) |

About `semantic_neighbor`: it costs nothing, but all it records is that A and B are semantically similar, which the vector index
already knows -- so its recall gain is small and its value lies mainly in use A (recognizing a coherent cluster).
The real new information comes from `co_occurs` (co-occurrence is orthogonal to similarity) and the LLM-annotated `causes` / `contrasts`
edges (their endpoints usually have very low cosine similarity, so vector retrieval never finds them). See `docs/pro_graph_design.md` §3.4.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Deterministic edges
RELATION_CO_OCCURS = "co_occurs"
RELATION_SEMANTIC_NEIGHBOR = "semantic_neighbor"
RELATION_TEMPORAL_NEXT = "temporal_next"  # Directed, uses its own TEMPORAL_NEXT edge table

# Relation types the LLM may annotate (the vocabulary is deliberately small: weak model + large ontology = garbage)
# `same_series` is deliberately not `same_event`: it links independent dated
# occurrences of one concrete activity/event type while preserving separate L2s.
# The reader may assemble these references into a query-time timeline without
# copying their bodies into another stored memory.
LLM_RELATION_TYPES = ("causes", "elaborates", "contrasts", "same_series")

def _co_occurs_weight(batch_size: int) -> float:
    """Normalize the co_occurs edge weight by batch size: the larger the batch, the less discriminative one co-occurrence is.

    A batch of 2-3 items (the most common case) is almost certainly genuinely related and gets full weight; in a batch of 9 each edge gets only 0.25.
    That beats uniformly lowering the prior -- cross-topic co-occurrence is itself a **signal** ("went to the park / cycling / ate out" written down together,
    their vectors mutually dissimilar yet the relation real), and small batches must not be devalued because large ones carry noise.
    """
    n = max(2, int(batch_size))
    return min(1.0, 2.0 / (n - 1))


def _pairs(items: Sequence[str]) -> List[Tuple[str, str]]:
    """All unordered pairs, without self-loops."""
    out: List[Tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] != items[j]:
                out.append((items[i], items[j]))
    return out


async def project_memories_to_graph(
    *,
    graph_store,
    embed_service,
    nodes: List[Any],
    vector_store=None,
    related_by_node: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    neighbors_by_node: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    edge_min_score: float = 0.55,
    co_occurs_max_batch: int = 8,
    temporal_spine: bool = False,
    temporal_spine_bulk: bool = False,
    hierarchy_topic_write: bool = False,
) -> Dict[str, int]:
    """Project the MemoryNodes of this write into the graph and create the edges.

    Args:
        nodes: MemoryNodes persisted by this write (callers should only pass L2_FACT)
        related_by_node: {node_id: [{"target_id", "relation_type"}, ...]}
            LLM-annotated relations from the reconcile output
        neighbors_by_node: {node_id: [(candidate_node_id, score), ...]}
            from the reconcile candidate pool, producing `semantic_neighbor` edges
        edge_min_score: edge creation threshold for `semantic_neighbor`
        co_occurs_max_batch: skip co_occurs when the batch holds more facts than this (avoids O(n²) blowup)
        vector_store: used by the temporal spine to read the event time memory_at (the graph lacks that field)
        temporal_spine: whether to build the directed within-Topic TEMPORAL_NEXT chain

    Returns:
        statistics dict (nodes / tags / co_occurs / semantic / typed / errors)
    """
    stats = {
        "nodes": 0, "tags": 0, "entities": 0, "co_occurs": 0, "semantic": 0,
        "typed": 0, "temporal": 0, "errors": 0,
        "hierarchy_paths": 0, "hierarchy_nodes": 0,
    }
    if graph_store is None or not nodes:
        return stats

    node_ids: List[str] = []
    topics_by_node: Dict[str, List[str]] = {}
    tag_assignments: List[Tuple[str, str, str]] = []
    entity_assignments: List[Tuple[str, List[Tuple[str, str]], str]] = []
    batch_topic_tags = bool(
        getattr(graph_store, "_supports_batch_topic_tags", False)
        and callable(getattr(graph_store, "add_topic_tags_batch", None))
    )
    hierarchy_scopes = set()

    # Body-text NER -> Entity (parallel to Topic; PERSON and friends are outside the BOOST whitelist)
    try:
        from .._retrieval.ner import extract_entities
    except Exception:
        extract_entities = None  # type: ignore
    _MAX_ENTITIES_PER_MEM = 8

    # -- 1. Node projection + TAGGED_WITH + MENTIONS --
    projectable_nodes = [node for node in nodes if getattr(node, "node_id", "")]
    batch_memory_nodes = bool(
        getattr(graph_store, "_supports_batch_memory_nodes", False)
        and callable(getattr(graph_store, "upsert_memory_nodes_batch", None))
    )
    if batch_memory_nodes:
        try:
            node_ids = list(await graph_store.upsert_memory_nodes_batch(
                projectable_nodes,
            ))
            if len(node_ids) != len(projectable_nodes):
                raise RuntimeError(
                    "batch memory projection returned "
                    f"{len(node_ids)}/{len(projectable_nodes)} node ids"
                )
            stats["nodes"] = len(node_ids)
        except Exception as e:
            logger.debug("[graph] batch memory projection failed; falling back: %s", e)
            node_ids = await _upsert_nodes_individually(
                graph_store, projectable_nodes, stats,
            )
    else:
        node_ids = await _upsert_nodes_individually(
            graph_store, projectable_nodes, stats,
        )
    projected_ids = set(node_ids)

    for node in projectable_nodes:
        nid = getattr(node, "node_id", "")
        if nid not in projected_ids:
            continue

        scope_key = node.get_topic_scope_key()

        # Explicit hierarchy is separate from legacy flat tags.  A memory is
        # attached only to each path's most-specific leaf; parents are reached
        # through SUBTOPIC_OF.  This avoids turning a broad node such as "sport"
        # into an unconditional expansion bucket.
        if hierarchy_topic_write and hasattr(graph_store, "add_topic_path"):
            custom = getattr(node, "custom", None) or {}
            from ...structure import topic_assignments_from_custom
            assignments = topic_assignments_from_custom(custom)
            try:
                if hasattr(graph_store, "clear_hierarchy_memberships"):
                    await graph_store.clear_hierarchy_memberships(nid)
                hierarchy_scopes.add(scope_key)
                for assignment in assignments:
                    path = assignment["path"]
                    subject = assignment["subject"]
                    try:
                        topic_ids = await graph_store.add_topic_path(
                            nid, path, scope_key, subject,
                            node_types=assignment.get("node_types"),
                            role=assignment.get("role") or "participant",
                            embed_service=embed_service,
                        )
                    except TypeError:
                        # Third-party stores may still implement the pre-typed
                        # optional API. Keep the feature backward compatible.
                        topic_ids = await graph_store.add_topic_path(
                            nid, path, scope_key, subject,
                            embed_service=embed_service,
                        )
                    if topic_ids:
                        stats["hierarchy_paths"] += 1
                        stats["hierarchy_nodes"] += len(topic_ids)
            except Exception as e:
                stats["errors"] += 1
                logger.debug(f"[graph] hierarchy projection({nid}) failed: {e}")

        for tag in (getattr(node, "tags", None) or []):
            if not tag:
                continue
            if batch_topic_tags:
                tag_assignments.append((nid, str(tag), scope_key))
                continue
            try:
                # The Topic isolation key **excludes session** (see MemoryNode.get_topic_scope_key):
                # topics are concepts accumulated across sessions. A session-bearing isolation_key would
                # create a separate, mutually unaware Topic per session for the same tag, starving the temporal spine and
                # topic summary。
                _tid = await graph_store.add_topic_tag(
                    nid, str(tag), scope_key,
                    embed_service=embed_service,
                )
                if _tid:
                    topics_by_node.setdefault(nid, []).append(_tid)
                stats["tags"] += 1
            except TypeError:
                # The backend add_topic_tag does not accept embed_service (old signature) -> degrade to no normalization
                try:
                    await graph_store.add_topic_tag(
                        nid, str(tag), scope_key,
                    )
                    stats["tags"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    logger.debug(f"[graph] add_topic_tag({nid}, {tag}) failed: {e}")
            except Exception as e:
                stats["errors"] += 1
                logger.debug(f"[graph] add_topic_tag({nid}, {tag}) failed: {e}")

        # Body-text NER -> Entity + MENTIONS (cleared before rebuilding on UPDATE, so nothing stale remains)
        if extract_entities is not None and hasattr(graph_store, "add_entity_mention"):
            try:
                content = getattr(node, "content", None) or ""
                ents = extract_entities(content, for_boost=True) if content else []
                seen_keys = set()
                clean_entities: List[Tuple[str, str]] = []
                for span, lab in ents[:_MAX_ENTITIES_PER_MEM]:
                    key = (str(span).strip().casefold(), (lab or "").upper())
                    if not key[0] or key in seen_keys:
                        continue
                    seen_keys.add(key)
                    clean_entities.append((str(span).strip(), str(lab)))
                entity_assignments.append((nid, clean_entities, scope_key))
            except Exception as e:
                stats["errors"] += 1
                logger.debug(f"[graph] entity mentions({nid}) failed: {e}")

    if tag_assignments:
        try:
            tagged = await graph_store.add_topic_tags_batch(
                tag_assignments,
                embed_service=embed_service,
            )
            for nid, topic_id in tagged:
                topics_by_node.setdefault(nid, []).append(topic_id)
            stats["tags"] += len(tagged)
            stats["errors"] += max(0, len(tag_assignments) - len(tagged))
        except Exception as e:
            stats["errors"] += len(tag_assignments)
            logger.debug("[graph] batch topic projection failed: %s", e)

    if entity_assignments:
        batch_entity_mentions = bool(
            getattr(graph_store, "_supports_batch_entity_mentions", False)
            and callable(getattr(graph_store, "replace_entity_mentions_batch", None))
        )
        if batch_entity_mentions:
            try:
                mentioned = await graph_store.replace_entity_mentions_batch(
                    entity_assignments,
                )
                stats["entities"] += len(mentioned)
            except Exception as e:
                logger.debug(
                    "[graph] batch entity projection failed; falling back: %s", e,
                )
                await _replace_entity_mentions_individually(
                    graph_store, entity_assignments, stats,
                )
        else:
            await _replace_entity_mentions_individually(
                graph_store, entity_assignments, stats,
            )

    if not node_ids:
        return stats

    if hierarchy_topic_write and hasattr(graph_store, "prune_hierarchy_topics"):
        for scope_key in hierarchy_scopes:
            try:
                await graph_store.prune_hierarchy_topics(scope_key)
            except Exception as e:
                stats["errors"] += 1
                logger.debug(f"[graph] hierarchy prune({scope_key}) failed: {e}")

    # -- 2. co_occurs: link every pair inside the batch --
    # Facts stated together in one turn of conversation carry a plot-level relation, the most useful class for multi-hop traversal
    # ("the restaurant a friend recommended" -- friend and restaurant co-occur in the same batch).
    pending_edges: List[Tuple[str, str, str, float, str]] = []
    if len(node_ids) > 1:
        if len(node_ids) <= co_occurs_max_batch:
            for src, tgt in _pairs(node_ids):
                _w = _co_occurs_weight(len(node_ids))
                pending_edges.append(
                    (src, tgt, RELATION_CO_OCCURS, _w, "co_occurs")
                )
        else:
            logger.debug(
                f"[graph] batch of {len(node_ids)} facts exceeds "
                f"co_occurs_max_batch={co_occurs_max_batch}, skipping co_occurs edges"
            )

    # -- 3. semantic_neighbor: persist the similarity the reconcile candidate pool threw away --
    for nid, neighbors in (neighbors_by_node or {}).items():
        if nid not in node_ids:
            continue
        for cand_id, score in neighbors or []:
            if not cand_id or cand_id == nid or score < edge_min_score:
                continue
            pending_edges.append((
                nid, cand_id, RELATION_SEMANTIC_NEIGHBOR, float(score), "semantic",
            ))

    # -- 4. LLM-annotated typed relations --
    for nid, rels in (related_by_node or {}).items():
        if nid not in node_ids:
            continue
        for rel in rels or []:
            target = (rel or {}).get("target_id")
            rtype = (rel or {}).get("relation_type")
            if not target or target == nid or rtype not in LLM_RELATION_TYPES:
                continue
            pending_edges.append((nid, target, rtype, 1.0, "typed"))

    await _add_edges(graph_store, pending_edges, stats)

    # -- 5. Temporal spine --
    # Incremental writes and full backfills take their predecessors from different places; see the respective docstrings
    if temporal_spine_bulk:
        stats["temporal"] += await _build_temporal_spine_bulk(
            graph_store, nodes, node_ids, topics_by_node, stats,
        )
    elif temporal_spine:
        stats["temporal"] += await _build_temporal_spine(
            graph_store, vector_store, nodes, node_ids, topics_by_node, stats,
        )

    logger.info(
        f"[graph] projected {stats['nodes']} nodes, {stats['tags']} tags, "
        f"entities={stats.get('entities', 0)}, "
        f"hierarchy_paths={stats.get('hierarchy_paths', 0)} "
        f"hierarchy_nodes={stats.get('hierarchy_nodes', 0)} "
        f"edges: co_occurs={stats['co_occurs']} semantic={stats['semantic']} "
        f"typed={stats['typed']} temporal={stats['temporal']} "
        f"errors={stats['errors']}"
    )
    return stats


async def _upsert_nodes_individually(
    graph_store, nodes: List[Any], stats: Dict[str, int],
) -> List[str]:
    """Original per-node projection path, also used after a batch failure."""
    node_ids: List[str] = []
    for node in nodes:
        nid = getattr(node, "node_id", "")
        try:
            # Do not write embeddings: the graph side performs no vector retrieval, and Kuzu indexed columns cannot be SET after CREATE
            await graph_store.upsert_memory_node(node)
            node_ids.append(nid)
            stats["nodes"] += 1
        except Exception as e:
            stats["errors"] += 1
            logger.debug(f"[graph] upsert_memory_node({nid}) failed: {e}")
    return node_ids


async def _replace_entity_mentions_individually(
    graph_store,
    assignments: List[Tuple[str, List[Tuple[str, str]], str]],
    stats: Dict[str, int],
) -> None:
    """Original clear-then-add behavior, preserving its per-memory boundary."""
    for nid, entities, scope_key in assignments:
        try:
            if hasattr(graph_store, "clear_entity_mentions"):
                await graph_store.clear_entity_mentions(nid)
            for name, label in entities:
                eid = await graph_store.add_entity_mention(
                    nid, name, label, scope_key,
                )
                if eid:
                    stats["entities"] += 1
        except Exception as e:
            stats["errors"] += 1
            logger.debug(f"[graph] entity mentions({nid}) failed: {e}")


async def resolve_memory_times(
    vector_store, node_ids: List[str],
) -> Dict[str, float]:
    """Go back to the VDB for the **event time** `memory_at` (unix seconds) of these nodes, falling back to `gmt_created` when absent.

    Why the VDB is required: the `Memory` table in the graph only has `created_at` (persist time) and **no
    `memory_at`**, and new Kuzu columns do not take effect on an existing DB. So the VDB is always authoritative for event time --
    which also matches the way `created_after` filters by event time
    (see `data/vector_store_base.node_effective_time`).
    """
    if vector_store is None or not node_ids:
        return {}
    try:
        nodes = await vector_store.get_by_ids(list(node_ids))
    except Exception as e:
        logger.debug(f"[graph] resolve_memory_times failed: {e}")
        return {}
    out: Dict[str, float] = {}
    for n in (nodes or []):
        if n is None:
            continue
        t = getattr(n, "memory_at", None) or getattr(n, "gmt_created", None)
        if t is None:
            continue
        try:
            out[n.node_id] = t.timestamp()
        except (AttributeError, TypeError, OSError):
            continue
    return out


async def _build_temporal_spine_bulk(
    graph_store,
    nodes: List[Any],
    node_ids: List[str],
    topics_by_node: Dict[str, List[str]],
    stats: Dict[str, int],
) -> int:
    """For full backfills: group the whole batch by Topic, order it by event time, and connect consecutive pairs into a chain.

    The difference from the incremental version (`_build_temporal_spine`) is **where the predecessor comes from**:
    - Incremental write: the batch holds one or two nodes and the predecessor lives in the store, so the graph plus VDB times must be queried,
      and intra-batch relations are expressed by `co_occurs`, which must not be treated as temporal predecessors;
    - Full backfill: the batch is the entire history and its predecessors are inside it, so sorting and linking needs no I/O at all.

    If a backfill used the incremental version, excluding in-batch nodes would leave no predecessor at all and not one edge would be created.
    """
    by_topic: Dict[str, List[Tuple[float, str]]] = {}
    new_ids = set(node_ids)
    for node in nodes:
        nid = getattr(node, "node_id", "")
        if nid not in new_ids:
            continue
        memory_at = getattr(node, "memory_at", None) or getattr(node, "gmt_created", None)
        if memory_at is None:
            continue
        try:
            ts = memory_at.timestamp()
        except (AttributeError, TypeError, OSError):
            continue
        for tid in topics_by_node.get(nid) or []:
            by_topic.setdefault(tid, []).append((ts, nid))

    edges: List[Tuple[str, str, str]] = []
    for tid, items in by_topic.items():
        items.sort()
        for (_t_prev, prev_id), (_t_next, next_id) in zip(items, items[1:]):
            edges.append((prev_id, next_id, tid))
    return await _add_temporal_edges(graph_store, edges, stats)


async def _build_temporal_spine(
    graph_store,
    vector_store,
    nodes: List[Any],
    node_ids: List[str],
    topics_by_node: Dict[str, List[str]],
    stats: Dict[str, int],
    max_topic_members: int = 200,
) -> int:
    """Attach every new memory after the nearest earlier one in event time under the same Topic.

    The implementation is two steps: the graph yields the member ids of the same Topic, the VDB yields those members' `memory_at`
    (the graph has no event time, see `resolve_memory_times`), and the predecessor is selected on the Python side.

    ⚠️ Known limitation: it only links backwards (prev → new) and **never repairs out-of-order insertion**. Memories are written roughly in time order,
    so most chains are correct; but if an older memory is written late into the middle of a chain, it only attaches behind its own predecessor,
    the original prev → next edge is not split apart, and the chain forks.
    Besides walking the spine, the read side still falls back to sorting by memory_at for temporal questions, so this imperfection is acceptable.
    """
    edges: List[Tuple[str, str, str]] = []
    new_ids = set(node_ids)
    for node in nodes:
        nid = getattr(node, "node_id", "")
        if nid not in new_ids:
            continue
        topic_ids = topics_by_node.get(nid) or []
        memory_at = getattr(node, "memory_at", None)
        if memory_at is None:
            continue
        if not topic_ids:
            continue
        try:
            ts = memory_at.timestamp()
            # Candidate members of the same Topic (excluding itself and the other nodes of this batch -- intra-batch relations use co_occurs)
            members: List[str] = []
            for tid in topic_ids:
                for m in await graph_store.get_topic_memories(
                    tid, limit=max_topic_members,
                ):
                    mid = m.get("node_id")
                    if mid and mid != nid and mid not in new_ids:
                        members.append(mid)
            if not members:
                continue

            times = await resolve_memory_times(vector_store, members)
            prev_id, prev_ts = None, None
            for mid, mts in times.items():
                if mts < ts and (prev_ts is None or mts > prev_ts):
                    prev_id, prev_ts = mid, mts
            if prev_id:
                edges.append((prev_id, nid, topic_ids[0]))
        except Exception as e:
            stats["errors"] += 1
            logger.debug(f"[graph] temporal spine for {nid} failed: {e}")
    return await _add_temporal_edges(graph_store, edges, stats)


async def _add_temporal_edges(
    graph_store,
    edges: List[Tuple[str, str, str]],
    stats: Dict[str, int],
) -> int:
    if not edges:
        return 0
    batch_temporal = bool(
        getattr(graph_store, "_supports_batch_temporal_edges", False)
        and callable(getattr(graph_store, "add_temporal_edges_batch", None))
    )
    if batch_temporal:
        try:
            results = await graph_store.add_temporal_edges_batch(edges)
            if len(results) != len(edges):
                raise RuntimeError(
                    f"batch temporal projection returned {len(results)}/{len(edges)} results"
                )
            return sum(bool(ok) for ok in results)
        except Exception as e:
            logger.debug("[graph] batch temporal edges failed; falling back: %s", e)

    count = 0
    for prev_id, next_id, topic_id in edges:
        try:
            if await graph_store.add_temporal_edge(prev_id, next_id, topic_id):
                count += 1
        except Exception as e:
            stats["errors"] += 1
            logger.debug(f"[graph] temporal spine {prev_id}->{next_id} failed: {e}")
    return count


async def _add_edge(
    graph_store, src: str, tgt: str, relation_type: str, weight: float, stats: Dict[str, int],
) -> bool:
    try:
        ok = await graph_store.add_edge(
            src, tgt, "RELATED_TO",
            {"relation_type": relation_type, "weight": float(weight)},
        )
        return bool(ok)
    except Exception as e:
        stats["errors"] += 1
        logger.debug(f"[graph] add_edge({src}->{tgt}, {relation_type}) failed: {e}")
        return False


async def _add_edges(
    graph_store,
    edges: List[Tuple[str, str, str, float, str]],
    stats: Dict[str, int],
) -> None:
    if not edges:
        return
    batch_related = bool(
        getattr(graph_store, "_supports_batch_related_edges", False)
        and callable(getattr(graph_store, "add_edges_batch", None))
    )
    if batch_related:
        payload = [
            (src, tgt, "RELATED_TO", {
                "relation_type": relation_type, "weight": float(weight),
            })
            for src, tgt, relation_type, weight, _counter in edges
        ]
        try:
            results = await graph_store.add_edges_batch(payload)
            if len(results) != len(edges):
                raise RuntimeError(
                    f"batch related projection returned {len(results)}/{len(edges)} results"
                )
            for ok, edge in zip(results, edges):
                if ok:
                    stats[edge[4]] += 1
            return
        except Exception as e:
            logger.debug("[graph] batch related edges failed; falling back: %s", e)

    for src, tgt, relation_type, weight, counter in edges:
        if await _add_edge(graph_store, src, tgt, relation_type, weight, stats):
            stats[counter] += 1


# ================================================================
# Sync contract: VDB-side status changes must be reflected in the graph, otherwise the graph rots
# ================================================================
#
# | VDB action | Graph side |
# |---|---|
# | ADD               | project_memories_to_graph (create nodes + edges) |
# | UPDATE (in place) | project_memories_to_graph (upsert overwrites content/tags) |
# | SUPERSEDE         | new node enters the graph; old node marked superseded, **edges kept** (history remains walkable) |
# | dedup -> SHADOW   | marked shadow and filtered out during read-side expansion |
# | delete / purge    | graph_store.delete_node / delete_by_metadata (already implemented) |
#
# The read side must filter on status, otherwise memories deleted by deduplication would circle back through the graph.

async def sync_graph_on_supersede(graph_store, old_node_ids: List[str]) -> int:
    """SUPERSEDE: mark the old nodes as superseded (edges kept, the history chain stays traversable)."""
    return await _mark_status(graph_store, old_node_ids, "superseded")


async def sync_graph_on_shadow(graph_store, node_ids: List[str]) -> int:
    """Deduplication / logical deletion: mark as shadow, and the read side always filters them out."""
    return await _mark_status(graph_store, node_ids, "shadow")


async def _mark_status(graph_store, node_ids: List[str], status: str) -> int:
    if graph_store is None or not node_ids:
        return 0
    n = 0
    for nid in node_ids:
        if not nid:
            continue
        try:
            if await graph_store.update_node(nid, {"status": status}):
                n += 1
        except Exception as e:
            logger.debug(f"[graph] mark {nid} status={status} failed: {e}")
    return n
