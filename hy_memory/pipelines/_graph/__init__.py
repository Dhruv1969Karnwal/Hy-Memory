"""Memory relation graph (pro mode).

- `projector`: write side -- project L2_FACT into the graph and build RELATED_TO / TAGGED_WITH edges
- `rerank`: read side -- usage A (in-pool support propagation) / B (k-hop threshold relaxation) / C (relation paths)
- `topic_summary`: write side -- split summaries by topic (needs an LLM, allowed on the write side)

Design doc: `docs/pro_graph_design.md`
"""

from .projector import (
    RELATION_CO_OCCURS,
    RELATION_SEMANTIC_NEIGHBOR,
    RELATION_TEMPORAL_NEXT,
    LLM_RELATION_TYPES,
    project_memories_to_graph,
    sync_graph_on_supersede,
    sync_graph_on_shadow,
)
from .rerank import (
    TYPE_PRIOR,
    build_adjacency,
    collect_directed_edges,
    propagate_support,
    apply_temporal_affinity,
    apply_type_quota,
    blend_support,
    edge_group,
    episode_support,
    normalize_by_degree,
    reachable_within,
    relax_admission,
    find_relation_paths,
)

__all__ = [
    "RELATION_CO_OCCURS",
    "RELATION_SEMANTIC_NEIGHBOR",
    "RELATION_TEMPORAL_NEXT",
    "LLM_RELATION_TYPES",
    "project_memories_to_graph",
    "sync_graph_on_supersede",
    "sync_graph_on_shadow",
    "TYPE_PRIOR",
    "build_adjacency",
    "collect_directed_edges",
    "propagate_support",
    "apply_temporal_affinity",
    "apply_type_quota",
    "blend_support",
    "edge_group",
    "episode_support",
    "normalize_by_degree",
    "reachable_within",
    "relax_admission",
    "find_relation_paths",
]
