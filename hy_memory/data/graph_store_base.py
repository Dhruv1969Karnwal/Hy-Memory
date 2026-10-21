"""
Agent Memory V2 - GraphStore abstract base class

Defines the unified interface of the graph storage layer (Provider pattern) so different graph database backends can be swapped in.

Currently supported:
- Kuzu (embedded)
- Neo4j (client-server)

Graph Schema:
  Node types:
    User    — user root node
    Memory  — memory node (L6_SCHEMA / L7_INTENTION)
                embedding FLOAT[dims]      — content vector V_con (HNSW indexed)
                beh_embedding FLOAT[dims]  — behaviour vector V_beh (HNSW indexed, may be NULL)
    Topic   — topic tag (LLM tags, not body-text NER entities)
    Entity  — body-text NER entities (GPE/ORG/PRODUCT...; PERSON excluded)
    VdbRef  — VDB shadow node (evidence reference)

  Edge types:
    HAS_MEMORY          : User → Memory
    TAGGED_WITH         : Memory → Topic
    MENTIONS            : Memory → Entity
    RELATED_TO          : Memory ↔ Memory (bidirectional)
    DERIVED_FROM        : Memory → VdbRef (evidence relation)
    CROSS_ABSTRACTS_TO  : Memory → Memory (directed, L6 basic → L6 core)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Sequence, Tuple

from ..models.memory import MemoryNode, MemoryLayer, MemoryStatus
from ..config import MemoryConfig


async def embed_text_for_graph(embed_service, text: str) -> Optional[List[float]]:
    """Embedding call for the graph layer, tolerating both `embed_queued` / `embed` method names.

    Historically each backend used its own: Kuzu called `embed_queued`, Neo4j called `embed`. The real
    `EmbedService` has both, so both work; but any implementation providing only one of them (test fakes,
    slimmed-down embed services) makes tag normalization on the other backend **silently degrade to exact string matching** --
    and without normalization the tag graph shatters into a pile of synonym islands. This unified entry point removes that hazard.
    """
    if embed_service is None or not text:
        return None
    for name in ("embed_queued", "embed"):
        fn = getattr(embed_service, name, None)
        if fn is None:
            continue
        return await fn(text)
    raise AttributeError(
        "embed_service must provide either embed_queued() or embed()"
    )


def make_entity_id(isolation_key: str, label: str, name: str) -> str:
    """Deterministic Entity primary key: isolation + label + casefold(name)."""
    import uuid as _uuid
    norm = (name or "").strip().casefold()
    lab = (label or "").strip().upper()
    return (
        f"ent_{_uuid.uuid5(_uuid.NAMESPACE_URL, f'{isolation_key}:{lab}:{norm}').hex[:12]}"
    )


def normalize_entity_name(name: str) -> str:
    return (name or "").strip().casefold()


class GraphStoreBase(ABC):
    """
    Graph store abstract base class

    Every graph database backend must implement this interface.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config

    _supports_batch_topic_tags: bool = False
    _supports_batch_memory_nodes: bool = False
    _supports_batch_entity_mentions: bool = False
    _supports_batch_related_edges: bool = False
    _supports_batch_temporal_edges: bool = False

    # ================================================================
    # Lifecycle
    # ================================================================

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the graph database connection and Schema"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the connection"""
        ...

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        ...

    # ================================================================
    # User node management
    # ================================================================

    @abstractmethod
    async def ensure_user_node(
        self,
        isolation_key: str,
        user_id: str = "",
        agent_id: str = "",
    ) -> None:
        """Ensure the User node exists"""
        ...

    # ================================================================
    # Memory node CRUD
    # ================================================================

    @abstractmethod
    async def upsert_memory_node(self, node: MemoryNode) -> str:
        """Create or update a Memory node plus the User→Memory edge and return node_id"""
        ...

    async def upsert_memory_nodes_batch(
        self, nodes: Sequence[MemoryNode],
    ) -> List[str]:
        """Batch-compatible Memory projection fallback.

        Backends with a native bulk primitive can override this method and set
        ``_supports_batch_memory_nodes``.  The default deliberately preserves
        the old per-node failure boundary for other and third-party backends.
        """
        out: List[str] = []
        for node in nodes:
            node_id = await self.upsert_memory_node(node)
            if node_id:
                out.append(node_id)
        return out

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Fetch a single Memory node"""
        ...

    @abstractmethod
    async def get_nodes_by_ids(self, node_ids: List[str]) -> List[MemoryNode]:
        """Fetch Memory nodes in batch"""
        ...

    @abstractmethod
    async def get_all_nodes(
        self,
        isolation_key: str,
        layer: Optional[MemoryLayer] = None,
        status: Optional[MemoryStatus] = None,
        limit: int = 100,
    ) -> List[MemoryNode]:
        """Fetch the node list matching the given filters"""
        ...

    async def get_nodes_by_user(
        self,
        user_id: str,
        layer: Optional[MemoryLayer] = None,
        status: Optional[MemoryStatus] = None,
        limit: int = 100,
    ) -> List[MemoryNode]:
        """Return nodes across every historical agent scope for one user.

        User-level profile artifacts use this to discover and retire legacy
        agent-scoped catalog/supplement nodes. Third-party backends may keep the
        safe empty fallback until they implement a native user_id query.
        """
        return []

    @abstractmethod
    async def get_profile(self, isolation_key: str) -> Optional[MemoryNode]:
        """Fetch the L5 Identity Profile node"""
        ...

    @abstractmethod
    async def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update node properties"""
        ...

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Delete a Memory node and its incident edges"""
        ...

    @abstractmethod
    async def delete_all_nodes(self, isolation_key: str) -> int:
        """Delete all Memory nodes under an isolation key and return the deleted count"""
        ...

    async def clear_projection(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        *,
        layer: str = "l2_fact",
    ) -> Dict[str, int]:
        """Drop one user's **pro projection** (by default only `l2_fact`), for use before a rebuild.

        Why it is needed: the graph is a rebuildable derivation, but a rebuild **does not delete old nodes**. After changing the projection rules
        (e.g. narrowing the Topic isolation key from `user::agent::session` to `user::agent`) the old
        session-granularity Topics keep their `TAGGED_WITH` edges in the store, so `get_all_topics` returns
        both generations at once and the results degrade.

        ⚠️ It must be narrowed by layer: a user's Memory nodes also include ones written by System 2 / ultra,
        `l6_schema`, and deleting unconditionally would destroy that data too.

        Also clears the Topic / Entity nodes that become orphans this way (with no Memory attached at all).

        Returns 0 by default (backends without this capability do nothing).
        """
        return {"memories": 0, "topics": 0, "entities": 0}

    @abstractmethod
    async def delete_by_metadata(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Delete Memory nodes by a combination of metadata fields.

        Granularity:
        - user_id only         → delete every memory of this user
        - user_id + agent_id   → delete all memories of this user under the given agent
        - user_id + agent_id + session_id → delete the memories of that exact session

        Returns:
            Number of deleted nodes
        """
        ...

    # ================================================================
    # Edge operations
    # ================================================================

    @abstractmethod
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a Memory→Memory relation edge (RELATED_TO)"""
        ...

    async def add_edges_batch(
        self,
        edges: Sequence[
            Tuple[str, str, str, Optional[Dict[str, Any]]]
        ],
    ) -> List[bool]:
        """Batch-compatible fallback for Memory→Memory edges."""
        return [
            bool(await self.add_edge(source_id, target_id, edge_type, properties))
            for source_id, target_id, edge_type, properties in edges
        ]

    @abstractmethod
    async def add_topic_tag(
        self,
        memory_node_id: str,
        topic_name: str,
        isolation_key: str,
    ) -> str:
        """Attach a topic to a memory node (Memory→Topic) and return topic_id"""
        ...

    async def add_topic_tags_batch(
        self,
        assignments: List[Tuple[str, str, str]],
        *,
        embed_service=None,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[str, str]]:
        """Batch-compatible fallback for Memory→Topic projection.

        Backends can override this to share topic lookup/embedding and reduce
        storage round trips.  The fallback preserves the existing semantics.
        """
        out: List[Tuple[str, str]] = []
        for memory_node_id, topic_name, isolation_key in assignments:
            topic_id = await self.add_topic_tag(
                memory_node_id,
                topic_name,
                isolation_key,
                embed_service=embed_service,
                similarity_threshold=similarity_threshold,
            )
            if topic_id:
                out.append((memory_node_id, topic_id))
        return out

    async def add_topic_path(
        self,
        memory_node_id: str,
        topic_path: List[str],
        isolation_key: str,
        subject: str,
        *,
        node_types: Optional[List[str]] = None,
        role: str = "subject",
        embed_service=None,
    ) -> List[str]:
        """Attach a memory to a typed, subject-scoped structural path.

        Implementations create explicit hierarchy nodes and `SUBTOPIC_OF` edges,
        then attach the memory only to the most-specific leaf.  The default no-op
        keeps third-party GraphStore implementations backward compatible.
        """
        return []

    async def clear_hierarchy_memberships(self, memory_node_id: str) -> int:
        """Remove direct `MEMBER_OF` edges before re-projecting an updated L2."""
        return 0

    async def prune_hierarchy_topics(self, isolation_key: str) -> int:
        """Delete empty hierarchy leaves, then empty ancestors, to convergence."""
        return 0

    async def get_hierarchy_topics(
        self, isolation_key: str, *, subject: str = "",
    ) -> List[Dict[str, Any]]:
        """Return hierarchy nodes with path/depth/member-count metadata."""
        return []

    async def get_hierarchy_topic_memories(
        self, hierarchy_topic_id: str, *, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return active canonical L2 members directly assigned to one node."""
        return []

    async def get_hierarchy_memberships(
        self, isolation_key: str,
    ) -> List[Dict[str, str]]:
        """Return direct leaf memberships for one cross-session topic scope."""
        return []

    # ================================================================
    # Entity (body-text NER) + MENTIONS
    # ================================================================

    async def add_entity_mention(
        self,
        memory_node_id: str,
        name: str,
        label: str,
        isolation_key: str,
    ) -> str:
        """Memory -[:MENTIONS]-> Entity. Returns entity_id, or an empty string on failure.

        Entity is not Topic: Topics come from LLM tags; Entities come from body-text spaCy NER.
        No-op by default (a backend without it does not affect the main path).
        """
        return ""

    async def replace_entity_mentions_batch(
        self,
        assignments: Sequence[
            Tuple[str, Sequence[Tuple[str, str]], str]
        ],
    ) -> List[Tuple[str, str]]:
        """Replace MENTIONS edges for several memories.

        Each item is ``(memory_node_id, [(name, label), ...], isolation_key)``.
        Empty entity lists still clear stale mentions, matching update-time
        projection semantics.  The default is the original clear-then-add
        sequence so non-batch backends retain their existing behavior.
        """
        out: List[Tuple[str, str]] = []
        for memory_node_id, entities, isolation_key in assignments:
            await self.clear_entity_mentions(memory_node_id)
            for name, label in entities:
                entity_id = await self.add_entity_mention(
                    memory_node_id, name, label, isolation_key,
                )
                if entity_id:
                    out.append((memory_node_id, entity_id))
        return out

    async def clear_entity_mentions(self, memory_node_id: str) -> int:
        """Delete all MENTIONS edges of a Memory (rebuilt before UPDATE projection). Returns the number of deleted edges."""
        return 0

    async def find_memories_by_entities(
        self,
        isolation_key: str,
        entities: Sequence[Tuple[str, str]],
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Look up active Memories carrying MENTIONS by entity name (casefold).

        entities: [(name, label), ...]; label is only a same-group filter hint (may be empty).
        Returns [{node_id, entity_name, entity_label, match_score}] with match_score in (0,1].
        """
        return []

    # ================================================================
    # Topic retrieval (tag convergence + tag bridge recall)
    # ================================================================

    async def get_all_topics(
        self,
        isolation_key: str,
    ) -> List[Dict[str, Any]]:
        """Fetch all Topic nodes under an isolation_key

        Returns [{topic_id, name, embedding, degree}]:
        - `embedding` may be None (legacy data / no embed_service passed at write time)
        - `degree` = number of active memories attached to this Topic, used for profile aggregation
          (Topic degree + freshness ranking ≈ "what kind of person the user is", at zero LLM cost)

        Both backends must return the same keys, otherwise anything relying on `degree` silently fails on one backend.
        """
        return []  # Default no-op, overridden by subclasses

    async def tag_bridge_search(
        self,
        topic_ids: List[str],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Tag bridge recall: reverse lookup of the Memory nodes attached to Topic nodes

        i.e. MATCH (t:Topic)<-[:TAGGED_WITH]-(m:Memory) WHERE m.status = 'active'
        Returns [{node_id, content, layer, confidence, topic_name, custom_json}]
        """
        return []  # Default no-op, overridden by subclasses

    # ================================================================
    # V3: VdbRef (shadow node) + DERIVED_FROM edge
    # ================================================================

    async def ensure_vdbref(self, node_id: str, layer: str) -> None:
        """Ensure the VdbRef shadow node exists (created when missing, skipped when present)"""
        ...  # Default no-op, overridden by subclasses

    async def add_derived_from(self, memory_node_id: str, vdbref_node_id: str) -> bool:
        """Add a DERIVED_FROM edge: Memory → VdbRef (evidence relation)"""
        return False  # Default no-op

    async def find_referencing_memories(
        self,
        vdb_node_ids: List[str],
        limit: int = 50,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Reverse lookup: given a list of VDB node_ids, find the Graph Memory nodes referencing them

        i.e. MATCH (v:VdbRef)<-[:DERIVED_FROM]-(m:Memory)
        Returns [{node_id, content, layer, confidence, evidence_vdb_id}]
        """
        return []  # Default no-op


    async def get_evidence_vdbrefs(self, memory_node_id: str) -> List[Dict[str, Any]]:
        """Fetch every DERIVED_FROM evidence VdbRef of one Graph Memory node

        Returns [{node_id, layer}]
        """
        return []  # Default no-op

    # ================================================================
    # Graph retrieval
    # ================================================================

    @abstractmethod
    async def expand_from_anchors(
        self,
        anchor_ids: List[str],
        hop: int = 1,
        max_nodes: int = 50,
    ) -> List[Dict[str, Any]]:
        """Expand N hops from the anchor nodes along graph edges"""
        ...

    async def get_topic_memories(
        self, topic_id: str, *, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List the active memories under one Topic, ordered ascending by memory_at.

        Used by Topic-scoped summaries and the temporal spine.
        Deliberately does not reuse `tag_bridge_search` -- that one is a recall channel of the legacy reader,
        and shared tags were decided not to be traversal edges (τ=0, see docs/pro_graph_design.md §3.0).

        ⚠️ **The graph holds no event time**: the Kuzu `Memory` table only has `created_at` (persist time),
        not `memory_at`, and new columns do not take effect on an existing DB. The VDB is therefore always authoritative for time --
        callers take the node_ids back to the VDB to read `memory_at` and sort there
        (see `pipelines/_graph/projector.resolve_memory_times`).

        Returns:
            [{"node_id", "content"}, ...], ordered ascending by created_at
        """
        return []  # Default no-op, overridden by subclasses

    async def add_temporal_edge(
        self, prev_id: str, next_id: str, topic_id: str = "",
    ) -> bool:
        """Create one temporal spine edge (TEMPORAL_NEXT, **directed**: prev → next).

        An empty `topic_id` means a **global spine** (topic-agnostic, linking the two adjacent memories by event time);
        a non-empty one means a within-Topic spine. Both coexist: the Topic spine expresses "how this topic evolved",
        while the global spine expresses "what happened around those days".

        ⚠️ Temporal relations deliberately **do not form a same-day clique**: 20 memories in one day would be 190 edges and
        a degree of 19; one hop would pull in the whole day and two hops three days -- the same flooding as a tag hub.
        In the global spine each node links only to its nearest predecessor in time, so the degree stays at 2; same-day memories remain reachable while the candidate volume is bounded by the hop budget.

        Difference from `supersedes`: `supersedes` means contradictory replacement (the old one no longer holds),
        while `TEMPORAL_NEXT` means progression (the old one still holds, it is only earlier).
        """
        return False  # Default no-op, overridden by subclasses

    async def add_temporal_edges_batch(
        self, edges: Sequence[Tuple[str, str, str]],
    ) -> List[bool]:
        """Batch-compatible fallback for directed TEMPORAL_NEXT edges."""
        return [
            bool(await self.add_temporal_edge(prev_id, next_id, topic_id))
            for prev_id, next_id, topic_id in edges
        ]

    async def get_related_edges(
        self,
        node_ids: List[str],
        *,
        max_edges: int = 2000,
        relation_types: Optional[List[str]] = None,
        include_temporal: bool = True,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return every RELATED_TO edge with one endpoint in node_ids (undirected, each pair returned once after deduplication).

        This is the **only graph query primitive** of read-side graph reranking; one query serves three uses:
          - Use A (in-pool support propagation): keep the edges whose both endpoints are in the recall pool
          - Use B (threshold relaxation): take the edges whose other endpoint is outside the pool as candidates
          - Use C (relation paths): build adjacency from the same edges on the Python side and find paths of ≤2 hops

        The result carries the target status / layer to save a second lookup: the read side must be able to filter
        SHADOW (deduplication-deleted) nodes, otherwise deleted memories would circle back through the graph.

        With `include_temporal=True`, TEMPORAL_NEXT edges are returned too,
        with `relation_type="temporal_next"` + `directed=True` (direction source→target,
        i.e. "earlier → later"). A/B treat them as ordinary edges (weighted by the τ prior), C uses their direction.

        Returns:
            [{"source_id", "target_id", "relation_type", "weight",
              "target_status", "target_layer", "directed"}, ...]
        """
        return []  # Default no-op, overridden by subclasses

    async def expand_with_tags(
        self,
        anchor_ids: List[str],
        hop: int = 2,
        max_nodes: int = 500,
        isolation_key: str = "",
    ) -> List[Dict[str, Any]]:
        """Expand N hops from the anchors along RELATED_TO + TAGGED_WITH (both directions).

        Paths include:
        - Memory -[RELATED_TO]- Memory (both directions)
        - Memory -[TAGGED_WITH]-> Topic <-[TAGGED_WITH]- Memory (bridged through a Topic)

        When isolation_key is non-empty, only nodes with the same isolation_key are expanded.

        Returns [{node_id, content, layer, confidence, hop, source}]
        """
        return []  # Default no-op, overridden by subclasses

    # ================================================================
    # Vector search (Graph HNSW)
    # ================================================================

    async def vector_search(
        self,
        query_embedding: List[float],
        isolation_key: str,
        layers: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        user_id: str = "",
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """V_con vector search over Graph Memory nodes

        Uses the HNSW index on the embedding column for ANN retrieval.
        
        When isolation_key is empty but user_id is not, a cross-agent search runs
        (matching every node whose isolation_key starts with user_id + "::").
        
        Returns [{node_id, content, layer, confidence, score}]
        """
        return []  # Default no-op, overridden by subclasses

    async def beh_vector_search(
        self,
        query_beh_embedding: List[float],
        isolation_key: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """V_beh vector search over L6 basic nodes (used by the sweeper)

        Uses the HNSW index on the beh_embedding column for ANN retrieval.
        Returns [{node_id, content, embedding, beh_distance}]
        """
        return []  # Default no-op, overridden by subclasses

    async def update_embedding(
        self,
        node_id: str,
        embedding: Optional[List[float]] = None,
        beh_embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update the node embedding / beh_embedding vector properties"""
        return False  # Default no-op

    # ================================================================
    # Cross-domain schema induction
    # ================================================================

    async def add_cross_abstracts_to(
        self,
        basic_id: str,
        core_id: str,
    ) -> bool:
        """Add a CROSS_ABSTRACTS_TO edge: L6 basic → L6 core (directed)"""
        return False  # Default no-op

    async def find_cores_from_basics(
        self,
        basic_ids: List[str],
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """Find L6 core nodes starting from L6 basic nodes along RELATED_TO / CROSS_ABSTRACTS_TO

        Returns [{node_id, content, confidence}], already deduplicated
        """
        return []  # Default no-op

    async def get_cross_abstracts_targets(
        self,
        basic_id: str,
    ) -> List[str]:
        """List the core node_ids that one L6 basic already points to via CROSS_ABSTRACTS_TO"""
        return []  # Default no-op
