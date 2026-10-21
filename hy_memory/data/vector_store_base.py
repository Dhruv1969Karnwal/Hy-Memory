"""
Agent Memory V2 - VectorStore abstract base class

Defines the unified interface of the vector storage layer (Provider pattern) so the backend can be swapped.

Currently supported:
- Chroma   (embedded, default, zero external dependencies)
- Qdrant   (embedded / remote)
- FAISS    (embedded, CPU)

Stored payload structure:
  - node_id: str
  - isolation_key: str (user_id::agent_id::session_id)
  - user_id / agent_id / session_id: str
  - layer: str
  - content: str                  (core semantics; embedding uses only this field)
  - status: str
  - confidence: float
  - memory_at / temporal_anchor / gmt_created / gmt_modified: str (ISO) | None
  - valid_from / valid_until: str (ISO) | None
  - supersedes: List[str] | None  (IDs of the old nodes this one supersedes)
  - superseded_by: List[str] | None (which newer nodes superseded it)
  - is_latest: bool               (chain tip flag, defaults to True, payload index added)
  - speculate: str | None         (inference annotation, excluded from embedding)
  - source_raw_memory_id: str | None (L3_SUMMARY anchor)
  - emotional_valence / emotional_arousal: float
  - meta_tags: List[str]
  - source_session_id: str
"""

from abc import ABC, abstractmethod
import asyncio
from typing import Optional, List, Dict, Any
import logging
import uuid

from ..models.memory import (
    MemoryNode, MemoryLayer, MemoryStatus,
    MetaCognitionTag,
)
from ..config import MemoryConfig


# ================================================================
# Time filter: event time (memory_at) first, falling back to write time (gmt_created) when missing
# ================================================================
#
# The semantics of `created_after` are **the event described by the memory happened after that time**, so filtering
# must use `memory_at` (event time, caller-supplied), never `gmt_created` (persist time).
#
# The early implementation filtered on gmt_created, which is effectively no filter for bulk imports -- every memory
# from one import shares nearly the same gmt_created, so all of them match however far apart their memory_at values are.
#
# Fallback rule: when `memory_at` is empty, fall back to `gmt_created`. Both writer entry points guarantee
# memory_at has a value, but side writes such as L0_BASIC_INFO (`agent/tools/basic_profile.py`)
# and historical data from before the owner field may leave it empty; the fallback keeps those nodes behaving
# exactly as they did before the change, instead of being silently filtered out because the field changed.

def node_effective_time(payload: Dict[str, Any]) -> Optional[float]:
    """Effective timestamp of a payload: memory_at first, then gmt_created; None when both are missing.

    Chroma compat -- it does not support None, so `_payload_to_chroma_metadata` writes empty values as `""`.
    """
    for key in ("memory_at", "gmt_created"):
        v = payload.get(key)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def passes_created_after(
    payload: Dict[str, Any], created_after: Optional[float]
) -> bool:
    """Post-filter version of the time filter (for Chroma / FAISS, whose filter DSL cannot express null checks).

    Returns True when both time fields are missing -- better to over-recall than to lose data over a missing field.
    """
    if not created_after:
        return True
    t = node_effective_time(payload)
    if t is None:
        return True
    return t >= float(created_after)

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """
    Abstract base class for vector stores

    Every vector database backend must implement this interface.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        base_name = config.vector_store.collection_name or "agent_memories"
        dims = config.vector_store.embedding_dims
        # Append the dimension suffix automatically so different embedding models cannot collide inside one collection
        if dims:
            self._collection_name = f"{base_name}_{dims}"
        else:
            self._collection_name = base_name

    _supports_retrieval_clauses: bool = False
    _supports_narrow_episode_queries: bool = False

    # ================================================================
    # Lifecycle
    # ================================================================

    @abstractmethod
    async def initialize(self) -> None:
        """Open the vector database connection and make sure the collection/index exists"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the connection and release resources"""
        ...

    # ================================================================
    # Writes
    # ================================================================

    @abstractmethod
    async def upsert(self, node: MemoryNode) -> str:
        """
        Insert or update the vector and payload of one MemoryNode.

        Args:
            node: node.embedding must already be set

        Returns:
            node_id (str)
        """
        ...

    @abstractmethod
    async def upsert_batch(self, nodes: List[MemoryNode]) -> List[str]:
        """Bulk write"""
        ...

    @abstractmethod
    async def update_embedding(self, node_id: str, embedding: List[float]) -> bool:
        """Update only the vector (payload unchanged)"""
        ...

    @abstractmethod
    async def update_payload(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update payload fields only (vector unchanged)
        
        Args:
            node_id: node ID
            updates: fields and values to update, e.g. {"status": "shadow"}
            
        Returns:
            whether it succeeded
        """
        ...

    # ================================================================
    # Retrieval
    # ================================================================

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        isolation_key: str = "",
        isolation_keys: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        layers: Optional[List[MemoryLayer]] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        status_filter: Optional[List[MemoryStatus]] = None,
        only_latest: bool = True,
        tags_match_any: Optional[List[str]] = None,
        created_after: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic retrieval

        Supports three filter modes (highest priority first):
        1. Exact keys: isolation_keys non-empty → IN filter
        2. Single key: isolation_key non-empty → exact match (backward compatible)
        3. user level / agent level: user_ids/user_id + optional agent_ids → combined field filter

        Args:
            query_embedding: query vector
            isolation_key: a single isolation key (backward compatible)
            isolation_keys: several isolation keys (exact match mode)
            user_id: a single user ID (backward compatible)
            user_ids: several user IDs (cross-user search)
            agent_ids: list of agent IDs (used together with user_id/user_ids)
            include_sources: call-source whitelist, filtered before the backend top-k
            exclude_sources: call-source blacklist, filtered before the backend top-k; higher priority
            layers: optional, restricts the layers searched
            created_after: optional Unix timestamp (float); only memories with gmt_created >= this value are returned
            limit: number of results to return
            score_threshold: minimum score threshold
            status_filter: optional status filter (defaults to ACTIVE only)
            tags_match_any: optional, only memories whose tags field intersects this list
                            (depends on backend capability; readers may ignore it)

        Returns:
            [{"node_id": ..., "score": ..., "node": MemoryNode}, ...]
        """
        ...

    @abstractmethod
    async def get_by_id(self, node_id: str) -> Optional[MemoryNode]:
        """Fetch by ID"""
        ...

    async def search_batch(
        self, requests: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Execute independent semantic searches as one logical batch.

        Backends with a native batch endpoint should override this method.  The
        fallback deliberately preserves each request's exact arguments and
        ordering, so callers can adopt batching without changing retrieval
        semantics on older or non-Qdrant stores.
        """
        if not requests:
            return []
        return list(await asyncio.gather(
            *(self.search(**dict(request)) for request in requests),
        ))

    async def get_by_ids(self, node_ids: List[str]) -> List[MemoryNode]:
        """
        Bulk fetch by ID (default implementation: sequential get_by_id calls).
        Backends may override this method to use a batch API for efficiency.
        """
        results = []
        for node_id in node_ids:
            node = await self.get_by_id(node_id)
            if node is not None:
                results.append(node)
        return results

    async def list_by_episode(
        self,
        episode_ids: List[str],
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        layers: Optional[List[MemoryLayer]] = None,
        limit: int = 200,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[MemoryNode]:
        """Bound the memories of one discussion by episode_id.

        An episode is a "single discussion" cut inside the SDK by write-time gaps (see `MemoryNode.episode_id`).
        The read side uses it to bound the memories of one matter into candidates, far cheaper than O(n²) temporal edges.

        Default implementation: `list_by_user` plus local filtering. Backends may override it to push the filter down
        (Qdrant / Chroma / FAISS payloads are dynamic, so extra filter conditions need no migration).
        """
        wanted = {e for e in (episode_ids or []) if e}
        if not wanted:
            return []
        nodes = await self.list_by_user(
            user_id=user_id, agent_id=agent_id, layers=layers,
            limit=max(limit * 10, 1000), with_vectors=False,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
        )
        out = [
            n for n in (nodes or [])
            if (getattr(n, "episode_id", "") or "") in wanted
        ]
        return out[:limit]

    async def list_by_session(
        self,
        *,
        user_id: str,
        agent_id: Optional[str],
        session_id: str,
        limit: int = 10000,
        status_filter: Optional[List] = None,
        layers: Optional[List[MemoryLayer]] = None,
        only_latest: bool = False,
        with_vectors: bool = False,
        allow_raw_layer: bool = False,
    ) -> List[MemoryNode]:
        """List one session; remote stores should override with pushed filters."""
        nodes = await self.list_by_user(
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            status_filter=status_filter,
            layers=layers,
            only_latest=only_latest,
            with_vectors=with_vectors,
            allow_raw_layer=allow_raw_layer,
        )
        wanted = session_id or "default_session"
        return [
            node for node in nodes
            if (getattr(node, "session_id", "") or "default_session") == wanted
        ][:limit]

    async def list_by_source_raw_ids(
        self,
        source_raw_ids: List[str],
        *,
        user_id: str,
        agent_id: Optional[str],
        limit: int = 10000,
        status_filter: Optional[List] = None,
        layers: Optional[List[MemoryLayer]] = None,
        only_latest: bool = False,
        with_vectors: bool = False,
    ) -> List[MemoryNode]:
        """List nodes whose direct or accumulated provenance references raw IDs."""
        wanted = {str(value) for value in source_raw_ids if value}
        if not wanted:
            return []
        nodes = await self.list_by_user(
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
            status_filter=status_filter,
            layers=layers,
            only_latest=only_latest,
            with_vectors=with_vectors,
        )
        out = []
        for node in nodes:
            direct = getattr(node, "source_raw_memory_id", "") or ""
            accumulated = (getattr(node, "custom", None) or {}).get(
                "source_raw_memory_ids", []
            )
            if direct in wanted or wanted.intersection(str(v) for v in accumulated or []):
                out.append(node)
        return out[:limit]

    async def get_embeddings(self, node_ids: List[str]) -> Dict[str, List[float]]:
        """
        Bulk fetch node vectors (so dedup can compute pairwise cosine locally without calling the embedding API).

        The default implementation reuses get_by_ids and reads node.embedding; a backend whose get_by_id returns no
        vectors by default (e.g. the qdrant batch client, tencent) must override this to make sure vectors come back.

        Returns:
            {node_id: embedding}; node_ids whose vector is unavailable do not appear in the result.
        """
        out: Dict[str, List[float]] = {}
        nodes = await self.get_by_ids(node_ids)
        for n in nodes:
            emb = getattr(n, "embedding", None)
            if emb:
                out[n.node_id] = list(emb)
        return out

    # ================================================================
    # Deletes
    # ================================================================

    @abstractmethod
    async def delete(self, node_id: str) -> bool:
        """Delete a single vector point"""
        ...

    @abstractmethod
    async def delete_by_isolation_key(self, isolation_key: str) -> int:
        """Delete every vector under one isolation key"""
        ...

    @abstractmethod
    async def delete_by_metadata(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Delete vectors by a combination of metadata fields (same filtering idea as search).

        Granularity:
        - user_id only           → delete every memory of that user
        - user_id + agent_id     → delete every memory of that user under the given agent
        - user_id + agent_id + session_id → delete the memories of that exact session

        Returns:
            number of vectors deleted (-1 when the backend cannot report an exact count)
        """
        ...

    # ================================================================
    # Enumeration
    # ================================================================

    @abstractmethod
    async def list_by_user(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10000,
        status_filter: Optional[List] = None,
        layers: Optional[List[MemoryLayer]] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        only_latest: bool = False,
        with_vectors: bool = True,
        allow_raw_layer: bool = False,
    ) -> List[MemoryNode]:
        """
        Enumerate a user's memory nodes.

        Args:
            user_id: user ID (single value). Mutually exclusive with user_ids; one of the two is required
            agent_id: optional, restricts to one agent (single value)
            limit: maximum number of results
            status_filter: optional, only nodes with the given statuses (e.g. [MemoryStatus.ACTIVE]).
                Omitting it means **no status restriction**, returning every node including SUPERSEDED / SHADOW
            layers: optional, only nodes on the given layers (e.g. [MemoryLayer.L2_FACT])
            user_ids: optional, bulk enumeration over several users; takes precedence over user_id
            agent_ids: optional, bulk enumeration over several agents; takes precedence over agent_id
            only_latest: whether to return only evolution chain tips (is_latest=True).
                **Defaults to False**, i.e. every version on the chain is returned. Full-copy flows such as clone
                depend on that default; before changing it, confirm that no caller needs historical versions
            with_vectors: whether to return embeddings. Pass False when only payload/tags are needed, to avoid pulling large vectors.
            allow_raw_layer: whether L1_RAW raw dialogue rows may be returned. **Defaults to False**:
                external read paths never return raw dialogue. Only internal pipelines (the writer assembling
                multi-turn history) set it to True explicitly when they need it.
        """
        ...

    # ================================================================
    # Statistics
    # ================================================================

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        ...

    @abstractmethod
    async def count(self, isolation_key: str) -> int:
        """Count the vectors under one isolation key"""
        ...

    # ================================================================
    # Tag Index (per-user tag embedding) -- used by reader_hybrid_tag channel B
    # ================================================================
    #
    # Design:
    #   - Separate collection (naming is owned by `_retrieval.config.tag_index_collection_name`),
    #     isolated from the main memories collection
    #   - point_id comes from uuid5(user_id + ":" + tag), which makes upserts idempotent
    #   - payload: {user_id, tag}
    #   - Only the Qdrant backend implements this in the first version; other backends inherit the default (raises NotImplementedError)
    #     reader_hybrid_tag probes the capability at init and silently degrades to hybrid behaviour if unsupported
    #
    # A backend where this is optional may:
    #   - Implement nothing (the default raises NotImplementedError)
    #   - Or override `_supports_tag_index = True` and provide all 5 methods

    _supports_tag_index: bool = False

    async def upsert_tag_embedding(
        self, user_id: str, tag: str, embedding: List[float]
    ) -> None:
        """Write the embedding of (user_id, tag). Idempotent: the same (user_id, tag) always lands on the same point."""
        raise NotImplementedError("tag_index not supported by this backend")

    async def has_tag_embedding(self, user_id: str, tag: str) -> bool:
        """Whether (user_id, tag) is already present in tag_index."""
        raise NotImplementedError("tag_index not supported by this backend")

    async def delete_tag_embedding(self, user_id: str, tag: str) -> None:
        """Delete the embedding of (user_id, tag). A missing entry is an idempotent no-op."""
        raise NotImplementedError("tag_index not supported by this backend")

    async def search_tag_embeddings(
        self,
        user_id: str,
        query_embedding: List[float],
        topk: int = 5,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Vector search inside the given user's tag_index, returning [{"tag": ..., "score": ...}, ...].
        Sorted by score descending; entries scoring below min_score are filtered out.
        """
        raise NotImplementedError("tag_index not supported by this backend")

    async def replace_retrieval_clauses(
        self, parent: MemoryNode, clauses: List[str], embeddings: List[List[float]],
    ) -> int:
        """Replace hidden search aliases for one visible parent memory."""
        return 0

    async def delete_retrieval_clauses(self, parent_id: str) -> int:
        """Delete all hidden search aliases belonging to one parent memory."""
        return 0

    async def delete_retrieval_clauses_by_metadata(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Delete hidden aliases for the same scope as delete_all()."""
        return 0

    async def search_retrieval_clauses(
        self,
        *,
        query_embedding: List[float],
        user_id: str,
        agent_ids: Optional[List[str]] = None,
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search hidden aliases, returning parent_id/score pairs."""
        return []

    async def count_memories_with_tag(
        self, user_id: str, tag: str, isolation_key: str = ""
    ) -> int:
        """
        Count memories in the main memories collection whose tags field contains the given tag and that belong to
        this user (optionally with isolation_key). Used when deleting a memory to decide whether the tag can stay.
        """
        raise NotImplementedError("tag_index not supported by this backend")

    # ================================================================
    # Entity Store (mirrors mem0's separate {collection}_entities collection)
    #   - entity node payload: {data, entity_type, linked_memory_ids, user_id, agent_id}
    #   - upsert: within one user, entities whose text vectors are >= 0.95 similar are the same entity;
    #     their linked_memory_ids are merged, otherwise a new record is created.
    #   - Default NotImplementedError; currently only chroma implements it (experimental).
    # ================================================================

    async def upsert_entity(
        self,
        *,
        entity_text: str,
        entity_type: str,
        embedding: List[float],
        memory_id: str,
        user_id: str,
        agent_id: str = "",
        merge_threshold: float = 0.95,
    ) -> str:
        """Write one entity into the entity store and link it to memory_id.

        If a similar entity already exists for that user (cosine >= merge_threshold), memory_id is merged
        into its linked_memory_ids; otherwise a new entity record is created. Returns entity_id.
        """
        raise NotImplementedError("entity store not supported by this backend")

    async def search_entities(
        self,
        *,
        query_embedding: List[float],
        user_id: str,
        agent_ids: Optional[List[str]] = None,
        top_k: int = 500,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Vector search inside the given user's entity store.

        Returns [{"entity_id", "data", "entity_type", "linked_memory_ids", "score"}, ...],
        sorted by score descending.
        """
        raise NotImplementedError("entity store not supported by this backend")

    async def list_entities(
        self, *, user_id: str, agent_ids: Optional[List[str]] = None, top_k: int = 10000
    ) -> List[Dict[str, Any]]:
        """List every entity record of the given user (for migration / cleanup / statistics)."""
        raise NotImplementedError("entity store not supported by this backend")

    async def delete_entities_for_memory(
        self, *, memory_id: str, user_id: str, agent_ids: Optional[List[str]] = None
    ) -> int:
        """Remove memory_id from the linked_memory_ids of every entity; an entity left with an empty chain is deleted.

        Returns the number of entities affected. Called when a memory is deleted.
        """
        raise NotImplementedError("entity store not supported by this backend")

    # ================================================================
    # Keyword Search (Full-text / BM25)
    # ================================================================

    # Whether the score returned by keyword_search is already an [0,1] normalized relevance score.
    #   - True: the reader uses it directly, without the normalize_bm25 sigmoid
    #     (tencent sparse IP scores ~0.x, qdrant / future binary hit scores of 1.0, etc.
    #      are already inside [0,1]; a sigmoid calibrated for "raw classic BM25 scores 0~20" would squash them to ~0)
    #   - False (default): raw classic BM25 scores (magnitude 0~20+), which the reader normalizes with a sigmoid
    #     through normalize_bm25(midpoint, steepness)
    _keyword_score_normalized: bool = False

    @property
    def keyword_score_normalized(self) -> bool:
        return self._keyword_score_normalized

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        layers: Optional[List[MemoryLayer]] = None,
        status_filter: Optional[List[MemoryStatus]] = None,
        only_latest: bool = True,
        isolation_key: str = "",
        isolation_keys: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full-text keyword search via payload text index.

        Returns results in the same format as search():
            [{"node_id": ..., "score": ..., "node": MemoryNode}, ...]

        Backend implementations should use their native full-text search
        capabilities. Default implementation returns empty (no-op fallback).
        """
        return []

    async def keyword_search_batch(
        self, requests: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Execute independent keyword searches as one logical batch.

        The default is behaviorally identical to calling ``keyword_search``
        for every request.  Native backends may collapse the transport calls.
        """
        if not requests:
            return []
        return list(await asyncio.gather(
            *(self.keyword_search(**dict(request)) for request in requests),
        ))

    # ================================================================
    # Shared utility methods (used by every backend)
    # ================================================================

    @staticmethod
    def _node_id_to_point_id(node_id: str) -> str:
        """
        Convert a node_id into a deterministic UUID-form point_id.

        A node_id that is already UUID-form is used as is; otherwise a deterministic UUID is generated.
        """
        try:
            uuid.UUID(node_id)
            return node_id
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, node_id))

    @staticmethod
    def _tag_point_id(user_id: str, tag: str) -> str:
        """
        Deterministic point_id dedicated to tag_index.
        The same (user_id, tag) always yields the same UUID, so upserts are naturally idempotent.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tag_index::{user_id}::{tag}"))

    @staticmethod
    def _build_search_text(content: Any, tags: Any = None) -> str:
        """Build the canonical lexical-index text for a memory payload.

        Keep this in one place so an in-place payload UPDATE can rebuild the
        same text that an UPSERT would have produced.  Otherwise ``content``
        and ``tags`` can change while Qdrant's text/sparse indexes continue to
        describe the old memory.
        """
        # search_text: content + tags concatenated, then run through lemmatize_for_bm25 (jieba for Chinese, spaCy for English)
        # before being written. Together with the Qdrant whitespace tokenizer, query and index sides both tokenize with jieba,
        # which fixes the all-zero Chinese keyword search problem.
        # Note: lemmatization only affects search_text (inverted index only); the content field keeps the source text untouched,
        # so the memory.content returned to user searches is unaffected.
        from ..pipelines._retrieval.lemmatize import lemmatize_for_bm25
        if isinstance(tags, str):
            tag_values = [tags]
        else:
            tag_values = list(tags or [])
        tags_text = " ".join(str(tag) for tag in tag_values if tag)
        content_text = str(content or "")
        raw_search_text = (
            f"{content_text} {tags_text}".strip()
            if tags_text else content_text
        )
        return lemmatize_for_bm25(raw_search_text) if raw_search_text else ""

    @staticmethod
    def _node_to_payload(node: MemoryNode) -> Dict[str, Any]:
        """Convert a MemoryNode into the generic payload dict (identical structure for every backend)"""
        search_text = VectorStoreBase._build_search_text(node.content, node.tags)

        return {
            "node_id": node.node_id,
            "isolation_key": node.get_isolation_key(),
            "user_id": node.user_id,
            "agent_id": node.agent_id,
            "session_id": node.session_id,
            "source": node.source,
            # Episode (internal session): the read side bounds candidates of one discussion with it. Payloads are dynamic,
            # so Qdrant / Chroma / FAISS need no migration for the new field.
            "episode_id": getattr(node, "episode_id", "") or "",
            "owner": node.owner,
            "layer": node.layer.value,
            "content": node.content,
            "search_text": search_text,
            "status": node.status.value,
            "confidence": node.confidence,
            "source_type": node.source_type.value,
            "emotional_valence": node.emotional_valence,
            "emotional_arousal": node.emotional_arousal,
            "specificity_score": node.specificity_score,
            "rarity_score": node.rarity_score,
            "longtail_flag": node.longtail_flag,
            "meta_tags": [t.value for t in node.meta_tags],
            "source_session_id": node.source_session_id,
            "memory_at": int(node.memory_at.timestamp()) if node.memory_at else None,
            "temporal_anchor": node.temporal_anchor,
            "gmt_created": int(node.gmt_created.timestamp()) if node.gmt_created else None,
            "gmt_modified": int(node.gmt_modified.timestamp()) if node.gmt_modified else None,
            "valid_from": int(node.valid_from.timestamp()) if node.valid_from else None,
            "valid_until": int(node.valid_until.timestamp()) if node.valid_until else None,
            "access_count": node.access_count,
            "last_accessed_at": int(node.last_accessed_at.timestamp()) if node.last_accessed_at else None,
            # Evolution graph
            "supersedes": node.supersedes,
            "superseded_by": node.superseded_by,
            "is_latest": node.is_latest,
            # Inference annotation (payload-only, excluded from embedding)
            "speculate": node.speculate,
            # Topic tags (payload-only, used by reconcile as a soft hint)
            "tags": list(node.tags or []),
            # NER labels aligned with tags; empty string for topic words
            "tag_ners": list(getattr(node, "tag_ners", None) or []),
            # Summary anchor
            "source_raw_memory_id": node.source_raw_memory_id,
            # Custom extension fields (tools store structured KV here, e.g. basic_info_kv)
            "custom": dict(node.custom or {}),
        }

    @staticmethod
    def _payload_to_node(payload: Dict[str, Any]) -> MemoryNode:
        """Rebuild a MemoryNode from a payload dict"""
        return MemoryNode.from_dict(payload)
