"""
Agent Memory V2 - VectorStore (Qdrant)

Qdrant-backed vector storage layer. It uses the synchronous QdrantClient plus asyncio.to_thread,
which fully avoids the cross-event-loop problems of AsyncQdrantClient.

Responsibilities:
- Append-only storage for L1 raw conversations
- Semantic retrieval over L2 facts
- Vector matching for schema / intention
- Time-enhanced vector retrieval

Collection strategy:
- Single-collection design: vectors of every layer go into the same collection
- payload filter distinguishes isolation_key / layer / status
- simpler operations; can be split later if needed
"""

import concurrent.futures
import os
import threading
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
import logging

from ..models.memory import (
    MemoryNode, MemoryLayer, MemoryStatus,
)
from ..config import MemoryConfig
from ..utils.async_executor import run_in_executor_guarded
from .vector_store_base import VectorStoreBase
from .vector_store_chroma import _vdb_executor

logger = logging.getLogger(__name__)


def _run_in_vdb_pool(func, *args, **kwargs):
    """Run a synchronous function in the dedicated VDB thread pool"""
    import functools
    call = functools.partial(func, *args, **kwargs) if args or kwargs else func
    return run_in_executor_guarded(_vdb_executor, call)


class _LocalQdrantOwner:
    """Keep one local Qdrant client and its SQLite state on one owner thread.

    Local Qdrant may enable SQLite's ``check_same_thread`` on environments
    compiled with ``THREADSAFE=2``. The public async surface still uses the
    shared VDB pool, but every actual client access is forwarded to this
    instance's single owner thread.
    """

    def __init__(self, factory: Callable[[], Any]):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qdrant-local-owner",
        )
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        try:
            self._client = self._executor.submit(factory).result()
        except BaseException:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    def _submit(self, callback: Callable[[], Any]) -> Any:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Local Qdrant client is closed")
            future = self._executor.submit(callback)
        return future.result()

    def __getattr__(self, name: str) -> Any:
        is_callable, value = self._submit(
            lambda: (
                callable(attribute := getattr(self._client, name)),
                None if callable(attribute) else attribute,
            )
        )
        if not is_callable:
            return value

        def invoke(*args: Any, **kwargs: Any) -> Any:
            return self._submit(
                lambda: getattr(self._client, name)(*args, **kwargs)
            )

        return invoke

    def close(self) -> None:
        def close_owned_client() -> None:
            client, self._client = self._client, None
            if client is not None:
                client.close()

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            future = self._executor.submit(close_owned_client)
        try:
            future.result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)


class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant vector store implementation

    Wraps the synchronous QdrantClient with asyncio.to_thread to stay asynchronous,
    fully bypassing the cross-loop asyncio.Event binding inside httpx/httpcore.
    """

    _supports_retrieval_clauses = True
    _supports_narrow_episode_queries = True

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._client = None  # synchronous QdrantClient
        # Native qdrant BM25 runs on a sparse vector (fastembed Qdrant/bm25) and returns a
        # real raw BM25 score (order 0~20+), which the reader normalizes with the normalize_bm25 sigmoid
        # → so _keyword_score_normalized stays False (same as classic BM25).
        self._keyword_score_normalized = False
        # Whether the collection has a sparse "bm25" vector and fastembed is available → decides whether
        # keyword_search uses real BM25; an old dense-only collection has no sparse field and degrades.
        self._supports_fulltext = False
        self._sparse_name = "bm25"
        self._tag_index_ready = False
        self._tag_index_init_lock = threading.Lock()
        self._retrieval_clause_index_ready = False
        self._retrieval_clause_index_init_lock = threading.Lock()
        self._is_local = False

    async def initialize(self) -> None:
        """Initialize the Qdrant client and make sure the collection exists"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                VectorParams, Distance,
                PayloadSchemaType, TextIndexParams, TokenizerType,
            )
            try:
                from qdrant_client.models import SparseVectorParams
            except Exception:
                SparseVectorParams = None
        except ImportError:
            raise ImportError(
                "qdrant-client is required for Qdrant backend. "
                "Install with: pip install hy-memory[qdrant]"
            )

        from ..pipelines._retrieval import bm25_fastembed
        # Whether a new collection carries sparse BM25: needs both SparseVectorParams and fastembed to be available
        enable_sparse = (SparseVectorParams is not None) and bm25_fastembed.is_available()

        vs_config = self.config.vector_store

        # Prefer the remote connection
        if vs_config.host:
            import os as _os
            # Enlarge the connection pool for high concurrency (the default keepalive=20 is too small)
            _pool_size = int(_os.environ.get("QDRANT_POOL_SIZE", "256"))
            self._client = QdrantClient(
                host=vs_config.host,
                port=vs_config.port or 6333,
                api_key=vs_config.api_key or None,
                timeout=60,
                pool_size=_pool_size,
            )
            logger.info(f"[qdrant] pool_size={_pool_size}")
        else:
            self._is_local = True
            # Compatibility: strip unknown fields from the meta.json written by newer qdrant-client versions
            self._patch_local_meta(vs_config.persist_directory)
            self._client = _LocalQdrantOwner(
                lambda: QdrantClient(path=vs_config.persist_directory)
            )

        # Make sure the collection exists (synchronous call, executed inside to_thread)
        def _ensure_collection():
            collections = self._client.get_collections()
            collection_names = [c.name for c in collections.collections]

            created_with_sparse = False
            if self._collection_name not in collection_names:
                create_kwargs = dict(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=vs_config.embedding_dims,
                        distance=Distance.COSINE,
                    ),
                )
                if enable_sparse:
                    create_kwargs["sparse_vectors_config"] = {
                        self._sparse_name: SparseVectorParams()
                    }
                    created_with_sparse = True
                self._client.create_collection(**create_kwargs)
                # Local Qdrant deliberately uses no payload index; creating them one by one only warns
                # and may block initialization for a long time. Only remote services get indexes; the
                # query Filter semantics of both match, the only difference is the missing index speed-up at local data sizes.
                if not self._is_local:
                    for field_name, field_type in [
                        ("isolation_key", PayloadSchemaType.KEYWORD),
                        ("user_id", PayloadSchemaType.KEYWORD),
                        ("agent_id", PayloadSchemaType.KEYWORD),
                        ("session_id", PayloadSchemaType.KEYWORD),
                        ("source", PayloadSchemaType.KEYWORD),
                        ("source_raw_memory_id", PayloadSchemaType.KEYWORD),
                        ("custom.source_raw_memory_ids", PayloadSchemaType.KEYWORD),
                        ("layer", PayloadSchemaType.KEYWORD),
                        ("status", PayloadSchemaType.KEYWORD),
                        ("is_latest", PayloadSchemaType.BOOL),
                    ]:
                        self._client.create_payload_index(
                            collection_name=self._collection_name,
                            field_name=field_name,
                            field_schema=field_type,
                        )
                # Full-text index on search_text (content + tags) for keyword search
                # Note: the tokenizer is WHITESPACE rather than MULTILINGUAL -- it pairs with the
                # lemmatize_for_bm25 (jieba) pre-tokenized search_text built in _node_to_payload, so
                # query and inverted index share one jieba tokenization and avoid MULTILINGUAL/jieba boundary differences.
                if not self._is_local:
                    try:
                        self._client.create_payload_index(
                            collection_name=self._collection_name,
                            field_name="search_text",
                            field_schema=TextIndexParams(
                                type="text",
                                tokenizer=TokenizerType.WHITESPACE,
                                min_token_len=2,
                                max_token_len=20,
                            ),
                        )
                    except Exception as e:
                        logger.warning(
                            f"[qdrant] Failed to create search_text text index: {e}"
                        )
                logger.info(
                    f"Created Qdrant collection: {self._collection_name} "
                    f"(sparse_bm25={'on' if created_with_sparse else 'off'})"
                )
            else:
                # Backfill indexes on an existing remote collection (idempotent); unsupported on local, so skipped.
                if not self._is_local:
                    self._ensure_text_index()
                    for field_name in (
                        "session_id",
                        "source",
                        "source_raw_memory_id",
                        "custom.source_raw_memory_ids",
                    ):
                        try:
                            self._client.create_payload_index(
                                collection_name=self._collection_name,
                                field_name=field_name,
                                field_schema=PayloadSchemaType.KEYWORD,
                            )
                        except Exception as exc:
                            logger.debug(
                                "[qdrant] ensure episode index %s skipped: %s",
                                field_name, exc,
                            )

            # Probe whether this collection has a sparse "bm25" vector (compatible with old dense-only collections)
            has_sparse = created_with_sparse
            if not has_sparse:
                try:
                    info = self._client.get_collection(self._collection_name)
                    sp = getattr(getattr(info, "config", None), "params", None)
                    sparse_cfg = getattr(sp, "sparse_vectors", None) if sp else None
                    has_sparse = bool(sparse_cfg) and self._sparse_name in sparse_cfg
                except Exception as e:
                    logger.debug(f"[qdrant] sparse probe failed: {e}")
                    has_sparse = False
            return has_sparse

        try:
            has_sparse = await _run_in_vdb_pool(_ensure_collection)
        except BaseException:
            client, self._client = self._client, None
            if client is not None:
                await _run_in_vdb_pool(client.close)
            raise
        self._supports_fulltext = bool(has_sparse and bm25_fastembed.is_available())
        logger.info(
            f"[qdrant] fulltext(BM25 sparse)={'on' if self._supports_fulltext else 'off'}"
        )
        logger.debug(f"VectorStore initialized (Qdrant sync client), collection={self._collection_name}")

    # ================================================================
    # Writes
    # ================================================================

    @staticmethod
    def _dense_of(vec):
        """Extract the dense part from the vector qdrant returns.

        In a collection with sparse vectors, point.vector is a named multi-vector dict
        (dense under the default empty key "", sparse under "bm25"); a dense-only collection returns
        a plain List[float]. Always normalized into List[float] (None when it cannot be obtained).
        """
        if isinstance(vec, dict):
            return vec.get("") or vec.get("dense")
        return vec

    def _build_vector_field(self, payload: Dict[str, Any], embedding: List[float]):
        """
        Build the PointStruct.vector field.

        - fulltext off: returns a plain dense list (backward compatible with old dense-only collections)
        - fulltext on: returns {"": dense, "bm25": SparseVector(...)}, the sparse vector encoded
          from search_text by fastembed BM25; if encoding fails it falls back to plain dense.

        L1_RAW is skipped: the raw conversation layer is consumed by no recall path (the reader recalls
        L0/L2/L3/L4, it drops to SHADOW after extraction, and System2 only takes L2/L4), so encoding it sparse only wastes write and storage cost.
        """
        # Defence: if upstream wrongly passes a named multi-vector dict (e.g. a clone copied a read-path embedding),
        # extract the dense part first instead of wrapping it again into an illegal nested structure.
        embedding = self._dense_of(embedding)
        if not self._supports_fulltext:
            return embedding
        if payload.get("layer") == MemoryLayer.L1_RAW.value:
            return embedding
        from ..pipelines._retrieval import bm25_fastembed
        from qdrant_client.models import SparseVector

        text = payload.get("search_text") or payload.get("content") or ""
        sp = bm25_fastembed.encode_doc(text) if text else None
        if not sp:
            return embedding
        indices, values = sp
        return {"": embedding, self._sparse_name: SparseVector(indices=indices, values=values)}

    async def upsert(self, node: MemoryNode) -> str:
        """Insert or update one MemoryNode's vector and payload"""
        from qdrant_client.models import PointStruct

        if node.embedding is None:
            raise ValueError(f"Node {node.node_id} has no embedding, cannot upsert to vector store")

        payload = self._node_to_payload(node)
        point_id = self._node_id_to_point_id(node.node_id)
        vector_field = self._build_vector_field(payload, node.embedding)

        def _upsert():
            self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector_field,
                        payload=payload,
                    )
                ],
            )

        await _run_in_vdb_pool(_upsert)
        logger.debug(
            f"[vector-store] upsert: id={node.node_id} "
            f"layer={node.layer.value if hasattr(node.layer, 'value') else node.layer} "
            f"isolation_key={payload.get('isolation_key', '')} "
            f"sparse={'yes' if isinstance(vector_field, dict) else 'no'} "
            f"content={node.content[:200]}"
        )
        return node.node_id

    async def upsert_batch(self, nodes: List[MemoryNode]) -> List[str]:
        """Batch write"""
        from qdrant_client.models import PointStruct

        points = []
        ids = []
        for node in nodes:
            if node.embedding is None:
                logger.warning(f"Skipping node {node.node_id}: no embedding")
                continue
            payload = self._node_to_payload(node)
            points.append(PointStruct(
                id=self._node_id_to_point_id(node.node_id),
                vector=self._build_vector_field(payload, node.embedding),
                payload=payload,
            ))
            ids.append(node.node_id)

        if points:
            def _upsert_batch():
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
            await _run_in_vdb_pool(_upsert_batch)
        return ids

    async def update_embedding(self, node_id: str, embedding: List[float]) -> bool:
        """Update only the vector (payload unchanged)"""
        from qdrant_client.models import PointVectors

        point_id = self._node_id_to_point_id(node_id)
        try:
            def _update():
                self._client.update_vectors(
                    collection_name=self._collection_name,
                    points=[PointVectors(id=point_id, vector=embedding)],
                )
            await _run_in_vdb_pool(_update)
            return True
        except Exception as e:
            logger.warning(f"Failed to update embedding for {node_id}: {e}")
            return False

    async def update_payload(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update payload and keep the lexical payload / sparse vector consistent with the content."""
        if not updates:
            return True
        from qdrant_client.models import (
            DeleteVectors,
            DeleteVectorsOperation,
            PointVectors,
            SetPayload,
            SetPayloadOperation,
            SparseVector,
            UpdateVectors,
            UpdateVectorsOperation,
        )
        point_id = self._node_id_to_point_id(node_id)
        updates = dict(updates)
        new_embedding = updates.pop("embedding", None)  # pull the vector out for separate handling
        try:
            lexical_changed = "content" in updates or "tags" in updates
            if lexical_changed:
                current_payload: Dict[str, Any] = {}
                if "content" not in updates or "tags" not in updates:
                    def _retrieve_payload():
                        rows = self._client.retrieve(
                            collection_name=self._collection_name,
                            ids=[point_id],
                            with_payload=True,
                            with_vectors=False,
                        )
                        return dict(rows[0].payload or {}) if rows else {}
                    current_payload = await _run_in_vdb_pool(_retrieve_payload)
                updates["search_text"] = self._build_search_text(
                    updates.get("content", current_payload.get("content", "")),
                    updates.get("tags", current_payload.get("tags", [])),
                )

            sparse_vector = None
            delete_sparse = False
            if lexical_changed and self._supports_fulltext:
                from ..pipelines._retrieval import bm25_fastembed
                encoded = bm25_fastembed.encode_doc(updates["search_text"])
                if encoded:
                    sparse_vector = SparseVector(
                        indices=encoded[0], values=encoded[1],
                    )
                else:
                    # Empty/unencodable replacement text must not leave the old
                    # document vector searchable in the sparse channel.
                    delete_sparse = True

            def _update():
                # Payload + vector changes belong to one logical memory update.
                # Qdrant's batch endpoint applies both operations in one HTTP
                # request, avoiding two serial network round trips per UPDATE.
                vector_update: Any = None
                if self._supports_fulltext:
                    named_vectors: Dict[str, Any] = {}
                    if new_embedding is not None:
                        named_vectors[""] = new_embedding
                    if sparse_vector is not None:
                        named_vectors[self._sparse_name] = sparse_vector
                    vector_update = named_vectors or None
                elif new_embedding is not None:
                    # Legacy collections expose one unnamed dense vector.
                    vector_update = new_embedding

                operations = []
                if updates:
                    operations.append(SetPayloadOperation(set_payload=SetPayload(
                        payload=updates,
                        points=[point_id],
                    )))
                if vector_update:
                    operations.append(UpdateVectorsOperation(update_vectors=UpdateVectors(
                        points=[PointVectors(id=point_id, vector=vector_update)],
                    )))
                if delete_sparse:
                    operations.append(DeleteVectorsOperation(delete_vectors=DeleteVectors(
                        points=[point_id], vector=[self._sparse_name],
                    )))

                if len(operations) > 1 and hasattr(
                    self._client, "batch_update_points"
                ):
                    self._client.batch_update_points(
                        collection_name=self._collection_name,
                        update_operations=operations,
                    )
                    return
                if updates:
                    self._client.set_payload(
                        collection_name=self._collection_name,
                        payload=updates,
                        points=[point_id],
                    )
                if vector_update:
                    self._client.update_vectors(
                        collection_name=self._collection_name,
                        points=[PointVectors(id=point_id, vector=vector_update)],
                    )
                if delete_sparse:
                    self._client.delete_vectors(
                        collection_name=self._collection_name,
                        points=[point_id],
                        vectors=[self._sparse_name],
                    )
            await _run_in_vdb_pool(_update)
            logger.debug(
                f"[vector-store] update_payload: id={node_id} updates={list(updates.keys())} "
                f"embedding={'yes' if new_embedding is not None else 'no'}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to update payload for {node_id}: {e}")
            return False

    async def update_payload_batch(
        self, updates_by_id: List[Tuple[str, Dict[str, Any]]],
    ) -> List[str]:
        """Apply payload/vector updates while rebuilding lexical indexes."""
        if not updates_by_id:
            return []
        from qdrant_client.models import (
            DeleteVectors,
            DeleteVectorsOperation,
            PointVectors,
            SetPayload,
            SetPayloadOperation,
            SparseVector,
            UpdateVectors,
            UpdateVectorsOperation,
        )

        prepared = []
        missing_payload_ids = []
        for node_id, raw_updates in updates_by_id:
            updates = dict(raw_updates or {})
            lexical_changed = "content" in updates or "tags" in updates
            if lexical_changed and ("content" not in updates or "tags" not in updates):
                missing_payload_ids.append(node_id)
            prepared.append((node_id, updates, lexical_changed))

        existing_payloads: Dict[str, Dict[str, Any]] = {}
        if missing_payload_ids:
            point_ids = [self._node_id_to_point_id(nid) for nid in missing_payload_ids]

            def _retrieve_payloads():
                return self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=False,
                )

            rows = await _run_in_vdb_pool(_retrieve_payloads)
            for row in rows or []:
                payload = dict(row.payload or {})
                stored_id = str(payload.get("node_id") or "")
                if stored_id:
                    existing_payloads[stored_id] = payload

        operations = []
        accepted_ids = []
        for node_id, updates, lexical_changed in prepared:
            new_embedding = updates.pop("embedding", None)
            point_id = self._node_id_to_point_id(node_id)
            if lexical_changed:
                current = existing_payloads.get(node_id, {})
                updates["search_text"] = self._build_search_text(
                    updates.get("content", current.get("content", "")),
                    updates.get("tags", current.get("tags", [])),
                )
            if updates:
                operations.append(SetPayloadOperation(set_payload=SetPayload(
                    payload=updates, points=[point_id],
                )))
            vector_update: Any = None
            named_vectors: Dict[str, Any] = {}
            if self._supports_fulltext and new_embedding is not None:
                named_vectors[""] = new_embedding
            elif new_embedding is not None:
                vector_update = new_embedding
            delete_sparse = False
            if lexical_changed and self._supports_fulltext:
                from ..pipelines._retrieval import bm25_fastembed
                encoded = bm25_fastembed.encode_doc(updates["search_text"])
                if encoded:
                    named_vectors[self._sparse_name] = SparseVector(
                        indices=encoded[0], values=encoded[1],
                    )
                else:
                    delete_sparse = True
            if named_vectors:
                vector_update = named_vectors
            if vector_update:
                operations.append(UpdateVectorsOperation(update_vectors=UpdateVectors(
                    points=[PointVectors(id=point_id, vector=vector_update)],
                )))
            if delete_sparse:
                operations.append(DeleteVectorsOperation(delete_vectors=DeleteVectors(
                    points=[point_id], vector=[self._sparse_name],
                )))
            if updates or vector_update or delete_sparse:
                accepted_ids.append(node_id)

        if not operations:
            return []

        def _update_batch():
            self._client.batch_update_points(
                collection_name=self._collection_name,
                update_operations=operations,
            )

        await _run_in_vdb_pool(_update_batch)
        logger.debug(
            "[vector-store] update_payload_batch: nodes=%d operations=%d",
            len(accepted_ids), len(operations),
        )
        return accepted_ids

    # ================================================================
    # Retrieval
    # ================================================================

    @staticmethod
    def _build_search_filter(
        *,
        isolation_key: str = "",
        isolation_keys: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        layers: Optional[List[MemoryLayer]] = None,
        status_filter: Optional[List[MemoryStatus]] = None,
        only_latest: bool = True,
        tags_match_any: Optional[List[str]] = None,
        created_after: Optional[float] = None,
    ):
        """Build the exact payload filter shared by scalar and batch search."""
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsNullCondition,
            MatchAny,
            MatchValue,
            PayloadField,
            Range,
        )

        must_conditions = []
        must_not_conditions = []
        if isolation_keys:
            must_conditions.append(
                FieldCondition(key="isolation_key", match=MatchAny(any=isolation_keys))
            )
        elif isolation_key:
            must_conditions.append(
                FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key))
            )
        else:
            effective_uids = user_ids if user_ids else ([user_id] if user_id else [])
            if effective_uids:
                match = (
                    MatchValue(value=effective_uids[0])
                    if len(effective_uids) == 1
                    else MatchAny(any=effective_uids)
                )
                must_conditions.append(FieldCondition(key="user_id", match=match))
                if agent_ids:
                    must_conditions.append(
                        FieldCondition(key="agent_id", match=MatchAny(any=agent_ids))
                    )

        if layers:
            must_conditions.append(
                FieldCondition(key="layer", match=MatchAny(any=[layer.value for layer in layers]))
            )
        if status_filter:
            must_conditions.append(
                FieldCondition(
                    key="status",
                    match=MatchAny(any=[status.value for status in status_filter]),
                )
            )
        else:
            must_conditions.append(
                FieldCondition(
                    key="status", match=MatchValue(value=MemoryStatus.ACTIVE.value)
                )
            )
        if only_latest:
            must_conditions.append(
                FieldCondition(key="is_latest", match=MatchValue(value=True))
            )
        if include_sources:
            must_conditions.append(
                FieldCondition(key="source", match=MatchAny(any=list(include_sources)))
            )
        if exclude_sources:
            must_not_conditions.append(
                FieldCondition(key="source", match=MatchAny(any=list(exclude_sources)))
            )
        if tags_match_any:
            tag_list = sorted({tag for tag in tags_match_any if tag})
            if tag_list:
                must_conditions.append(
                    FieldCondition(key="tags", match=MatchAny(any=tag_list))
                )
        if created_after:
            must_conditions.append(
                Filter(
                    should=[
                        FieldCondition(key="memory_at", range=Range(gte=created_after)),
                        Filter(
                            must=[
                                IsNullCondition(is_null=PayloadField(key="memory_at")),
                                FieldCondition(
                                    key="gmt_created", range=Range(gte=created_after)
                                ),
                            ]
                        ),
                    ]
                )
            )
        return Filter(must=must_conditions, must_not=must_not_conditions or None)

    def _query_points_to_hits(self, points) -> List[Dict[str, Any]]:
        output = []
        for hit in points or []:
            node = self._payload_to_node(hit.payload)
            output.append({
                "node_id": node.node_id,
                "score": float(getattr(hit, "score", 0.0) or 0.0),
                "node": node,
            })
        return output

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
        """Semantic retrieval. With only_latest=True only is_latest=True nodes are searched (default)."""
        query_filter = self._build_search_filter(
            isolation_key=isolation_key,
            isolation_keys=isolation_keys,
            user_id=user_id,
            user_ids=user_ids,
            agent_ids=agent_ids,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            layers=layers,
            status_filter=status_filter,
            only_latest=only_latest,
            tags_match_any=tags_match_any,
            created_after=created_after,
        )

        def _search():
            # Compatible with qdrant-client >=1.17 (query_points) and <1.17 (search)
            if hasattr(self._client, 'query_points'):
                query_response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                return query_response.points
            else:
                return self._client.search(
                    collection_name=self._collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )

        results = await _run_in_vdb_pool(_search)

        output = self._query_points_to_hits(results)

        logger.debug(
            f"[vector-store] search: isolation_key={isolation_key} "
            f"isolation_keys={isolation_keys} user_id={user_id} agent_ids={agent_ids} "
            f"limit={limit} found={len(output)}"
        )
        for i, item in enumerate(output):
            logger.debug(
                f"[vector-store] hit[{i}] score={item['score']:.4f} "
                f"id={item['node_id']} "
                f"layer={item['node'].layer.value if hasattr(item['node'].layer, 'value') else ''} "
                f"content={item['node'].content[:200]}"
            )

        return output

    async def search_batch(
        self, requests: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Collapse independent remote dense searches into one Qdrant RPC."""
        if not requests:
            return []
        if self._is_local or not hasattr(self._client, "query_batch_points"):
            return await super().search_batch(requests)

        from qdrant_client.models import QueryRequest

        batch = []
        for raw in requests:
            request = dict(raw)
            embedding = request.pop("query_embedding")
            limit = int(request.pop("limit", 10))
            score_threshold = float(request.pop("score_threshold", 0.0))
            query_filter = self._build_search_filter(
                isolation_key=request.pop("isolation_key", ""),
                isolation_keys=request.pop("isolation_keys", None),
                user_id=request.pop("user_id", None),
                user_ids=request.pop("user_ids", None),
                agent_ids=request.pop("agent_ids", None),
                include_sources=request.pop("include_sources", None),
                exclude_sources=request.pop("exclude_sources", None),
                layers=request.pop("layers", None),
                status_filter=request.pop("status_filter", None),
                only_latest=request.pop("only_latest", True),
                tags_match_any=request.pop("tags_match_any", None),
                created_after=request.pop("created_after", None),
            )
            if request:
                raise TypeError(
                    f"Unsupported Qdrant batch search arguments: {sorted(request)}"
                )
            batch.append(QueryRequest(
                query=embedding,
                filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vector=False,
            ))

        def _search_batch():
            return self._client.query_batch_points(
                collection_name=self._collection_name,
                requests=batch,
            )

        try:
            responses = await _run_in_vdb_pool(_search_batch)
            if len(responses) != len(requests):
                raise RuntimeError(
                    f"Qdrant batch response count {len(responses)} != {len(requests)}"
                )
            return [
                self._query_points_to_hits(getattr(response, "points", []))
                for response in responses
            ]
        except Exception as exc:
            logger.warning(
                "[qdrant] dense batch failed; falling back to scalar search: %s", exc,
            )
            return await super().search_batch(requests)

    async def get_by_id(self, node_id: str) -> Optional[MemoryNode]:
        """Fetch by ID"""
        point_id = self._node_id_to_point_id(node_id)
        try:
            def _retrieve():
                return self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=True,
                )
            results = await _run_in_vdb_pool(_retrieve)
            if results:
                node = self._payload_to_node(results[0].payload)
                if results[0].vector:
                    node.embedding = self._dense_of(results[0].vector)
                return node
        except Exception as e:
            logger.warning(f"Failed to get vector point {node_id}: {e}")
        return None

    async def get_by_ids(self, node_ids: List[str]) -> List[MemoryNode]:
        """Fetch many by ID (a single retrieve call)"""
        if not node_ids:
            return []
        point_ids = [self._node_id_to_point_id(nid) for nid in node_ids]
        try:
            def _retrieve():
                return self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=False,
                )
            results = await _run_in_vdb_pool(_retrieve)
            nodes = []
            for point in results:
                node = self._payload_to_node(point.payload)
                nodes.append(node)
            return nodes
        except Exception as e:
            logger.warning(f"Failed to batch get vector points {node_ids}: {e}")
            return []

    async def get_embeddings(self, node_ids: List[str]) -> Dict[str, List[float]]:
        """Fetch vectors in bulk (one retrieve with with_vectors=True) so dedup can compute cosine locally."""
        if not node_ids:
            return {}
        point_ids = [self._node_id_to_point_id(nid) for nid in node_ids]
        try:
            def _retrieve():
                return self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=True,
                )
            results = await _run_in_vdb_pool(_retrieve)
            out: Dict[str, List[float]] = {}
            for point in results:
                node = self._payload_to_node(point.payload)
                vec = self._dense_of(point.vector)
                if node.node_id and vec and not isinstance(vec, dict):
                    out[node.node_id] = list(vec)
            return out
        except Exception as e:
            logger.warning(f"Failed to batch get embeddings {node_ids}: {e}")
            return {}

    # ================================================================
    # Deletion
    # ================================================================

    async def delete(self, node_id: str) -> bool:
        """Delete a single vector point"""
        from qdrant_client.models import PointIdsList

        point_id = self._node_id_to_point_id(node_id)
        try:
            def _delete():
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=PointIdsList(points=[point_id]),
                )
            await _run_in_vdb_pool(_delete)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete vector point {node_id}: {e}")
            return False

    async def delete_by_isolation_key(self, isolation_key: str) -> int:
        """Delete every vector under one isolation key"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        try:
            def _delete():
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key))
                            ]
                        )
                    ),
                )
            await _run_in_vdb_pool(_delete)
            return -1  # Qdrant does not report how many were deleted
        except Exception as e:
            logger.warning(f"Failed to delete vectors for {isolation_key}: {e}")
            return 0

    async def delete_by_metadata(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Delete vectors by a combination of metadata fields"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        try:
            must_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
            if agent_id is not None:
                must_conditions.append(
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
                )
            if session_id is not None:
                must_conditions.append(
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                )

            def _delete():
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(must=must_conditions)
                    ),
                )
            await _run_in_vdb_pool(_delete)
            return -1  # Qdrant does not report how many were deleted
        except Exception as e:
            logger.warning(f"Failed to delete vectors by metadata (user_id={user_id}): {e}")
            return 0

    # ================================================================
    # Enumeration
    # ================================================================

    async def list_by_user(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10000,
        status_filter: Optional[List] = None,
        layers: Optional[List] = None,
        user_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        only_latest: bool = False,
        with_vectors: bool = True,
        allow_raw_layer: bool = False,
    ) -> List[MemoryNode]:
        """Enumerate a user's memory nodes via the scroll API, supporting many users / many agents

        allow_raw_layer: defaults to False, i.e. L1_RAW is not returned (external read path).
            Internal pipelines reading raw conversations must set it True explicitly.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        effective_uids = list(user_ids) if user_ids else ([user_id] if user_id else [])
        if not effective_uids:
            logger.warning("[vector-store] list_by_user called without user_id/user_ids, refused")
            return []
        effective_aids = list(agent_ids) if agent_ids else ([agent_id] if agent_id else [])

        if len(effective_uids) == 1:
            must_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=effective_uids[0]))
            ]
        else:
            must_conditions = [
                FieldCondition(key="user_id", match=MatchAny(any=effective_uids))
            ]
        must_not_conditions: List = []
        if effective_aids:
            if len(effective_aids) == 1:
                must_conditions.append(
                    FieldCondition(key="agent_id", match=MatchValue(value=effective_aids[0]))
                )
            else:
                must_conditions.append(
                    FieldCondition(key="agent_id", match=MatchAny(any=effective_aids))
                )
        if only_latest:
            must_conditions.append(
                FieldCondition(key="is_latest", match=MatchValue(value=True))
            )
        if include_sources:
            must_conditions.append(
                FieldCondition(key="source", match=MatchAny(any=list(include_sources)))
            )
        if exclude_sources:
            must_not_conditions.append(
                FieldCondition(key="source", match=MatchAny(any=list(exclude_sources)))
            )
        if status_filter:
            status_values = [s.value if hasattr(s, 'value') else str(s) for s in status_filter]
            must_conditions.append(
                FieldCondition(key="status", match=MatchAny(any=status_values))
            )
        if layers:
            layer_values = [
                layer.value if hasattr(layer, "value") else str(layer)
                for layer in layers
            ]
            if not allow_raw_layer:
                layer_values = [v for v in layer_values if v != MemoryLayer.L1_RAW.value]
            if layer_values:
                must_conditions.append(
                    FieldCondition(key="layer", match=MatchAny(any=layer_values))
                )
            else:
                # Only raw was named and the filtered layer list is empty: add an exclusion so the result
                # is certainly empty; it must not be omitted (omitting would open all layers, wider than intended)
                must_not_conditions.append(
                    FieldCondition(key="layer", match=MatchValue(value=MemoryLayer.L1_RAW.value))
                )
        elif not allow_raw_layer:
            must_not_conditions.append(
                FieldCondition(key="layer", match=MatchValue(value=MemoryLayer.L1_RAW.value))
            )

        scroll_filter = Filter(
            must=must_conditions,
            must_not=must_not_conditions or None,
        )

        def _scroll_all():
            nodes = []
            offset = None
            if self._is_local:
                try:
                    local_page = max(
                        100,
                        int(os.getenv("MEMORY_QDRANT_LOCAL_SCROLL_BATCH_SIZE", "1000")),
                    )
                except (TypeError, ValueError):
                    local_page = 1000
                batch_size = min(limit, local_page)
            else:
                try:
                    remote_page = max(
                        100,
                        int(os.getenv("MEMORY_QDRANT_REMOTE_SCROLL_BATCH_SIZE", "1000")),
                    )
                except (TypeError, ValueError):
                    remote_page = 1000
                batch_size = min(limit, remote_page)

            while len(nodes) < limit:
                scroll_result = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=scroll_filter,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )

                # Compatible with qdrant-client >= 1.17 (returns a ScrollResult object)
                # and < 1.17 (returns tuple(points, next_offset))
                if isinstance(scroll_result, tuple):
                    points, next_offset = scroll_result
                else:
                    points = scroll_result.points
                    next_offset = scroll_result.next_page_offset

                for point in points:
                    node = self._payload_to_node(point.payload)
                    if with_vectors and point.vector:
                        node.embedding = self._dense_of(point.vector)
                    nodes.append(node)

                if next_offset is None or not points:
                    break
                offset = next_offset

            return nodes[:limit]

        try:
            nodes = await _run_in_vdb_pool(_scroll_all)
        except Exception as e:
            logger.error(f"[vector-store] list_by_user scroll failed: {e}", exc_info=True)
            nodes = []

        logger.info(
            f"[vector-store] list_by_user: user_ids={effective_uids} "
            f"agent_ids={effective_aids} found={len(nodes)}"
        )
        return nodes

    async def _scroll_filtered_nodes(
        self,
        scroll_filter,
        *,
        limit: int,
        with_vectors: bool,
        operation: str,
    ) -> List[MemoryNode]:
        """Run one bounded payload-filtered scroll without a full user scan."""
        def _scroll_all():
            nodes = []
            offset = None
            # Local Qdrant re-evaluates filters for every page.  A larger page
            # materially reduces repeated scans for session/provenance reads.
            batch_size = min(limit, 1000)
            while len(nodes) < limit:
                scroll_result = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=scroll_filter,
                    limit=min(batch_size, limit - len(nodes)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                if isinstance(scroll_result, tuple):
                    points, next_offset = scroll_result
                else:
                    points = scroll_result.points
                    next_offset = scroll_result.next_page_offset
                for point in points:
                    node = self._payload_to_node(point.payload)
                    if with_vectors and point.vector:
                        node.embedding = self._dense_of(point.vector)
                    nodes.append(node)
                if next_offset is None or not points:
                    break
                offset = next_offset
            return nodes[:limit]

        try:
            nodes = await _run_in_vdb_pool(_scroll_all)
        except Exception as exc:
            logger.error(
                "[vector-store] %s failed: %s", operation, exc, exc_info=True,
            )
            return []
        logger.info("[vector-store] %s found=%d", operation, len(nodes))
        return nodes

    async def list_by_session(
        self,
        *,
        user_id: str,
        agent_id: Optional[str],
        session_id: str,
        limit: int = 10000,
        status_filter: Optional[List] = None,
        layers: Optional[List] = None,
        only_latest: bool = False,
        with_vectors: bool = False,
        allow_raw_layer: bool = False,
    ) -> List[MemoryNode]:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

        must = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="session_id", match=MatchValue(value=session_id)),
        ]
        if agent_id:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if only_latest:
            must.append(FieldCondition(key="is_latest", match=MatchValue(value=True)))
        if status_filter:
            must.append(FieldCondition(
                key="status",
                match=MatchAny(any=[
                    value.value if hasattr(value, "value") else str(value)
                    for value in status_filter
                ]),
            ))
        must_not = []
        if layers:
            layer_values = [
                value.value if hasattr(value, "value") else str(value)
                for value in layers
            ]
            if not allow_raw_layer:
                layer_values = [
                    value for value in layer_values
                    if value != MemoryLayer.L1_RAW.value
                ]
            if not layer_values:
                return []
            must.append(FieldCondition(
                key="layer", match=MatchAny(any=layer_values),
            ))
        elif not allow_raw_layer:
            must_not.append(FieldCondition(
                key="layer", match=MatchValue(value=MemoryLayer.L1_RAW.value),
            ))
        return await self._scroll_filtered_nodes(
            Filter(must=must, must_not=must_not or None),
            limit=limit,
            with_vectors=with_vectors,
            operation="list_by_session",
        )

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
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

        wanted = list(dict.fromkeys(value for value in episode_ids if value))
        if not wanted:
            return []
        must = [FieldCondition(key="episode_id", match=MatchAny(any=wanted))]
        must_not = []
        if user_id:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        if agent_id:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if layers:
            must.append(FieldCondition(
                key="layer", match=MatchAny(any=[layer.value for layer in layers]),
            ))
        if include_sources:
            must.append(FieldCondition(
                key="source", match=MatchAny(any=list(include_sources)),
            ))
        if exclude_sources:
            must_not.append(FieldCondition(
                key="source", match=MatchAny(any=list(exclude_sources)),
            ))
        return await self._scroll_filtered_nodes(
            Filter(must=must, must_not=must_not or None),
            limit=limit,
            with_vectors=False,
            operation="list_by_episode",
        )

    async def list_by_source_raw_ids(
        self,
        source_raw_ids: List[str],
        *,
        user_id: str,
        agent_id: Optional[str],
        limit: int = 10000,
        status_filter: Optional[List] = None,
        layers: Optional[List] = None,
        only_latest: bool = False,
        with_vectors: bool = False,
    ) -> List[MemoryNode]:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

        wanted = list(dict.fromkeys(str(value) for value in source_raw_ids if value))
        if not wanted:
            return []
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if agent_id:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if only_latest:
            must.append(FieldCondition(key="is_latest", match=MatchValue(value=True)))
        if status_filter:
            must.append(FieldCondition(
                key="status",
                match=MatchAny(any=[
                    value.value if hasattr(value, "value") else str(value)
                    for value in status_filter
                ]),
            ))
        if layers:
            must.append(FieldCondition(
                key="layer",
                match=MatchAny(any=[
                    value.value if hasattr(value, "value") else str(value)
                    for value in layers
                ]),
            ))
        provenance = MatchAny(any=wanted)
        return await self._scroll_filtered_nodes(
            Filter(
                must=must,
                should=[
                    FieldCondition(key="source_raw_memory_id", match=provenance),
                    FieldCondition(key="custom.source_raw_memory_ids", match=provenance),
                ],
            ),
            limit=limit,
            with_vectors=with_vectors,
            operation="list_by_source_raw_ids",
        )

    # ================================================================
    # Statistics
    # ================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            def _stats():
                return self._client.get_collection(self._collection_name)
            info = await _run_in_vdb_pool(_stats)
            return {
                "collection": self._collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            return {"error": str(e)}

    async def count(self, isolation_key: str) -> int:
        """Count the vectors under one isolation key"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            def _count():
                return self._client.count(
                    collection_name=self._collection_name,
                    count_filter=Filter(
                        must=[
                            FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key))
                        ]
                    ),
                    exact=True,
                )
            result = await _run_in_vdb_pool(_count)
            return result.count
        except Exception as e:
            logger.warning(f"Failed to count vectors for {isolation_key}: {e}")
            return 0

    async def close(self) -> None:
        """Close the client"""
        client, self._client = self._client, None
        if client is not None:
            await _run_in_vdb_pool(client.close)

    # ================================================================
    # Keyword Search (Full-text)
    # ================================================================

    def _ensure_text_index(self) -> None:
        """
        Backfill the search_text text index on an existing collection.

        Expected state: tokenizer = WHITESPACE (matching the jieba pre-tokenization in _node_to_payload).

        Three cases:
          1) no index -> create a WHITESPACE index
          2) a WHITESPACE index already exists -> skip
          3) a different tokenizer (e.g. an old MULTILINGUAL index) -> WARNING only, never rebuild on our own:
             existing payloads still carry search_text in the old format (raw untokenized original text), and a
             new tokenizer would yield useless tokens. Run scripts/migrate_search_text_to_jieba.py first to
             rewrite the payloads, then drop and recreate the index.
        """
        try:
            from qdrant_client.models import TextIndexParams, TokenizerType

            # Inspect the current state first
            existing_tokenizer = None
            try:
                info = self._client.get_collection(self._collection_name)
                payload_schema = getattr(info, "payload_schema", None) or {}
                schema_entry = payload_schema.get("search_text") if isinstance(payload_schema, dict) else None
                if schema_entry is not None:
                    params = getattr(schema_entry, "params", None)
                    tok = getattr(params, "tokenizer", None) if params else None
                    if tok is not None:
                        # The tokenizer qdrant returns may be an enum or a string
                        existing_tokenizer = getattr(tok, "value", None) or str(tok)
            except Exception as e:
                logger.debug(f"[qdrant] inspect search_text index failed (will try create anyway): {e}")

            if existing_tokenizer is not None:
                if str(existing_tokenizer).lower().endswith("whitespace") or str(existing_tokenizer).lower() == "whitespace":
                    return  # already the new format, nothing to do
                logger.warning(
                    f"[qdrant] search_text text index uses old tokenizer={existing_tokenizer!r} "
                    f"(expected whitespace). Keyword search will be DEGRADED on this collection until "
                    f"you run scripts/migrate_search_text_to_jieba.py to re-upsert payloads + rebuild index. "
                    f"collection={self._collection_name}"
                )
                return

            # No index -> just create one
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="search_text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WHITESPACE,
                    min_token_len=2,
                    max_token_len=20,
                ),
            )
            logger.info(
                "[qdrant] Created search_text text index (whitespace) on existing collection"
            )
        except Exception as e:
            # Fallback: unsupported or a concurrency race, stay silent
            logger.debug(f"[qdrant] _ensure_text_index swallowed: {e}")

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
        Keyword retrieval.

        - Collection carries a sparse "bm25" vector and fastembed is available (_supports_fulltext):
          use query_points(using="bm25") for real BM25 sparse retrieval, returning true BM25 scores
          (order 0~20+, normalized reader-side by the normalize_bm25 sigmoid).
        - Otherwise (old dense-only collection): degrade to [] and hybrid becomes pure vector search.
          Note: the old scroll text-match + binary 1.0 pseudo-BM25 is no longer used.
        """
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue, MatchAny,
        )

        if not query or not query.strip() or top_k <= 0:
            return []

        if not self._supports_fulltext:
            if not getattr(self, "_warned_no_fulltext", False):
                self._warned_no_fulltext = True
                logger.warning(
                    f"[qdrant] keyword_search degraded to empty: collection "
                    f"{self._collection_name} has no sparse 'bm25' vector (or fastembed "
                    f"missing). Recreate collection (auto-adds sparse) + re-upsert to "
                    f"enable real BM25. pip install fastembed"
                )
            return []

        from ..pipelines._retrieval import bm25_fastembed
        from qdrant_client.models import SparseVector

        q_sp = bm25_fastembed.encode_query(query.strip())
        if not q_sp:
            return []
        q_indices, q_values = q_sp

        must_conditions = []
        must_not_conditions = []
        if isolation_keys:
            must_conditions.append(
                FieldCondition(key="isolation_key", match=MatchAny(any=isolation_keys))
            )
        elif isolation_key:
            must_conditions.append(
                FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key))
            )
        elif user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )
        if agent_ids:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchAny(any=agent_ids))
            )
        if layers:
            must_conditions.append(
                FieldCondition(
                    key="layer",
                    match=MatchAny(any=[layer.value for layer in layers]),
                )
            )
        if status_filter:
            must_conditions.append(
                FieldCondition(key="status", match=MatchAny(any=[s.value for s in status_filter]))
            )
        else:
            must_conditions.append(
                FieldCondition(key="status", match=MatchValue(value=MemoryStatus.ACTIVE.value))
            )
        if only_latest:
            must_conditions.append(
                FieldCondition(key="is_latest", match=MatchValue(value=True))
            )
        if include_sources:
            must_conditions.append(
                FieldCondition(key="source", match=MatchAny(any=list(include_sources)))
            )
        if exclude_sources:
            must_not_conditions.append(
                FieldCondition(key="source", match=MatchAny(any=list(exclude_sources)))
            )
        query_filter = Filter(
            must=must_conditions,
            must_not=must_not_conditions or None,
        ) if (must_conditions or must_not_conditions) else None

        def _fetch_bm25(limit):
            resp = self._client.query_points(
                collection_name=self._collection_name,
                query=SparseVector(indices=q_indices, values=q_values),
                using=self._sparse_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return resp.points

        def _bm25_search():
            import os
            if os.getenv("MEMORY_BM25_STABLE_TIES_ENABLED", "false").lower() == "true":
                from ..utils.keyword_ties import bounded_keyword_ties
                return bounded_keyword_ties(_fetch_bm25, top_k)
            return _fetch_bm25(top_k)

        try:
            points = await _run_in_vdb_pool(_bm25_search)
        except Exception as e:
            logger.warning(f"[qdrant] keyword_search(bm25 sparse) failed: {e}")
            return []

        output = []
        for point in points:
            node = self._payload_to_node(point.payload)
            output.append({
                "node_id": node.node_id,
                "score": float(getattr(point, "score", 0.0) or 0.0),  # real BM25 score
                "node": node,
            })

        logger.debug(
            f"[vector-store] keyword_search(bm25): query='{query[:50]}' "
            f"user_id={user_id} top_k={top_k} found={len(output)}"
        )
        return output

    async def keyword_search_batch(
        self, requests: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Collapse independent remote sparse-BM25 searches into one RPC.

        Stable-tie mode keeps the scalar contract.  The first ``top_k + 1``
        page is fetched in the batch; only queries whose cutoff is inside an
        equal-score group fall back to the existing bounded expansion logic.
        """
        if not requests:
            return []
        if (
            self._is_local
            or not self._supports_fulltext
            or not hasattr(self._client, "query_batch_points")
        ):
            return await super().keyword_search_batch(requests)

        from ..pipelines._retrieval import bm25_fastembed
        from qdrant_client.models import QueryRequest, SparseVector

        stable_ties = (
            os.getenv("MEMORY_BM25_STABLE_TIES_ENABLED", "false").lower()
            == "true"
        )

        output: List[List[Dict[str, Any]]] = [[] for _ in requests]
        def _prepare_and_search_batch():
            batch = []
            valid_indices = []
            top_ks: Dict[int, int] = {}
            for index, raw in enumerate(requests):
                request = dict(raw)
                query = str(request.pop("query", "")).strip()
                top_k = int(request.pop("top_k", 10))
                if not query or top_k <= 0:
                    continue
                sparse_query = bm25_fastembed.encode_query(query)
                if not sparse_query:
                    continue
                query_filter = self._build_search_filter(
                    user_id=request.pop("user_id", None),
                    agent_ids=request.pop("agent_ids", None),
                    include_sources=request.pop("include_sources", None),
                    exclude_sources=request.pop("exclude_sources", None),
                    layers=request.pop("layers", None),
                    status_filter=request.pop("status_filter", None),
                    only_latest=request.pop("only_latest", True),
                )
                if request:
                    raise TypeError(
                        f"Unsupported Qdrant batch keyword arguments: {sorted(request)}"
                    )
                indices, values = sparse_query
                batch.append(QueryRequest(
                    query=SparseVector(indices=indices, values=values),
                    using=self._sparse_name,
                    filter=query_filter,
                    limit=top_k + 1 if stable_ties else top_k,
                    with_payload=True,
                    with_vector=False,
                ))
                valid_indices.append(index)
                top_ks[index] = top_k
            if not batch:
                return valid_indices, top_ks, []
            responses = self._client.query_batch_points(
                collection_name=self._collection_name,
                requests=batch,
            )
            return valid_indices, top_ks, responses

        try:
            valid_indices, top_ks, responses = await _run_in_vdb_pool(
                _prepare_and_search_batch
            )
            if len(responses) != len(valid_indices):
                raise RuntimeError(
                    f"Qdrant BM25 batch response count {len(responses)} "
                    f"!= {len(valid_indices)}"
                )
        except Exception as exc:
            logger.warning(
                "[qdrant] BM25 batch failed; falling back to scalar search: %s", exc,
            )
            return await super().keyword_search_batch(requests)

        ambiguous = []
        if stable_ties:
            from ..utils.keyword_ties import content_key

        for index, response in zip(valid_indices, responses):
            points = list(getattr(response, "points", []) or [])
            top_k = top_ks[index]
            if stable_ties:
                if len(points) > top_k:
                    boundary = float(points[top_k - 1].score)
                    if float(points[-1].score) >= boundary:
                        ambiguous.append(index)
                        continue
                points = sorted(points, key=content_key)[:top_k]
            output[index] = self._query_points_to_hits(points)

        if ambiguous:
            resolved = await super().keyword_search_batch(
                [requests[index] for index in ambiguous]
            )
            for index, hits in zip(ambiguous, resolved):
                output[index] = hits
        return output

    # ================================================================
    # Qdrant-specific methods
    # ================================================================

    @staticmethod
    def _patch_local_meta(persist_directory: str) -> None:
        """
        Compatibility patch: strip fields the current version cannot read from the local Qdrant meta.json.

        Newer qdrant-client (>=1.13) writes fields such as strict_mode_config and metadata into the
        collection config of meta.json, while the pydantic models of older versions (<1.13) set
        extra='forbid', so loading raises ValidationError.
        """
        import json as _json
        meta_path = Path(persist_directory) / "meta.json"
        if not meta_path.exists():
            return

        try:
            from qdrant_client.http.models import CreateCollection
            known_fields = set(CreateCollection.model_fields.keys())
        except Exception:
            return

        try:
            raw = meta_path.read_text(encoding="utf-8")
            meta = _json.loads(raw)
        except Exception:
            return

        changed = False
        for coll_name, coll_cfg in meta.get("collections", {}).items():
            if not isinstance(coll_cfg, dict):
                continue
            extra_keys = set(coll_cfg.keys()) - known_fields
            for k in extra_keys:
                del coll_cfg[k]
                changed = True

        if changed:
            meta_path.write_text(_json.dumps(meta), encoding="utf-8")
            logger.info("Patched Qdrant meta.json: removed unsupported fields")

    # ================================================================
    # Tag Index (per-user tag embeddings, used by reader_hybrid_tag)
    # ================================================================

    _supports_tag_index = True

    def _tag_index_collection(self) -> str:
        """
        Name of the separate tag_index collection; same database as the main memories collection, different table.

        Naming follows `_retrieval.config.tag_index_collection_name`.
        """
        override = str(
            getattr(self.config.vector_store, "tag_index_collection_name", "") or ""
        ).strip()
        if override:
            return override
        # Deferred import to avoid a circular dependency
        from ..pipelines._retrieval.config import tag_index_collection_name
        return tag_index_collection_name(self._collection_name)

    async def _ensure_tag_index_collection(self) -> None:
        """
        Ensure the tag_index collection exists. It is created lazily so an empty one does not hold resources at startup.

        Idempotent: safe to call repeatedly.
        """
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

        if self._tag_index_ready:
            return
        coll = self._tag_index_collection()

        def _ensure():
            with self._tag_index_init_lock:
                if self._tag_index_ready:
                    return
                collections = self._client.get_collections()
                names = [c.name for c in collections.collections]
                if coll not in names:
                    self._client.create_collection(
                        collection_name=coll,
                        vectors_config=VectorParams(
                            size=self.config.vector_store.embedding_dims,
                            distance=Distance.COSINE,
                        ),
                    )
                    for field_name, field_type in [
                        ("user_id", PayloadSchemaType.KEYWORD),
                        ("tag", PayloadSchemaType.KEYWORD),
                    ]:
                        try:
                            self._client.create_payload_index(
                                collection_name=coll,
                                field_name=field_name,
                                field_schema=field_type,
                            )
                        except Exception:
                            pass  # ignore conditions such as already existing
                    logger.info(f"Created Qdrant tag_index collection: {coll}")
                self._tag_index_ready = True

        await _run_in_vdb_pool(_ensure)

    async def upsert_tag_embedding(
        self, user_id: str, tag: str, embedding: List[float]
    ) -> None:
        from qdrant_client.models import PointStruct

        await self._ensure_tag_index_collection()
        coll = self._tag_index_collection()
        point_id = self._tag_point_id(user_id, tag)

        def _upsert():
            self._client.upsert(
                collection_name=coll,
                points=[PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"user_id": user_id, "tag": tag},
                )],
            )

        await _run_in_vdb_pool(_upsert)
        logger.debug(f"[tag-index] upsert: user={user_id} tag={tag}")

    async def existing_tag_embeddings(
        self, user_id: str, tags: List[str],
    ) -> List[str]:
        """Return tags whose deterministic points already exist, in one read."""
        if not tags:
            return []
        coll = self._tag_index_collection()
        point_ids = [self._tag_point_id(user_id, tag) for tag in tags]

        def _retrieve():
            try:
                return self._client.retrieve(
                    collection_name=coll,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                return []

        points = await _run_in_vdb_pool(_retrieve)
        found = []
        for point in points or []:
            payload = getattr(point, "payload", {}) or {}
            if payload.get("user_id") == user_id and payload.get("tag"):
                found.append(payload["tag"])
        return found

    async def upsert_tag_embeddings_batch(
        self, user_id: str, items: Dict[str, List[float]],
    ) -> int:
        """Upsert multiple per-user tag embeddings in one Qdrant request."""
        if not items:
            return 0
        from qdrant_client.models import PointStruct

        await self._ensure_tag_index_collection()
        coll = self._tag_index_collection()
        points = [
            PointStruct(
                id=self._tag_point_id(user_id, tag),
                vector=embedding,
                payload={"user_id": user_id, "tag": tag},
            )
            for tag, embedding in items.items()
        ]

        def _upsert():
            self._client.upsert(collection_name=coll, points=points)

        await _run_in_vdb_pool(_upsert)
        logger.debug(
            "[tag-index] batch upsert: user=%s tags=%d", user_id, len(points),
        )
        return len(points)

    async def has_tag_embedding(self, user_id: str, tag: str) -> bool:
        coll = self._tag_index_collection()
        point_id = self._tag_point_id(user_id, tag)

        def _retrieve():
            try:
                points = self._client.retrieve(
                    collection_name=coll,
                    ids=[point_id],
                    with_payload=False,
                    with_vectors=False,
                )
                return bool(points)
            except Exception:
                # A missing collection also counts as False
                return False

        return await _run_in_vdb_pool(_retrieve)

    async def delete_tag_embedding(self, user_id: str, tag: str) -> None:
        from qdrant_client.models import PointIdsList

        coll = self._tag_index_collection()
        point_id = self._tag_point_id(user_id, tag)

        def _delete():
            try:
                self._client.delete(
                    collection_name=coll,
                    points_selector=PointIdsList(points=[point_id]),
                )
            except Exception as e:
                logger.debug(f"[tag-index] delete no-op ({user_id}, {tag}): {e}")

        await _run_in_vdb_pool(_delete)

    async def search_tag_embeddings(
        self,
        user_id: str,
        query_embedding: List[float],
        topk: int = 5,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Vector search inside the per-user tag_index.

        Semantics:
          - **topk leads**: first take the topk closest tags by cosine
          - **min_score is a floor**: then softly drop the completely unrelated ones from topk
          - tag embeddings are weak in signal on their own (single word/phrase embeddings), so applying min_score as a
            server-side score_threshold would also block tags just under the threshold, producing an empty
            recall whenever the user has no highly related tag. This is a soft application-layer filter instead.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        coll = self._tag_index_collection()
        qf = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

        def _search():
            try:
                # No score_threshold is passed: topk first, min_score afterwards (soft filtering)
                if hasattr(self._client, "query_points"):
                    resp = self._client.query_points(
                        collection_name=coll,
                        query=query_embedding,
                        query_filter=qf,
                        limit=topk,
                        with_payload=True,
                        with_vectors=False,
                    )
                    return resp.points
                return self._client.search(
                    collection_name=coll,
                    query_vector=query_embedding,
                    query_filter=qf,
                    limit=topk,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                logger.debug(f"[tag-index] search failed (user={user_id}): {e}")
                return []

        hits = await _run_in_vdb_pool(_search)
        # Application-layer soft filter: topk is already fetched, only noise tags below min_score are dropped
        result: List[Dict[str, Any]] = []
        for h in hits or []:
            score = float(getattr(h, "score", 0.0))
            if score < min_score:
                continue
            payload = getattr(h, "payload", {}) or {}
            tag = payload.get("tag")
            if tag:
                result.append({"tag": tag, "score": score})
        return result

    async def count_memories_with_tag(
        self, user_id: str, tag: str, isolation_key: str = ""
    ) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        must = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            # On a list-valued field, Qdrant's MatchValue means "array contains value"
            FieldCondition(key="tags", match=MatchValue(value=tag)),
            FieldCondition(key="status", match=MatchAny(any=[MemoryStatus.ACTIVE.value])),
        ]
        if isolation_key:
            must.append(FieldCondition(key="isolation_key", match=MatchValue(value=isolation_key)))
        qf = Filter(must=must)

        def _count():
            try:
                result = self._client.count(
                    collection_name=self._collection_name,
                    count_filter=qf,
                    exact=True,
                )
                return int(getattr(result, "count", 0))
            except Exception as e:
                logger.debug(f"[tag-index] count_memories_with_tag failed ({user_id}, {tag}): {e}")
                return -1

        c = await _run_in_vdb_pool(_count)
        # Returns -1 when it fails (the caller can skip cleanup accordingly)
        return c

    # ================================================================
    # Hidden retrieval clauses (search-only parent aliases)
    # ================================================================

    def _retrieval_clause_collection(self) -> str:
        return f"{self._collection_name}__retrieval_clauses"

    @staticmethod
    def _retrieval_clause_point_id(parent_id: str, clause_index: int) -> str:
        import uuid
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"retrieval_clause::{parent_id}::{int(clause_index)}",
        ))

    async def _ensure_retrieval_clause_collection(self) -> None:
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

        if self._retrieval_clause_index_ready:
            return
        coll = self._retrieval_clause_collection()

        def _ensure():
            with self._retrieval_clause_index_init_lock:
                if self._retrieval_clause_index_ready:
                    return
                names = [c.name for c in self._client.get_collections().collections]
                if coll not in names:
                    self._client.create_collection(
                        collection_name=coll,
                        vectors_config=VectorParams(
                            size=self.config.vector_store.embedding_dims,
                            distance=Distance.COSINE,
                        ),
                    )
                    for field_name in ("parent_id", "user_id", "agent_id"):
                        try:
                            self._client.create_payload_index(
                                collection_name=coll,
                                field_name=field_name,
                                field_schema=PayloadSchemaType.KEYWORD,
                            )
                        except Exception:
                            pass
                self._retrieval_clause_index_ready = True

        await _run_in_vdb_pool(_ensure)

    async def delete_retrieval_clauses(self, parent_id: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        coll = self._retrieval_clause_collection()

        def _delete():
            try:
                self._client.delete(
                    collection_name=coll,
                    points_selector=FilterSelector(filter=Filter(must=[
                        FieldCondition(key="parent_id", match=MatchValue(value=parent_id)),
                    ])),
                    wait=True,
                )
                return -1
            except Exception:
                return 0

        return await _run_in_vdb_pool(_delete)

    async def delete_retrieval_clauses_by_metadata(
        self,
        *,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if agent_id is not None:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if session_id is not None:
            must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
        coll = self._retrieval_clause_collection()

        def _delete():
            try:
                self._client.delete(
                    collection_name=coll,
                    points_selector=FilterSelector(filter=Filter(must=must)),
                    wait=True,
                )
                return -1
            except Exception:
                return 0

        return await _run_in_vdb_pool(_delete)

    async def clone_retrieval_clauses(
        self, *, src_user_id: str, dst_user_id: str,
        id_map: Dict[str, str], agent_id: Optional[str] = None,
    ) -> int:
        """Copy stored clause vectors with new parent IDs, without embedding calls.

        Clone storage irrespective of the current feature flag: a disabled reader
        must not silently drop an existing index from a user snapshot.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

        coll = self._retrieval_clause_collection()

        def _clone():
            names = {c.name for c in self._client.get_collections().collections}
            if coll not in names:
                return 0
            must = [FieldCondition(key="user_id", match=MatchValue(value=src_user_id))]
            if agent_id is not None:
                must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
            offset, count = None, 0
            while True:
                source, next_offset = self._client.scroll(
                    collection_name=coll, scroll_filter=Filter(must=must),
                    limit=256, offset=offset, with_payload=True, with_vectors=True,
                )
                copies = []
                for point in source:
                    payload = dict(point.payload or {})
                    old_parent = payload.get("parent_id")
                    if old_parent not in id_map:
                        continue
                    if not point.vector:
                        raise ValueError("clone clause vector missing")
                    payload["parent_id"] = id_map[old_parent]
                    payload["user_id"] = dst_user_id
                    copies.append(PointStruct(
                        id=self._retrieval_clause_point_id(payload["parent_id"], payload["clause_index"]),
                        vector=point.vector, payload=payload,
                    ))
                if copies:
                    self._client.upsert(collection_name=coll, points=copies, wait=True)
                    count += len(copies)
                if next_offset is None:
                    return count
                offset = next_offset

        return await _run_in_vdb_pool(_clone)

    async def export_retrieval_clause_snapshot(
        self, *, src_user_id: str, agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read cloneable clause payloads/vectors once for a fan-out."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        coll = self._retrieval_clause_collection()

        def _export():
            names = {c.name for c in self._client.get_collections().collections}
            if coll not in names:
                return []
            must = [FieldCondition(key="user_id", match=MatchValue(value=src_user_id))]
            if agent_id is not None:
                must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
            rows = []
            offset = None
            while True:
                page, next_offset = self._client.scroll(
                    collection_name=coll,
                    scroll_filter=Filter(must=must),
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                for point in page:
                    if not point.vector:
                        raise ValueError("clone clause vector missing")
                    rows.append({
                        "payload": dict(point.payload or {}),
                        "vector": point.vector,
                    })
                if next_offset is None:
                    return rows
                offset = next_offset

        return await _run_in_vdb_pool(_export)

    async def import_retrieval_clause_snapshot(
        self,
        snapshot: List[Dict[str, Any]],
        *,
        dst_user_id: str,
        id_map: Dict[str, str],
    ) -> int:
        """Write a remapped clause snapshot for one destination user."""
        if not snapshot:
            return 0
        from qdrant_client.models import PointStruct

        coll = self._retrieval_clause_collection()

        def _import():
            copies = []
            for row in snapshot:
                payload = dict(row.get("payload") or {})
                old_parent = payload.get("parent_id")
                if old_parent not in id_map:
                    continue
                vector = row.get("vector")
                if not vector:
                    raise ValueError("clone clause vector missing")
                payload["parent_id"] = id_map[old_parent]
                payload["user_id"] = dst_user_id
                copies.append(PointStruct(
                    id=self._retrieval_clause_point_id(
                        payload["parent_id"], payload["clause_index"],
                    ),
                    vector=vector,
                    payload=payload,
                ))
            if copies:
                self._client.upsert(
                    collection_name=coll, points=copies, wait=True,
                )
            return len(copies)

        return await _run_in_vdb_pool(_import)

    async def replace_retrieval_clauses(
        self, parent: MemoryNode, clauses: List[str], embeddings: List[List[float]],
    ) -> int:
        from qdrant_client.models import PointStruct

        await self._ensure_retrieval_clause_collection()
        await self.delete_retrieval_clauses(parent.node_id)
        if not clauses:
            return 0
        if len(clauses) != len(embeddings):
            raise ValueError("retrieval clause count does not match embedding count")
        coll = self._retrieval_clause_collection()
        points = [
            PointStruct(
                id=self._retrieval_clause_point_id(parent.node_id, index),
                vector=embedding,
                payload={
                    "parent_id": parent.node_id,
                    "user_id": parent.user_id,
                    "agent_id": parent.agent_id or "",
                    "session_id": parent.session_id or "",
                    "memory_at": int(parent.memory_at.timestamp()) if parent.memory_at else None,
                    "clause_index": index,
                },
            )
            for index, (clause, embedding) in enumerate(zip(clauses, embeddings))
        ]

        def _upsert():
            self._client.upsert(collection_name=coll, points=points, wait=True)

        await _run_in_vdb_pool(_upsert)
        return len(points)

    async def search_retrieval_clauses(
        self,
        *,
        query_embedding: List[float],
        user_id: str,
        agent_ids: Optional[List[str]] = None,
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if agent_ids:
            must.append(FieldCondition(key="agent_id", match=MatchAny(any=agent_ids)))
        query_filter = Filter(must=must)
        coll = self._retrieval_clause_collection()

        def _search():
            try:
                if hasattr(self._client, "query_points"):
                    return self._client.query_points(
                        collection_name=coll,
                        query=query_embedding,
                        query_filter=query_filter,
                        limit=top_k,
                        score_threshold=min_score,
                        with_payload=True,
                        with_vectors=False,
                    ).points
                return self._client.search(
                    collection_name=coll,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=min_score,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.debug("[retrieval-clause] search failed: %s", exc)
                return []

        hits = await _run_in_vdb_pool(_search)
        best: Dict[str, Dict[str, Any]] = {}
        for hit in hits or []:
            payload = dict(getattr(hit, "payload", {}) or {})
            parent_id = str(payload.get("parent_id") or "")
            if not parent_id:
                continue
            item = {
                "parent_id": parent_id,
                "score": float(hit.score),
                "clause_index": payload.get("clause_index"),
            }
            if parent_id not in best or item["score"] > best[parent_id]["score"]:
                best[parent_id] = item
        return sorted(best.values(), key=lambda item: item["score"], reverse=True)

    # ================================================================
    # Entity Store (a separate {collection}_entities collection, aligned with mem0)
    #   - mirrors the tag_index aux-collection pattern
    #   - Qdrant payloads natively support lists, so linked_memory_ids stores a list directly (no JSON)
    # ================================================================

    def _entity_collection(self) -> str:
        """Name of the separate entity store collection (aligned with _retrieval.config.entity_collection_name)."""
        from ..pipelines._retrieval.config import entity_collection_name
        return entity_collection_name(self._collection_name)

    async def _ensure_entity_collection(self) -> None:
        """Create the entity store collection lazily. Idempotent."""
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

        coll = self._entity_collection()

        def _ensure():
            collections = self._client.get_collections()
            names = [c.name for c in collections.collections]
            if coll in names:
                return
            self._client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(
                    size=self.config.vector_store.embedding_dims,
                    distance=Distance.COSINE,
                ),
            )
            for field_name in ("user_id", "agent_id"):
                try:
                    self._client.create_payload_index(
                        collection_name=coll,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
            logger.info(f"Created Qdrant entity collection: {coll}")

        await _run_in_vdb_pool(_ensure)

    @staticmethod
    def _entity_filter(user_id: str, agent_ids: Optional[List[str]]):
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if agent_ids:
            if len(agent_ids) == 1:
                must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_ids[0])))
            else:
                must.append(FieldCondition(key="agent_id", match=MatchAny(any=list(agent_ids))))
        return Filter(must=must)

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
        import uuid as _uuid
        from qdrant_client.models import PointStruct

        await self._ensure_entity_collection()
        coll = self._entity_collection()
        qf = self._entity_filter(user_id, [agent_id] if agent_id else None)

        def _do():
            # 1) find the most similar entity of the same user
            try:
                if hasattr(self._client, "query_points"):
                    resp = self._client.query_points(
                        collection_name=coll, query=embedding, query_filter=qf,
                        limit=1, with_payload=True, with_vectors=False,
                    )
                    hits = resp.points
                else:
                    hits = self._client.search(
                        collection_name=coll, query_vector=embedding, query_filter=qf,
                        limit=1, with_payload=True, with_vectors=False,
                    )
            except Exception:
                hits = []

            if hits:
                top = hits[0]
                score = float(getattr(top, "score", 0.0))
                if score >= merge_threshold:
                    payload = dict(getattr(top, "payload", {}) or {})
                    linked = payload.get("linked_memory_ids") or []
                    if not isinstance(linked, list):
                        linked = []
                    if memory_id not in linked:
                        linked.append(memory_id)
                        # set_payload updates the payload only and leaves the vector untouched
                        self._client.set_payload(
                            collection_name=coll,
                            payload={"linked_memory_ids": linked},
                            points=[top.id],
                        )
                    return str(top.id)

            # 2) create a new one
            eid = str(_uuid.uuid4())
            self._client.upsert(
                collection_name=coll,
                points=[PointStruct(
                    id=eid,
                    vector=embedding,
                    payload={
                        "data": entity_text,
                        "entity_type": entity_type or "",
                        "linked_memory_ids": [memory_id],
                        "user_id": user_id,
                        "agent_id": agent_id or "",
                    },
                )],
            )
            return eid

        return await _run_in_vdb_pool(_do)

    async def search_entities(
        self,
        *,
        query_embedding: List[float],
        user_id: str,
        agent_ids: Optional[List[str]] = None,
        top_k: int = 500,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        coll = self._entity_collection()
        qf = self._entity_filter(user_id, agent_ids)

        def _search():
            try:
                if hasattr(self._client, "query_points"):
                    resp = self._client.query_points(
                        collection_name=coll, query=query_embedding, query_filter=qf,
                        limit=top_k, with_payload=True, with_vectors=False,
                    )
                    return resp.points
                return self._client.search(
                    collection_name=coll, query_vector=query_embedding, query_filter=qf,
                    limit=top_k, with_payload=True, with_vectors=False,
                )
            except Exception as e:
                logger.debug(f"[entity] search failed (user={user_id}): {e}")
                return []

        hits = await _run_in_vdb_pool(_search)
        out: List[Dict[str, Any]] = []
        for h in hits or []:
            score = float(getattr(h, "score", 0.0))
            if score < min_score:
                continue
            payload = getattr(h, "payload", {}) or {}
            linked = payload.get("linked_memory_ids") or []
            if not isinstance(linked, list):
                linked = []
            out.append({
                "entity_id": str(h.id),
                "data": payload.get("data", ""),
                "entity_type": payload.get("entity_type", ""),
                "linked_memory_ids": linked,
                "score": score,
            })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    async def list_entities(
        self, *, user_id: str, agent_ids: Optional[List[str]] = None, top_k: int = 10000
    ) -> List[Dict[str, Any]]:
        coll = self._entity_collection()
        qf = self._entity_filter(user_id, agent_ids)

        def _scroll():
            try:
                points, _ = self._client.scroll(
                    collection_name=coll, scroll_filter=qf,
                    limit=top_k, with_payload=True, with_vectors=False,
                )
                return points
            except Exception as e:
                logger.debug(f"[entity] list failed (user={user_id}): {e}")
                return []

        points = await _run_in_vdb_pool(_scroll)
        out: List[Dict[str, Any]] = []
        for p in points or []:
            payload = getattr(p, "payload", {}) or {}
            linked = payload.get("linked_memory_ids") or []
            if not isinstance(linked, list):
                linked = []
            out.append({
                "entity_id": str(p.id),
                "data": payload.get("data", ""),
                "entity_type": payload.get("entity_type", ""),
                "linked_memory_ids": linked,
            })
        return out

    async def delete_entities_for_memory(
        self, *, memory_id: str, user_id: str, agent_ids: Optional[List[str]] = None
    ) -> int:
        from qdrant_client.models import PointIdsList

        coll = self._entity_collection()
        qf = self._entity_filter(user_id, agent_ids)

        def _do():
            try:
                points, _ = self._client.scroll(
                    collection_name=coll, scroll_filter=qf,
                    limit=100000, with_payload=True, with_vectors=False,
                )
            except Exception as e:
                logger.debug(f"[entity] delete scroll failed (user={user_id}): {e}")
                return 0
            affected = 0
            for p in points or []:
                payload = dict(getattr(p, "payload", {}) or {})
                linked = payload.get("linked_memory_ids") or []
                if not isinstance(linked, list) or memory_id not in linked:
                    continue
                affected += 1
                remaining = [m for m in linked if m != memory_id]
                if not remaining:
                    try:
                        self._client.delete(
                            collection_name=coll,
                            points_selector=PointIdsList(points=[p.id]),
                        )
                    except Exception as e:
                        logger.debug(f"[entity] delete point {p.id} failed: {e}")
                else:
                    try:
                        self._client.set_payload(
                            collection_name=coll,
                            payload={"linked_memory_ids": remaining},
                            points=[p.id],
                        )
                    except Exception as e:
                        logger.debug(f"[entity] update point {p.id} failed: {e}")
            return affected

        return await _run_in_vdb_pool(_do)
