"""
Agent Memory V2 - VectorStore (FAISS)

FAISS-based vector storage layer.

Properties:
- Uses faiss-cpu, pure CPU computation
- Payloads live in an in-memory dict, metadata is filtered manually after search
- faiss.write_index() plus JSON to persist payloads
- Sync API wrapped into async by _run_in_vdb_pool()
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio
import concurrent.futures
import json
import logging
import threading
import numpy as np

from ..models.memory import (
    MemoryNode, MemoryLayer, MemoryStatus,
)
from ..config import MemoryConfig
from ..utils.async_executor import run_in_executor_guarded
from .vector_store_base import VectorStoreBase, passes_created_after

logger = logging.getLogger(__name__)

# Dedicated VDB thread pool
from .vector_store_chroma import _vdb_executor


def _run_in_vdb_pool(func, *args, **kwargs):
    """Run a sync function in the dedicated VDB thread pool."""
    import functools
    call = functools.partial(func, *args, **kwargs) if args or kwargs else func
    return run_in_executor_guarded(_vdb_executor, call)


class FaissVectorStore(VectorStoreBase):
    """
    FAISS vector store implementation

    Similarity search with faiss-cpu, payloads kept in an in-memory dict,
    persisted to disk (faiss index file + JSON payload file).
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._index = None
        # point_id (str) → { "payload": dict, "embedding": list, "idx": int }
        self._data: Dict[str, Dict[str, Any]] = {}
        # mapping from the faiss internal int index → point_id str
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx: int = 0
        self._lock = threading.Lock()
        self._dims = config.vector_store.embedding_dims

    async def initialize(self) -> None:
        """Initialise the FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS backend. "
                "Install with: pip install hy-memory[faiss]"
            )

        def _init():
            persist_dir = Path(self.config.vector_store.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            index_path = persist_dir / f"{self._collection_name}.faiss"
            payload_path = persist_dir / f"{self._collection_name}.json"

            if index_path.exists() and payload_path.exists():
                # load from disk
                self._index = faiss.read_index(str(index_path))
                with open(payload_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._data = saved.get("data", {})
                self._idx_to_id = {int(k): v for k, v in saved.get("idx_to_id", {}).items()}
                self._next_idx = saved.get("next_idx", 0)
                logger.info(
                    f"FAISS index loaded from {index_path}, "
                    f"points={self._index.ntotal}"
                )
            else:
                # create a new index (IndexFlatIP; cosine similarity requires normalising vectors first)
                self._index = faiss.IndexFlatIP(self._dims)
                logger.info(f"Created new FAISS index, dims={self._dims}")

        await _run_in_vdb_pool(_init)
        logger.debug(f"VectorStore initialized (FAISS), collection={self._collection_name}")

    # ================================================================
    # Persistence
    # ================================================================

    def _persist(self) -> None:
        """Persist the index and payloads to disk."""
        import faiss

        persist_dir = Path(self.config.vector_store.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        index_path = persist_dir / f"{self._collection_name}.faiss"
        payload_path = persist_dir / f"{self._collection_name}.json"

        faiss.write_index(self._index, str(index_path))
        saved = {
            "data": self._data,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "next_idx": self._next_idx,
        }
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False)

    @staticmethod
    def _normalize(vec: List[float]) -> np.ndarray:
        """L2 normalisation, used for cosine similarity."""
        arr = np.array(vec, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    # ================================================================
    # Writes
    # ================================================================

    async def upsert(self, node: MemoryNode) -> str:
        """Write or update one MemoryNode."""
        if node.embedding is None:
            raise ValueError(f"Node {node.node_id} has no embedding, cannot upsert to vector store")

        payload = self._node_to_payload(node)
        point_id = self._node_id_to_point_id(node.node_id)

        def _upsert():
            with self._lock:
                vec = self._normalize(node.embedding)

                if point_id in self._data:
                    # update: FAISS has no in-place update, record the new index position
                    old_idx = self._data[point_id]["idx"]
                    # mark the old index entry invalid
                    if old_idx in self._idx_to_id:
                        del self._idx_to_id[old_idx]

                idx = self._next_idx
                self._next_idx += 1
                self._index.add(vec)
                self._idx_to_id[idx] = point_id
                self._data[point_id] = {
                    "payload": payload,
                    "idx": idx,
                }
                self._persist()

        await _run_in_vdb_pool(_upsert)
        logger.debug(
            f"[vector-store] upsert: id={node.node_id} "
            f"layer={node.layer.value if hasattr(node.layer, 'value') else node.layer} "
            f"isolation_key={payload.get('isolation_key', '')} "
            f"content={node.content[:200]}"
        )
        return node.node_id

    async def upsert_batch(self, nodes: List[MemoryNode]) -> List[str]:
        """Batch write."""
        valid_nodes = [n for n in nodes if n.embedding is not None]
        if not valid_nodes:
            return []

        for n in nodes:
            if n.embedding is None:
                logger.warning(f"Skipping node {n.node_id}: no embedding")

        def _upsert_batch():
            with self._lock:
                ids = []
                vecs = []
                for node in valid_nodes:
                    point_id = self._node_id_to_point_id(node.node_id)
                    payload = self._node_to_payload(node)
                    vec = self._normalize(node.embedding)

                    if point_id in self._data:
                        old_idx = self._data[point_id]["idx"]
                        if old_idx in self._idx_to_id:
                            del self._idx_to_id[old_idx]

                    idx = self._next_idx
                    self._next_idx += 1
                    self._idx_to_id[idx] = point_id
                    self._data[point_id] = {
                        "payload": payload,
                        "idx": idx,
                    }
                    ids.append(node.node_id)
                    vecs.append(vec)

                if vecs:
                    batch = np.vstack(vecs)
                    self._index.add(batch)
                    self._persist()

                return ids

        return await _run_in_vdb_pool(_upsert_batch)

    async def update_embedding(self, node_id: str, embedding: List[float]) -> bool:
        """Update only the vector (payload unchanged)."""
        point_id = self._node_id_to_point_id(node_id)
        try:
            def _update():
                with self._lock:
                    if point_id not in self._data:
                        return False

                    old_idx = self._data[point_id]["idx"]
                    if old_idx in self._idx_to_id:
                        del self._idx_to_id[old_idx]

                    vec = self._normalize(embedding)
                    idx = self._next_idx
                    self._next_idx += 1
                    self._index.add(vec)
                    self._idx_to_id[idx] = point_id
                    self._data[point_id]["idx"] = idx
                    self._persist()
                    return True

            return await _run_in_vdb_pool(_update)
        except Exception as e:
            logger.warning(f"Failed to update embedding for {node_id}: {e}")
            return False

    async def update_payload(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update payload fields; if updates contains 'embedding', update the vector as well (re-index the entry)."""
        if not updates:
            return True
        point_id = self._node_id_to_point_id(node_id)
        updates = dict(updates)
        new_embedding = updates.pop("embedding", None)  # pull the vector out and handle it separately
        try:
            def _update():
                with self._lock:
                    if point_id not in self._data:
                        return False
                    # 1) payload fields
                    for k, v in updates.items():
                        self._data[point_id]["payload"][k] = v
                    # 2) vector: add it again (FaissFlat cannot change it in place, same approach as update_embedding)
                    if new_embedding is not None:
                        old_idx = self._data[point_id]["idx"]
                        if old_idx in self._idx_to_id:
                            del self._idx_to_id[old_idx]
                        vec = self._normalize(new_embedding)
                        idx = self._next_idx
                        self._next_idx += 1
                        self._index.add(vec)
                        self._idx_to_id[idx] = point_id
                        self._data[point_id]["idx"] = idx
                        self._data[point_id]["embedding"] = list(new_embedding)
                    self._persist()
                    return True
            result = await _run_in_vdb_pool(_update)
            if result:
                logger.debug(
                    f"[vector-store] update_payload: id={node_id} updates={list(updates.keys())} "
                    f"embedding={'yes' if new_embedding is not None else 'no'}"
                )
            return result
        except Exception as e:
            logger.warning(f"Failed to update payload for {node_id}: {e}")
            return False

    # ================================================================
    # Retrieval
    # ================================================================

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
        tags_match_any: Optional[List[str]] = None,  # reader_hybrid_tag route B; not implemented in faiss yet, ignored
        created_after: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic retrieval (FAISS search + manual metadata filtering)."""

        # precompute the filter sets
        keys_set = set(isolation_keys) if isolation_keys else None
        effective_uids = user_ids if user_ids else ([user_id] if user_id else [])
        uids_set = set(effective_uids) if effective_uids else None
        agent_ids_set = set(agent_ids) if agent_ids else None
        include_sources_set = set(include_sources) if include_sources else None
        exclude_sources_set = set(exclude_sources) if exclude_sources else set()

        layer_values = set(l.value for l in layers) if layers else None
        if status_filter:
            status_values = set(s.value for s in status_filter)
        else:
            status_values = {MemoryStatus.ACTIVE.value}

        def _search():
            with self._lock:
                if self._index.ntotal == 0:
                    return []

                vec = self._normalize(query_embedding)
                # IndexFlatIP supports IDSelector: build the allowed internal IDs from metadata first,
                # then let FAISS produce an exact top-k within that subset. Taking limit*N and filtering afterwards
                # would let the allow/deny lists eat the result quota and change retrieval semantics.
                import faiss
                allowed_indices = []
                for point_id, entry in self._data.items():
                    idx = entry.get("idx")
                    if idx is None or self._idx_to_id.get(int(idx)) != point_id:
                        continue
                    payload = entry.get("payload") or {}
                    if keys_set:
                        if payload.get("isolation_key") not in keys_set:
                            continue
                    elif isolation_key:
                        if payload.get("isolation_key") != isolation_key:
                            continue
                    elif uids_set:
                        if payload.get("user_id") not in uids_set:
                            continue
                        if agent_ids_set and payload.get("agent_id") not in agent_ids_set:
                            continue
                    source = payload.get("source") or "default"
                    if include_sources_set and source not in include_sources_set:
                        continue
                    if source in exclude_sources_set:
                        continue
                    if layer_values and payload.get("layer") not in layer_values:
                        continue
                    if payload.get("status") not in status_values:
                        continue
                    if only_latest and not payload.get("is_latest", True):
                        continue
                    if not passes_created_after(payload, created_after):
                        continue
                    allowed_indices.append(int(idx))
                if not allowed_indices:
                    return []
                selector = faiss.IDSelectorBatch(
                    np.asarray(allowed_indices, dtype=np.int64)
                )
                params = faiss.SearchParameters(sel=selector)
                k = min(limit, len(allowed_indices))
                scores, indices = self._index.search(vec, k, params=params)

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    point_id = self._idx_to_id.get(int(idx))
                    if point_id is None:
                        # index entry already deleted or superseded
                        continue

                    entry = self._data.get(point_id)
                    if entry is None:
                        continue

                    payload = entry["payload"]

                    # metadata filtering: exact keys > single key > user_ids/user_id + agent level
                    if keys_set:
                        if payload.get("isolation_key") not in keys_set:
                            continue
                    elif isolation_key:
                        if payload.get("isolation_key") != isolation_key:
                            continue
                    elif uids_set:
                        if payload.get("user_id") not in uids_set:
                            continue
                        if agent_ids_set and payload.get("agent_id") not in agent_ids_set:
                            continue
                    if layer_values and payload.get("layer") not in layer_values:
                        continue
                    if payload.get("status") not in status_values:
                        continue
                    # only search the tail nodes of evolution chains
                    if only_latest and not payload.get("is_latest", True):
                        continue
                    # time filter: event time memory_at wins, falling back to gmt_created when it is empty
                    if not passes_created_after(payload, created_after):
                        continue

                    # score is the inner product (after normalisation = cosine similarity)
                    sim = float(score)
                    if sim < score_threshold:
                        continue

                    results.append({
                        "point_id": point_id,
                        "payload": payload,
                        "score": sim,
                    })

                    if len(results) >= limit:
                        break

                return results

        raw_results = await _run_in_vdb_pool(_search)

        output = []
        for r in raw_results:
            node = self._payload_to_node(r["payload"])
            output.append({
                "node_id": node.node_id,
                "score": r["score"],
                "node": node,
            })

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

    async def get_by_id(self, node_id: str) -> Optional[MemoryNode]:
        """Get by id."""
        point_id = self._node_id_to_point_id(node_id)
        try:
            entry = self._data.get(point_id)
            if entry is None:
                return None
            node = self._payload_to_node(entry["payload"])
            # FAISS cannot look a vector up by id cheaply, skip embedding
            return node
        except Exception as e:
            logger.warning(f"Failed to get vector point {node_id}: {e}")
            return None

    async def get_embeddings(self, node_ids: List[str]) -> Dict[str, List[float]]:
        """Batch fetch vectors (reading embeddings straight from the in-memory _data) so dedup can compute cosine."""
        out: Dict[str, List[float]] = {}
        for nid in node_ids:
            point_id = self._node_id_to_point_id(nid)
            entry = self._data.get(point_id)
            if entry and entry.get("embedding"):
                out[nid] = list(entry["embedding"])
        return out

    # ================================================================
    # Deletions
    # ================================================================

    async def delete(self, node_id: str) -> bool:
        """Delete a single vector point."""
        point_id = self._node_id_to_point_id(node_id)
        try:
            def _delete():
                with self._lock:
                    if point_id not in self._data:
                        return False
                    idx = self._data[point_id]["idx"]
                    if idx in self._idx_to_id:
                        del self._idx_to_id[idx]
                    del self._data[point_id]
                    self._persist()
                    return True
            return await _run_in_vdb_pool(_delete)
        except Exception as e:
            logger.warning(f"Failed to delete vector point {node_id}: {e}")
            return False

    async def delete_by_isolation_key(self, isolation_key: str) -> int:
        """Delete every vector under one isolation key."""
        try:
            def _delete():
                with self._lock:
                    to_delete = []
                    for pid, entry in self._data.items():
                        if entry["payload"].get("isolation_key") == isolation_key:
                            to_delete.append(pid)

                    for pid in to_delete:
                        idx = self._data[pid]["idx"]
                        if idx in self._idx_to_id:
                            del self._idx_to_id[idx]
                        del self._data[pid]

                    if to_delete:
                        self._persist()
                    return len(to_delete)
            return await _run_in_vdb_pool(_delete)
        except Exception as e:
            logger.warning(f"Failed to delete vectors for {isolation_key}: {e}")
            return 0

    async def delete_by_metadata(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Delete vectors by a combination of metadata fields."""
        try:
            def _delete():
                with self._lock:
                    to_delete = []
                    for pid, entry in self._data.items():
                        payload = entry["payload"]
                        if payload.get("user_id") != user_id:
                            continue
                        if agent_id is not None and payload.get("agent_id") != agent_id:
                            continue
                        if session_id is not None and payload.get("session_id") != session_id:
                            continue
                        to_delete.append(pid)

                    for pid in to_delete:
                        idx = self._data[pid]["idx"]
                        if idx in self._idx_to_id:
                            del self._idx_to_id[idx]
                        del self._data[pid]

                    if to_delete:
                        self._persist()
                    return len(to_delete)
            return await _run_in_vdb_pool(_delete)
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
        """Enumerate a user's memory nodes, supporting multiple users / multiple agents

        allow_raw_layer: defaults to False, i.e. L1_RAW is not returned (external read path).
        """
        import asyncio
        import numpy as np

        effective_uids = list(user_ids) if user_ids else ([user_id] if user_id else [])
        if not effective_uids:
            logger.warning("[vector-store] list_by_user called without user_id/user_ids, refused")
            return []
        uid_set = set(effective_uids)
        aid_set = set(agent_ids) if agent_ids else ({agent_id} if agent_id else None)
        include_set = set(include_sources) if include_sources else None
        exclude_set = set(exclude_sources) if exclude_sources else set()

        status_values = {s.value if hasattr(s, 'value') else str(s) for s in status_filter} if status_filter else None
        layer_values = {l.value if hasattr(l, 'value') else str(l) for l in layers} if layers else None
        _raw_value = MemoryLayer.L1_RAW.value
        if layer_values is not None and not allow_raw_layer:
            layer_values.discard(_raw_value)
            if not layer_values:
                # only raw was named: use an empty allow-list that can never match, so the result is necessarily empty
                layer_values = {"__blocked__"}

        def _list():
            nodes = []
            with self._lock:
                for pid, entry in self._data.items():
                    payload = entry["payload"]
                    if payload.get("user_id") not in uid_set:
                        continue
                    if aid_set and payload.get("agent_id") not in aid_set:
                        continue
                    source = payload.get("source") or "default"
                    if include_set and source not in include_set:
                        continue
                    if source in exclude_set:
                        continue
                    if only_latest and not payload.get("is_latest", True):
                        continue
                    if status_values and payload.get("status", "") not in status_values:
                        continue
                    if layer_values and payload.get("layer", "") not in layer_values:
                        continue
                    if not allow_raw_layer and payload.get("layer", "") == _raw_value:
                        continue
                    node = self._payload_to_node(payload)
                    # read the vector from the FAISS index
                    idx = entry.get("idx")
                    if with_vectors and idx is not None and self._index is not None:
                        try:
                            vec = self._index.reconstruct(int(idx))
                            node.embedding = vec.tolist()
                        except Exception:
                            pass
                    nodes.append(node)
                    if len(nodes) >= limit:
                        break
            return nodes

        result = await _run_in_vdb_pool(_list)
        logger.info(
            f"[vector-store] list_by_user: user_ids={effective_uids} "
            f"agent_ids={sorted(aid_set) if aid_set else []} found={len(result)}"
        )
        return result

    # ================================================================
    # Statistics
    # ================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            return {
                "collection": self._collection_name,
                "points_count": len(self._data),
                "vectors_count": self._index.ntotal if self._index else 0,
                "status": "ok",
            }
        except Exception as e:
            return {"error": str(e)}

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
        wanted = {value for value in episode_ids if value}
        include_set = set(include_sources or [])
        exclude_set = set(exclude_sources or [])
        layer_set = {layer.value for layer in layers} if layers else None
        if not wanted:
            return []

        def _list():
            nodes = []
            with self._lock:
                for point_id, entry in self._data.items():
                    payload = entry.get("payload") or {}
                    if payload.get("episode_id") not in wanted:
                        continue
                    if user_id and payload.get("user_id") != user_id:
                        continue
                    if agent_id and payload.get("agent_id") != agent_id:
                        continue
                    if layer_set and payload.get("layer") not in layer_set:
                        continue
                    source = payload.get("source") or "default"
                    if include_set and source not in include_set:
                        continue
                    if source in exclude_set:
                        continue
                    idx = entry.get("idx")
                    if idx is None or self._idx_to_id.get(int(idx)) != point_id:
                        continue
                    nodes.append(self._payload_to_node(payload))
                    if len(nodes) >= limit:
                        break
            return nodes

        return await _run_in_vdb_pool(_list)

    async def count(self, isolation_key: str) -> int:
        """Count the vectors under one isolation key."""
        try:
            c = 0
            for entry in self._data.values():
                if entry["payload"].get("isolation_key") == isolation_key:
                    c += 1
            return c
        except Exception as e:
            logger.warning(f"Failed to count vectors for {isolation_key}: {e}")
            return 0

    async def close(self) -> None:
        """Close and persist."""
        if self._index is not None:
            try:
                await _run_in_vdb_pool(self._persist)
            except Exception as e:
                logger.warning(f"Failed to persist FAISS index on close: {e}")
            self._index = None
            self._data.clear()
            self._idx_to_id.clear()
