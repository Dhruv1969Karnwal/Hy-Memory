"""
HY Memory - Write (System 1)

Write flow:
  1. Validate parameters
  2. Vectorize (EmbedService)
  3. Qdrant persistence (VectorStore) - raw content is always stored as L1_RAW
  4. [optional] MemAgent smart processing (entity extraction, summary generation, conflict detection)
     - High-level info extracted by the agent is stored as L2_FACT/L3_SUMMARY (L4_IDENTITY is no longer separated; everything is L2_FACT)
     - intentions (forward-looking intentions) are stored as L7_INTENTION (with valid_until; expired ones are lazily converted to L2 by the reader)
     - after a successful extraction, L1_RAW is downgraded to SHADOW (excluded from recall);
       the same downgrade applies when the extraction result is empty (the extractor found no extractable fact) --
       "nothing worth recording" is a normal conclusion; the raw conversation should not stay in the recallable layer
  5. [optional] Merge check (Merger)

Mode behavior differences:
  - lite:  stores L1_RAW only, no LLM call, minimal embed-and-store
  - pro:   L1_RAW + MemAgent extracts high-level info → reconcile → L2/L4/L3
  - ultra: same as pro, but System2Writer then runs System 2 cognitive processing asynchronously
"""

import asyncio
import json
import os
import time
import weakref
from typing import Optional, Dict, Any, List, Tuple, Union
import uuid
from datetime import datetime
import logging

from .base import WritePipeline, WriteRequest, WriteResponse, PipelineContext
from ..config import MemoryConfig
from ..core.merger import Merger
from ..core.embed_service import EmbedService
from ..core.workload import (
    episode_slot,
    graph_slot,
    workload_snapshot,
    workload_scope,
)
from ..agent.mem_agent import MemAgent, ProcessMode
from ..agent.reconciler import MemoryReconciler
from ..agent.tools.basic_profile import upsert_basic_profile
from ..models.memory import MemoryNode, MemoryLayer, MemoryStatus, SourceType
from ..data.vector_store import create_vector_store
from ..data.vector_store_base import VectorStoreBase
from ..utils.tracer import PipelineTracer, create_tracer
from ..utils.log_setup import get_request_id
from ..utils.pipeline_observability import is_pipeline_trace_enabled
from ._retrieval import tag_index as _tag_index_helper
from ._retrieval.retrieval_clauses import (
    eligible_retrieval_clauses,
    sync_retrieval_clause_index,
)
from ._retrieval.session_episode import add_parent_episode_membership
from ..structure import merge_structural_custom, structural_custom

logger = logging.getLogger(__name__)

_RECONCILE_ENABLED = os.getenv("RECONCILE_ENABLED", "true").lower() == "true"
_MEMORY_AT_UNSET = object()


def _write_apply_verify_enabled() -> bool:
    """Enable expensive read-after-write audits only when explicitly requested."""
    return os.getenv("MEMORY_WRITE_APPLY_VERIFY_ENABLED", "false").lower() in (
        "true", "1", "yes",
    )

# Candidate lookup, reconcile and persistence for the same user must be serialized atomically. Otherwise
# two concurrent first writes could both observe empty L2/L4 and take the ADD fast path, producing duplicates. Bucketed per event loop to avoid reusing a Lock across loops.
_RECONCILE_LOOP_USER_LOCKS = weakref.WeakKeyDictionary()


def _reconcile_user_lock(user_id: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _RECONCILE_LOOP_USER_LOCKS.get(loop)
    if locks is None:
        locks = weakref.WeakValueDictionary()
        _RECONCILE_LOOP_USER_LOCKS[loop] = locks
    lock = locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[user_id] = lock
    return lock


def _count_user_owned(metas: List[Dict]) -> int:
    """Count user-side memories (a missing owner defaults to user, matching persistence)."""
    return sum(
        1 for m in (metas or [])
        if (m.get("owner") or "user") == "user"
    )


def _norm_owner(value: Any) -> Optional[str]:
    """Normalize the owner from extractor/reconcile: only 'user' / 'agent' are accepted, otherwise None."""
    if not value:
        return None
    v = str(value).strip().lower()
    if v in ("user", "agent"):
        return v
    # Compatibility: mem0-style 'assistant' → 'agent'
    if v == "assistant":
        return "agent"
    return None


class MemoryWriter(WritePipeline):
    """
    Core writer (System 1)

    Write flow: raw content stored as L1_RAW + single-path Qdrant storage + optional MemAgent LLM extraction of high-level info
    lite mode makes no LLM call; pro/ultra modes call MemAgent.
    """

    VERSION = "writer"

    def __init__(
        self,
        config: MemoryConfig,
        embed_service: Optional[EmbedService] = None,
        vector_store: Optional[VectorStoreBase] = None,
        cache=None,
        graph_store=None,
    ):
        self.config = config
        self._embed_service = embed_service
        self._external_vector_store = vector_store
        self._cache = cache
        # Memory relationship graph (pro mode). None means the graph is off → the whole projection is skipped.
        self._graph_store = graph_store

        self._merger: Optional[Merger] = None
        self._mem_agent: Optional[MemAgent] = None
        self._reconciler: Optional[MemoryReconciler] = None
        self._vector_store: Optional[VectorStoreBase] = None
        self._vector_store_initialized = False
        self._existing_tags_cache: Dict[Tuple[str, str], Tuple[float, Tuple[str, ...]]] = {}
        self._existing_tags_locks: Dict[Tuple[str, str], asyncio.Lock] = {}

        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._embed_service is None:
            self._embed_service = EmbedService(self.config)
        self._merger = Merger()
        self._mem_agent = MemAgent(self.config)
        self._reconciler = MemoryReconciler(
            self.config,
            llm_provider=self._mem_agent.llm_provider,
        )
        self._vector_store = self._external_vector_store or create_vector_store(self.config)
        if self._external_vector_store and getattr(self._external_vector_store, '_client', None):
            self._vector_store_initialized = True
        self._initialized = True
        logger.debug("MemoryWriter initialized")

    @property
    def embed_service(self) -> EmbedService:
        if self._embed_service is None:
            self._embed_service = EmbedService(self.config)
        return self._embed_service

    @property
    def merger(self) -> Merger:
        if self._merger is None:
            self._merger = Merger()
        return self._merger

    @property
    def mem_agent(self) -> MemAgent:
        if self._mem_agent is None:
            self._mem_agent = MemAgent(self.config)
        return self._mem_agent

    @property
    def llm_provider(self):
        """Shared chat LLM used by extraction and explicit episode finalization."""
        return self.mem_agent.llm_provider

    @property
    def reconciler(self) -> MemoryReconciler:
        if self._reconciler is None:
            self._reconciler = MemoryReconciler(
                self.config,
                llm_provider=self.mem_agent.llm_provider,
            )
        return self._reconciler

    async def _get_vector_store(self) -> VectorStoreBase:
        if self._vector_store is None:
            self._vector_store = self._external_vector_store or create_vector_store(self.config)
        if not self._vector_store_initialized:
            await self._vector_store.initialize()
            self._vector_store_initialized = True
        return self._vector_store

    # ================================================================
    # Private helpers
    # ================================================================

    @staticmethod
    def _build_custom(op, request_id: str, request: Optional[WriteRequest] = None) -> Dict[str, Any]:
        """Build the custom field of the VDB payload. Always includes request_id for traceability."""
        custom: Dict[str, Any] = {}
        if request_id:
            custom["request_id"] = request_id
        if op is not None and op.supersede_reason:
            custom["supersede_reason"] = op.supersede_reason
        MemoryWriter._attach_client_meta(custom, request)
        return custom

    def _session_episode_id(self, request: WriteRequest) -> str:
        """Stable parent id for facts when first-class session episodes are on."""
        if not getattr(self.config.recall, "session_episode_enabled", False):
            return ""
        from ..agent.episode_summary import session_episode_id
        return session_episode_id(
            request.user_id, request.agent_id or "default_agent",
            request.session_id or "default_session",
        )

    def _fact_custom(self, op, request_id: str, request: WriteRequest) -> Dict[str, Any]:
        custom = self._build_custom(op, request_id, request)
        if op is not None and getattr(
            getattr(self.config, "llm", None), "structure_metadata_enabled", False
        ):
            custom = merge_structural_custom(
                custom,
                topic_paths=getattr(op, "topic_paths", None),
                event_facet=getattr(op, "event_facet", None),
                topic_assignments=getattr(op, "topic_assignments", None),
                topic_paths_provided=bool(
                    getattr(op, "topic_paths_provided", False)
                    or getattr(op, "topic_paths", None)
                ),
                event_facet_provided=bool(
                    getattr(op, "event_facet_provided", False)
                    or getattr(op, "event_facet", None)
                ),
                topic_assignments_provided=bool(
                    getattr(op, "topic_assignments_provided", False)
                    or getattr(op, "topic_assignments", None)
                ),
            )
        parent = self._session_episode_id(request)
        if parent:
            custom = add_parent_episode_membership(
                custom, parent, similarity=1.0, method="write_session",
            )
        return custom

    @staticmethod
    def _merge_source_raw_ids(
        old_node: Optional[MemoryNode],
        source_raw_memory_id: str,
    ) -> Dict[str, Any]:
        """Preserve every raw source represented by an in-place fact merge."""
        custom = dict(getattr(old_node, "custom", None) or {})
        source_ids = list(custom.get("source_raw_memory_ids") or [])
        legacy_source = str(getattr(old_node, "source_raw_memory_id", "") or "")
        if legacy_source and legacy_source not in source_ids:
            source_ids.append(legacy_source)
        if (
            "s2_reviewed_source_count" not in custom
            and (int(custom.get("s2_reviewed_count") or 0) > 0
                 or int(custom.get("s2_evidence_count") or 0) > 0)
        ):
            # Freeze the provenance consumed by the previous evoling pass
            # before appending this write. The new source then reopens the node.
            custom["s2_reviewed_source_count"] = len(source_ids)
        for source_id in (str(source_raw_memory_id or ""),):
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
        if source_ids:
            custom["source_raw_memory_ids"] = source_ids
        return custom

    @staticmethod
    def _attach_client_meta(custom: Dict[str, Any], request: Optional[WriteRequest]) -> None:
        """Attach caller meta to a newly created node (a single-entry list).

        meta is stored as a list rather than a dict: each entry captures the context at that write,
        and later writes merging into the same node append to it instead of overwriting (see _merge_client_meta).
        A newly created node naturally holds only one entry.
        """
        meta = getattr(request, "client_meta", None) if request is not None else None
        if meta:
            custom["meta"] = [dict(meta)]

    @staticmethod
    def _merge_client_meta(
        old_custom: Optional[Dict[str, Any]],
        request: Optional[WriteRequest],
    ) -> Optional[Dict[str, Any]]:
        """Append this call's meta to the old node's meta list and return the full custom to write back.

        Used for in-place updates (UPDATE): the node identity is unchanged so the old meta must be kept, otherwise
        the context of this memory's earlier writes would be lost.

        Returning None means custom needs no change (no meta was passed), so the caller skips the read-back and write-back and
        callers not using this feature pay no extra cost.
        """
        meta = getattr(request, "client_meta", None) if request is not None else None
        if not meta:
            return None

        merged = dict(old_custom or {})
        existing = merged.get("meta")
        # Tolerance: legacy data or manual edits may leave meta as a non-list; instead of overwriting it,
        # demote it to the first element of the list to avoid silently dropping data
        if isinstance(existing, list):
            chain = list(existing)
        elif existing:
            chain = [existing]
        else:
            chain = []
        chain.append(dict(meta))
        merged["meta"] = chain
        return merged

    async def _maybe_index_entities(
        self, vector_store, node, request,
    ) -> Any:
        """When the entity_store switch is on and this is an L2_FACT, refresh the entity store after persisting (best-effort).

        Covers both the reconcile path (ADD/UPDATE/SUPERSEDE) and the first-write path (_direct_store).
        """
        try:
            if not getattr(self.config.recall, "entity_store_enabled", False):
                return
            if node is None or node.layer != MemoryLayer.L2_FACT:
                return
            from ._retrieval.entity_store import index_memory_entities
            await index_memory_entities(
                vector_store=vector_store,
                embed_service=self.embed_service,
                memory_id=node.node_id,
                content=node.content or "",
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
            )
        except Exception as e:
            logger.debug(f"[entity-index] skipped (non-fatal): {e}")

    @staticmethod
    def _collect_new_memories(
        extracted_info: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict]]:
        """
        Collect the new memory texts and the full meta list from the extract result.

        Note: the extractor returns basic_info in its JSON output and the writer handles it separately
        (via upsert_basic_profile() into the L0_BASIC_INFO evolution chain), so it is not collected here.

        Returns:
            (new_memory_texts, new_memories_meta)
            each meta: {"content", "layer", "tags"}
        """
        new_memory_texts: List[str] = []
        new_memories_meta: List[Dict] = []

        # 1) memory → one memory per entry (all L2_FACT)
        #    The new extractor emits `memory`; the legacy `facts` field is still supported.
        for item in (extracted_info.get("memory") or extracted_info.get("facts") or []):
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not content:
                continue
            new_memory_texts.append(content)
            new_memories_meta.append({
                "content": content,
                "layer": "L2_FACT",
                "tags": item.get("tags") or [],
                "owner": _norm_owner(item.get("owner")),
                "topic_paths": item.get("topic_paths") or [],
                "event_facet": item.get("event_facet") or {},
                "topic_assignments": item.get("topic_assignments") or [],
                "topic_paths_provided": "topic_paths" in item,
                "event_facet_provided": "event_facet" in item,
                "topic_assignments_provided": "topic_assignments" in item,
            })

        # 2) Backward compatibility: identity items from the old extractor also go into L2_FACT
        #    (the new extractor no longer emits identity; L4_IDENTITY is no longer written, only read from historical data)
        for item in (extracted_info.get("identity") or []):
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not content:
                continue
            new_memory_texts.append(content)
            new_memories_meta.append({
                "content": content,
                "layer": "L2_FACT",
                "tags": item.get("tags") or [],
                "owner": _norm_owner(item.get("owner")),
                "topic_paths": item.get("topic_paths") or [],
                "event_facet": item.get("event_facet") or {},
                "topic_assignments": item.get("topic_assignments") or [],
                "topic_paths_provided": "topic_paths" in item,
                "event_facet_provided": "event_facet" in item,
                "topic_assignments_provided": "topic_assignments" in item,
            })

        # 3) Compatibility with the legacy extract output (profile + facts) -- basic_info is not handled
        if not new_memory_texts:
            profile = extracted_info.get("profile", {})
            if profile and isinstance(profile, dict):
                profile_items = []
                for k, v in profile.items():
                    if k == "preferences":
                        continue
                    if v and str(v).lower() not in ("null", "none", ""):
                        profile_items.append(f"{k}: {v}")
                prefs = profile.get("preferences", [])
                if prefs and isinstance(prefs, list):
                    pref_strs = [str(p) for p in prefs if p]
                    if pref_strs:
                        profile_items.append(f"preferences: {', '.join(pref_strs)}")
                if profile_items:
                    t = "; ".join(profile_items)
                    new_memory_texts.append(t)
                    new_memories_meta.append({"content": t, "layer": "L2_FACT", "tags": []})
            for f in (extracted_info.get("facts") or []):
                if isinstance(f, dict) and f.get("content"):
                    new_memory_texts.append(f["content"])
                    new_memories_meta.append({"content": f["content"], "layer": "L2_FACT", "tags": f.get("tags") or []})

        return new_memory_texts, new_memories_meta

    async def _dedup_extracted(
        self,
        new_memory_texts: List[str],
        new_memories_meta: List[Dict],
        request: "WriteRequest",
        req_id: str,
    ) -> Tuple[List[str], List[Dict], Dict[str, List[float]]]:
        """Deduplicate the freshly extracted memories against each other (before storing).

        These entries are not persisted yet, have no node_id and no evolution chain → all are treated as non-chain, embedded once more,
        delete_from_store=False (only drop duplicates, never delete from the store), and a DEDUP log is recorded.
        Retention: the list index acts as a deterministic gmt (earlier entries win, matching extractor output order).

        Embed: prefer one batch HTTP call; only if that batch fails do we degrade to parallel per-item calls, balancing happy-path throughput
        with partial success during outages. DEDUP log.parsed records wall time and each item's meta.
        """
        import asyncio
        import time
        from ..pipelines._retrieval.dedup import DedupItem, execute_dedup

        async def _embed_one(text: str):
            t0 = time.perf_counter()
            try:
                vec, meta = await self.embed_service.embed_queued_with_meta(text)
                meta = dict(meta or {})
                meta["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                return vec, meta
            except Exception as e:
                return e

        # All texts enter the write queue concurrently; EmbedService batches them by queue_batch_size.
        # A PMV2 chunk yielding 15 memories converges from 15 HTTP calls to about 2 batch requests; compared with
        # forcing one huge batch, this keeps single-request wall time low and still allows merging across writes.
        t_wall0 = time.perf_counter()
        embed_results = await asyncio.gather(
            *[_embed_one(t) for t in new_memory_texts],
            return_exceptions=True,
        )
        embed_elapsed_ms = round((time.perf_counter() - t_wall0) * 1000, 2)
        per_item_elapsed_ms = round(
            embed_elapsed_ms / max(1, len(new_memory_texts)), 2,
        )

        embeds: List[Any] = []
        embed_metas: List[Dict[str, Any]] = []
        for i, r in enumerate(embed_results):
            if isinstance(r, Exception):
                logger.warning(
                    f"[write] extractor dedup embed[{i}] failed: {r}"
                )
                embeds.append(None)
                embed_metas.append({
                    "index": i,
                    "error": type(r).__name__,
                    "error_msg": str(r)[:300],
                })
            elif isinstance(r, tuple) and len(r) == 2:
                vec, meta = r
                embeds.append(vec)
                m = dict(meta or {})
                m["index"] = i
                m.setdefault("elapsed_ms", per_item_elapsed_ms)
                embed_metas.append(m)
            else:
                embeds.append(None)
                embed_metas.append({
                    "index": i,
                    "error": "unexpected_result",
                    "error_msg": repr(r)[:300],
                })

        items: List[DedupItem] = []
        for i, (text, emb) in enumerate(zip(new_memory_texts, embeds)):
            if not emb:
                continue
            items.append(DedupItem(
                node_id=str(i),               # use the index as a temporary id
                embedding=emb,
                content=text,
                is_latest=True,
                is_chain_head=False,
                gmt_created=float(i),         # earlier entries get a smaller gmt → kept first
                chain_node_ids=[str(i)],
            ))

        embed_info = {
            "embedding_source": "embed_service",
            "embed_elapsed_ms": embed_elapsed_ms,
            "embed_count": len(new_memory_texts),
            "embed_ok_count": sum(1 for e in embeds if e),
            "embed_metas": embed_metas,
        }

        # Emit a DEDUP log even with fewer than 2 comparable items, to audit embed latency / provider meta
        plan = await execute_dedup(
            items,
            vector_store=None,                # not stored yet
            cache=self._cache,
            trigger="extractor",
            request_id=req_id,
            user_id=request.user_id,
            agent_id=request.agent_id or "default_agent",
            delete_from_store=False,
            extra_parsed=embed_info,
        )
        drop_idx = {int(d) for d in plan.get("delete_ids", [])}
        kept_embeddings = {
            text: emb
            for text, emb in zip(new_memory_texts, embeds)
            if text and emb
        }
        if not drop_idx:
            return new_memory_texts, new_memories_meta, kept_embeddings

        kept_texts, kept_meta = [], []
        for i, (t, m) in enumerate(zip(new_memory_texts, new_memories_meta)):
            if i in drop_idx:
                continue
            kept_texts.append(t)
            kept_meta.append(m)
        logger.info(
            f"[write] extractor dedup dropped {len(drop_idx)} of {len(new_memory_texts)} extracted"
        )
        kept_embedding_map = {
            text: kept_embeddings[text]
            for text in kept_texts
            if text in kept_embeddings
        }
        return kept_texts, kept_meta, kept_embedding_map

    @staticmethod
    def _collect_intentions(extracted_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect intentions from the extract result (forward-looking, stored as L7_INTENTION).

        Each intention: {"content", "tags", "valid_until"}
        valid_until is the ISO date string emitted by the extractor (or null); it is parsed here into
        a datetime (23:59:59 of that day, lenient to end of day); parse failure or absence → None.
        """
        out: List[Dict[str, Any]] = []
        for item in (extracted_info.get("intentions") or []):
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            valid_until: Optional[datetime] = None
            raw_vu = item.get("valid_until")
            if raw_vu and str(raw_vu).lower() not in ("null", "none", ""):
                try:
                    # Accepts "YYYY-MM-DD" or a full ISO string; always normalized to end of day
                    d = datetime.fromisoformat(str(raw_vu).strip())
                    valid_until = d.replace(hour=23, minute=59, second=59, microsecond=0)
                except (ValueError, TypeError):
                    valid_until = None
            out.append({
                "content": content,
                "tags": item.get("tags") or [],
                "valid_until": valid_until,
                "owner": _norm_owner(item.get("owner")),
            })
        return out

    async def _store_intentions(
        self,
        intentions: List[Dict[str, Any]],
        request: WriteRequest,
        vector_store: VectorStoreBase,
        req_id: str,
    ) -> List[str]:
        """
        Upsert intentions directly as L7_INTENTION nodes (bypassing reconcile).

        Intentions are point-in-time signals and are never merged into existing facts; once expired the reader lazily
        converts them into L2_FACT. Returns the list of new node ids.

        **In-batch dedup (0805 fix)**: one extractor run may return several near-duplicate intentions with cosine≥0.95
        (LLM repetition), so this function does a mutually exclusive in-batch check after embedding and before upsert:
            Rule: same valid_until day (both None or equal dates) plus cosine ≥
                  MEMORY_DEDUP_THRESHOLD (default 0.95) → treated as duplicates
            (memory_at is identical within one request, so it needs no comparison)
            Retained: the earliest entry in the list (the first one in extractor output order)
        No VDB reads and no "new write vs existing store" comparison, to avoid adding read load to the VDB.
        Pollution already in the store is cleaned up gradually by async deletion on the recall side. Best-effort: any exception
        falls back to "no dedup, store everything" and never affects the main upsert flow.
        """
        if not intentions:
            return []

        # Batch embed
        contents = [it["content"] for it in intentions]
        embeddings: List[Optional[List[float]]] = [None] * len(contents)
        try:
            batch = await self.embed_service.embed_batch(contents)
            for i, emb in enumerate(batch):
                embeddings[i] = emb
        except Exception as e:
            logger.warning(f"[intention] batch embed failed, falling back to sequential: {e}")

        # Fill embed gaps one by one in order (when embed_batch fails)
        for i, it in enumerate(intentions):
            if embeddings[i] is None:
                try:
                    embeddings[i] = await self.embed_service.embed_queued(it["content"])
                except Exception as e:
                    logger.error(f"[intention] embed failed, skip: {e}")

        # ============ In-batch dedup (no VDB lookup, mutually exclusive check within this batch only) ============
        keep_mask: List[bool] = [True] * len(intentions)
        try:
            dropped = self._intention_batch_dedup(intentions, embeddings)
            for i in dropped:
                if 0 <= i < len(keep_mask):
                    keep_mask[i] = False
            if dropped:
                logger.info(
                    f"[intention] batch dedup dropped {len(dropped)} of "
                    f"{len(intentions)} extracted intentions"
                )
        except Exception as e:
            # Dedup failed → fall back to "upsert everything" (conservative)
            logger.warning(f"[intention] batch dedup failed (skip, upsert all): {e}")

        stored: List[str] = []
        for i, it in enumerate(intentions):
            if not keep_mask[i]:
                continue
            emb = embeddings[i]
            if emb is None:
                continue
            node = MemoryNode(
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                source=request.source,
                layer=MemoryLayer.L7_INTENTION,
                content=it["content"],
                owner=it.get("owner") or "user",
                supersedes=None,
                is_latest=True,
                source_type=SourceType.INFERRED,
                status=MemoryStatus.ACTIVE,
                embedding=emb,
                memory_at=request.memory_at,
                valid_until=it.get("valid_until"),
                tags=list(it.get("tags") or []),
                custom=self._build_custom(None, req_id, request),
            )
            try:
                nid = await vector_store.upsert(node)
                stored.append(nid)
                logger.debug(
                    f"[intention] L7 upsert: {it['content'][:80]} → node_id={nid} "
                    f"valid_until={it.get('valid_until')}"
                )
                # INTENTION_UPSERT pipeline log (intentions skip reconcile, instrumented separately) +
                # memory_operations audit. Carries memory_id so production can trace by request_id / memory_id.
                await self._emit_intention_upsert(
                    request,
                    vector_store=vector_store,
                    memory_id=nid,
                    content=it["content"],
                    tags=list(it.get("tags") or []),
                    valid_until=it.get("valid_until"),
                )
                if self._cache:
                    try:
                        await self._cache.store_memory_operation(
                            request_id=req_id,
                            user_id=request.user_id,
                            agent_id=request.agent_id or "default_agent",
                            op="INTENTION_UPSERT",
                            memory_id=nid,
                            content=it["content"],
                            layer=MemoryLayer.L7_INTENTION.value,
                            reason="intention (S1 direct upsert, no reconcile)",
                            supersedes=[],
                        )
                    except Exception as e:
                        logger.warning(f"[intention] store_memory_operation failed: {e}")
            except Exception as e:
                logger.error(f"[intention] upsert failed: {e}")
        return stored

    @staticmethod
    def _intention_batch_dedup(
        intentions: List[Dict[str, Any]],
        embeddings: List[Optional[List[float]]],
    ) -> List[int]:
        """In-batch dedup for L7 intentions (pure function).

        Duplicate rule:
          Same valid_until day (both None or equal dates) + cosine≥MEMORY_DEDUP_THRESHOLD
          → treated as duplicates. Within a group the smallest index is kept; the rest go to drop_idx.

        memory_at belongs to the same write request and is therefore identical, so it is not compared here.
        This function does no I/O and has no side effects, so it is test friendly.

        Returns:
            List[int]: indices of the intentions to drop (ascending).
        """
        import os

        try:
            threshold = float(os.environ.get("MEMORY_DEDUP_THRESHOLD", 0.95))
        except (TypeError, ValueError):
            threshold = 0.95

        def _vu_key(it: Dict[str, Any]) -> Any:
            """Normalize valid_until into a date key: None → None; datetime → date."""
            vu = it.get("valid_until")
            if vu is None:
                return None
            try:
                return vu.date() if hasattr(vu, "date") else vu
            except Exception:
                return None

        def _cosine(a: List[float], b: List[float]) -> float:
            if not a or not b:
                return float("-inf")
            la = sum(x * x for x in a) ** 0.5
            lb = sum(x * x for x in b) ** 0.5
            if la == 0.0 or lb == 0.0:
                return float("-inf")
            return sum(x * y for x, y in zip(a, b)) / (la * lb)

        n = len(intentions)
        if n < 2:
            return []

        # Bucket by valid_until
        buckets: Dict[Any, List[int]] = {}
        for i in range(n):
            if embeddings[i] is None:
                continue
            buckets.setdefault(_vu_key(intentions[i]), []).append(i)

        drop_idx = set()
        for key, idx_list in buckets.items():
            if len(idx_list) < 2:
                continue
            # Compare pairs inside the bucket by cosine; >= threshold drops the later one (smallest index kept)
            for a_pos in range(len(idx_list)):
                ia = idx_list[a_pos]
                if ia in drop_idx:
                    continue
                for b_pos in range(a_pos + 1, len(idx_list)):
                    ib = idx_list[b_pos]
                    if ib in drop_idx:
                        continue
                    if _cosine(embeddings[ia], embeddings[ib]) >= threshold:
                        drop_idx.add(ib)

        return sorted(drop_idx)

    # ================================================================
    # Episode (internal session): group "one thing" spread across write batches under the same episode_id
    # ================================================================
    #
    # Why it is needed: one discussion usually spans several `write()` calls (a dozen or twenty turns), and the memories
    # extracted from it belong to the same thing. Tagging them with one episode_id lets the read side **scope a candidate set** for weighting.
    #
    # Why no temporal edges: 20 memories in a day fully linked is 190 edges and degree 19; one hop pulls in the whole day.
    # One field plus one filtered query achieves the same, and it **does not depend on the graph** (it works with graph disabled).
    #
    # Why not reuse session_id: that is the caller's **business isolation key** (it goes into isolation_key),
    # possibly very long (a whole conversation) or literally default_session, so its granularity does not match "one discussion".
    #
    # Partitioning needs no LLM: a gap since the last write within `episode_gap_seconds` stays in the same episode,
    # a longer gap starts a new one; exceeding `episode_max_memories` also forces a roll so a long session does not become one huge episode.

    async def _resolve_episode_id(self, request: WriteRequest, n_new: int) -> str:
        """Return the episode_id this write belongs to; an empty string when the cache is unavailable (feature degrades)."""
        if self._cache is None:
            return ""
        rcfg = getattr(self.config, "recall", None)
        gap = float(getattr(rcfg, "episode_gap_seconds", 1800.0) or 1800.0)
        cap = int(getattr(rcfg, "episode_max_memories", 50) or 50)

        key = "::".join([
            request.user_id or "default",
            request.agent_id or "default_agent",
            request.session_id or "default_session",
        ])
        now_ts = (
            request.memory_at.timestamp() if request.memory_at
            else datetime.now().timestamp()
        )
        try:
            state = await self._cache.get_episode_state(key)
            if state and state.get("episode_id"):
                idle = now_ts - float(state.get("last_write_ts") or 0)
                count = int(state.get("memory_count") or 0)
                if 0 <= idle <= gap and count < cap:
                    episode_id = state["episode_id"]
                    count += n_new
                else:
                    # The previous episode closes here -- no new content will arrive for it, so this is the moment to summarize it.
                    # It is summarized once, with no need to feed the old summary back to the LLM for a merge (unlike rolling summary,
                    # which rewrites everything every N turns).
                    self._schedule_episode_summary(request, state["episode_id"])
                    episode_id, count = f"ep_{uuid.uuid4().hex[:16]}", n_new
            else:
                episode_id, count = f"ep_{uuid.uuid4().hex[:16]}", n_new

            await self._cache.upsert_episode_state(
                key,
                user_id=request.user_id or "",
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                episode_id=episode_id,
                last_write_ts=now_ts,
                memory_count=count,
            )
            await self._log_episode(
                request,
                episode_id=episode_id,
                reused=(state or {}).get("episode_id") == episode_id,
                idle_s=(
                    now_ts - float((state or {}).get("last_write_ts") or 0)
                    if state else None
                ),
                count=count,
                n_new=n_new,
                gap=gap,
                cap=cap,
            )
            return episode_id
        except Exception as e:
            logger.debug(f"[episode] resolve failed (non-fatal): {e}")
            return ""

    def _make_trace_sink(self, request: WriteRequest, step: str):
        """Give background summary tasks a `trace(prompt, response, parsed, elapsed_ms)` callback.

        Those tasks live in `agent/` and `_graph/` and do not carry cache/request context themselves;
        passing a closure is less error-prone than letting each of them assemble its own instrumentation arguments.
        """
        if not self._cache or not is_pipeline_trace_enabled():
            return None
        req_id = request.request_id or get_request_id() or ""
        user_id = request.user_id
        agent_id = request.agent_id or "default_agent"
        session_id = request.session_id or ""

        async def _sink(
            *, prompt: str = "", response: str = "",
            parsed: Optional[Dict[str, Any]] = None,
            memory_ids: Optional[List[str]] = None,
            elapsed_ms: float = 0.0,
            tokens: Optional[Dict[str, int]] = None,
        ) -> None:
            try:
                tk = tokens or {}
                await self._cache.store_pipeline_log(
                    request_id=req_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    step=step,
                    prompt=prompt,
                    response=response,
                    parsed=json.dumps(parsed or {}, ensure_ascii=False, default=str),
                    memory_ids=memory_ids or None,
                    elapsed_ms=float(elapsed_ms or 0.0),
                    prompt_tokens=int(tk.get("prompt", 0) or 0),
                    completion_tokens=int(tk.get("completion", 0) or 0),
                    total_tokens=int(tk.get("total", 0) or 0),
                )
            except Exception as e:
                logger.debug(f"[trace] {step} failed: {e}")

        return _sink

    async def _log_episode(
        self,
        request: WriteRequest,
        *,
        episode_id: str,
        reused: bool,
        idle_s: Optional[float],
        count: int,
        n_new: int,
        gap: float,
        cap: int,
    ) -> None:
        """Instrumentation for the episode partitioning decision.

        Partitioning gates everything downstream (read-side scoping and summary triggers follow it), so the **decision basis must be reviewable**:
        reused or new, how long since the last write, how many accumulated, and the thresholds. Without it, production only shows "the summary did not fire"
        or "the scoped candidates are wrong", with no way to attribute that to the split itself.
        """
        if not self._cache or not is_pipeline_trace_enabled():
            return
        parsed = {
            "episode_id": episode_id,
            "decision": "reuse" if reused else "new",
            "idle_seconds": round(idle_s, 3) if idle_s is not None else None,
            "user_memories_total": count,
            "user_memories_new": n_new,
            "gap_seconds": gap,
            "max_memories": cap,
        }
        try:
            await self._cache.store_pipeline_log(
                request_id=request.request_id or get_request_id() or "",
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step="EPISODE",
                prompt="",
                response=json.dumps(parsed, ensure_ascii=False, default=str),
                parsed=json.dumps(parsed, ensure_ascii=False, default=str),
                elapsed_ms=0,
            )
        except Exception as e:
            logger.debug(f"[episode] EPISODE trace failed: {e}")

    def _schedule_episode_summary(self, request: WriteRequest, episode_id: str) -> None:
        """Summarize a closed episode once in the background (fire-and-forget, never blocking the write)."""
        rcfg = getattr(self.config, "recall", None)
        if not episode_id or rcfg is None:
            return
        if not getattr(rcfg, "episode_summary_enabled", False):
            return
        if self.llm_provider is None or self.embed_service is None:
            return
        try:
            from ..agent.episode_summary import summarize_closed_episode

            async def _run():
                try:
                    nid = await summarize_closed_episode(
                        vector_store=self.vector_store,
                        embed_service=self.embed_service,
                        llm_provider=self.llm_provider,
                        episode_id=episode_id,
                        user_id=request.user_id,
                        agent_id=request.agent_id,
                        session_id=request.session_id or "",
                        min_memories=int(
                            getattr(rcfg, "episode_summary_min_memories", 3) or 3
                        ),
                        # This is an LLM call: its prompt and output must be traceable in production,
                        # otherwise a badcase shows only "the summary content is wrong" without the input
                        trace=self._make_trace_sink(request, "EPISODE_SUMMARY"),
                    )
                    if nid:
                        logger.info(
                            f"[episode-summary] episode={episode_id[:12]} → {nid[:12]}"
                        )
                except Exception as e:
                    logger.warning(f"[episode-summary] failed (non-fatal): {e}")

            asyncio.create_task(_run())
        except Exception as e:
            logger.debug(f"[episode-summary] schedule failed: {e}")

    # ================================================================
    # Memory relationship graph (pro mode, docs/pro_graph_design.md)
    # ================================================================

    @staticmethod
    def _graph_stub_node(
        node_id: str, content: str, layer, tags: List[str], request: WriteRequest,
        *, memory_at: Any = _MEMORY_AT_UNSET,
        custom: Optional[Dict[str, Any]] = None,
    ) -> MemoryNode:
        """Build a lightweight MemoryNode for an in-place UPDATE so the graph projection can use it.

        UPDATE goes through `update_payload` and has no MemoryNode object at hand; the graph side only needs
        node_id / content / tags / layer / isolation_key, not the embedding.
        """
        return MemoryNode(
            node_id=node_id,
            user_id=request.user_id,
            agent_id=request.agent_id or "default_agent",
            session_id=request.session_id or "default_session",
            source=request.source,
            layer=layer,
            content=content,
            tags=list(tags or []),
            custom=dict(custom or {}),
            memory_at=(request.memory_at if memory_at is _MEMORY_AT_UNSET else memory_at),
        )

    def _schedule_graph_projection(
        self,
        *,
        graph_payload: List[Tuple[Any, Any]],
        superseded_old_ids: List[str],
        candidate_scores: Dict[str, List[Tuple[str, float]]],
        request: Optional[WriteRequest] = None,
    ) -> None:
        """Start this request's graph projection early; a synchronous write waits for it before returning.

        Graph and Episode work may overlap, but the number of projections is bounded by a process-level gate. Kuzu's
        executor has a single worker, so Kuzu projections are serialized per request to keep several
        projections' fine-grained Cypher calls from interleaving into one very long queue.
        """
        graph_store = getattr(self, "_graph_store", None)
        if graph_store is None or not graph_payload:
            return None
        gcfg = getattr(self.config, "graph_store", None)
        if gcfg is not None and not getattr(gcfg, "memory_graph_write", True):
            return None

        async def _run_unbounded() -> None:
            try:
                from ._graph.projector import (
                    project_memories_to_graph, sync_graph_on_supersede,
                )

                nodes = [n for n, _op in graph_payload]
                # The target_id values in op.related are ids of existing memories
                related_by_node = {
                    n.node_id: list(getattr(op, "related", None) or [])
                    for n, op in graph_payload
                    if getattr(op, "related", None)
                }
                # Align candidate-pool scores to the just-persisted node_ids by content
                neighbors_by_node = {}
                for n, _op in graph_payload:
                    cands = (candidate_scores or {}).get(n.content or "")
                    if cands:
                        neighbors_by_node[n.node_id] = cands

                _t_proj = datetime.now()
                _proj_stats = await project_memories_to_graph(
                    graph_store=graph_store,
                    embed_service=self.embed_service,
                    nodes=nodes,
                    vector_store=await self._get_vector_store(),
                    related_by_node=related_by_node,
                    neighbors_by_node=neighbors_by_node,
                    edge_min_score=getattr(gcfg, "edge_min_score", 0.55),
                    co_occurs_max_batch=getattr(gcfg, "co_occurs_max_batch", 8),
                    temporal_spine=getattr(gcfg, "temporal_spine", False),
                    hierarchy_topic_write=getattr(
                        gcfg, "hierarchy_topic_write", False,
                    ),
                )
                if superseded_old_ids:
                    await sync_graph_on_supersede(graph_store, superseded_old_ids)
                # The graph is built asynchronously fire-and-forget and failures only warn -- without instrumentation it is completely invisible
                # whether it was built at all and how much (the usual reason the read side sees "no graph effect" is a failed write-side projection)
                _sink = self._make_trace_sink(request, "GRAPH_PROJECT")
                if _sink is not None:
                    _st = dict(_proj_stats or {})
                    _st["superseded_synced"] = len(superseded_old_ids or [])
                    await _sink(
                        parsed=_st,
                        response=json.dumps(_st, ensure_ascii=False, default=str),
                        memory_ids=[
                            getattr(n, "node_id", "") for n in nodes
                            if getattr(n, "node_id", "")
                        ],
                        elapsed_ms=(datetime.now() - _t_proj).total_seconds() * 1000,
                    )
                await self._maybe_topic_summaries(nodes, request)
            except Exception as e:
                logger.warning(f"[graph] projection failed (non-fatal): {e}")

        async def _run() -> None:
            provider_obj = getattr(
                getattr(self.config, "graph_store", None), "provider", ""
            )
            provider = str(getattr(provider_obj, "value", provider_obj) or "").lower()
            async with graph_slot(kuzu=(provider == "kuzu")):
                await _run_unbounded()

        try:
            import asyncio
            task = asyncio.get_running_loop().create_task(_run())
            if request is not None:
                # Request-local handle: a finalize for this exact write may wait for
                # its typed edges without coupling concurrent users to one global task.
                setattr(request, "_graph_projection_task", task)
            return task
        except RuntimeError:
            # No running loop (synchronous context) → skip outright; never block the main flow for the graph
            logger.debug("[graph] no running loop, projection skipped")
            return None

    @staticmethod
    async def _await_graph_projection(request: Optional[WriteRequest]) -> None:
        """Join a request's graph work exactly once before returning."""
        if request is None:
            return
        task = getattr(request, "_graph_projection_task", None)
        if task is None:
            return
        try:
            await task
        except Exception as exc:
            logger.warning("[graph] projection join failed (non-fatal): %s", exc)
        finally:
            try:
                delattr(request, "_graph_projection_task")
            except AttributeError:
                pass

    def _attach_graph_executor_diagnostics(self, response: WriteResponse) -> None:
        snapshot = getattr(self._graph_store, "executor_snapshot", None)
        if callable(snapshot):
            response.extra["graph_executor"] = snapshot()

    async def _maybe_topic_summaries(
        self, nodes: List[Any], request: Optional[WriteRequest] = None,
    ) -> None:
        """Refresh Topic-scoped summaries on demand (summarize per topic instead of per time window).

        Off by default (`MEMORY_TOPIC_SUMMARY_ENABLED`) -- it costs extra LLM calls.
        The whole block is wrapped in try: summaries are extra artifacts and failures do not affect the memories themselves.
        """
        gcfg = getattr(self.config, "graph_store", None)
        if gcfg is None or not getattr(gcfg, "topic_summary_enabled", False):
            return
        graph_store = getattr(self, "_graph_store", None)
        if graph_store is None or not nodes:
            return
        try:
            from ._graph.topic_summary import maybe_summarize_topics

            first = nodes[0]
            isolation_key = first.get_isolation_key()
            # Which topics this write touched: look up topic_id by tag name
            tag_names = {
                str(t) for n in nodes for t in (getattr(n, "tags", None) or []) if t
            }
            if not tag_names:
                return
            topic_ids = [
                t["topic_id"]
                for t in await graph_store.get_all_topics(
                    MemoryNode.topic_scope_of(isolation_key)
                )
                if t.get("name") in tag_names and t.get("topic_id")
            ]
            if not topic_ids:
                return

            await maybe_summarize_topics(
                graph_store=graph_store,
                vector_store=await self._get_vector_store(),
                embed_service=self.embed_service,
                llm_provider=self.reconciler._get_llm(),
                user_id=getattr(first, "user_id", ""),
                agent_id=getattr(first, "agent_id", None),
                session_id=getattr(first, "session_id", ""),
                isolation_key=isolation_key,
                touched_topic_ids=topic_ids,
                min_memories=getattr(gcfg, "topic_summary_min_memories", 5),
                step=getattr(gcfg, "topic_summary_step", 5),
                max_concurrency=getattr(gcfg, "topic_summary_concurrency", 3),
                trace=(
                    self._make_trace_sink(request, "TOPIC_SUMMARY")
                    if request is not None else None
                ),
            )
        except Exception as e:
            logger.warning(f"[topic-summary] failed (non-fatal): {e}")

    async def _reconcile_and_store(
        self,
        new_memory_texts: List[str],
        new_memories_meta: List[Dict],
        request: WriteRequest,
        vector_store: VectorStoreBase,
        req_id: str,
        source_raw_memory_id: str = "",
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[List[str], Optional[str], Dict[str, int], Dict[str, int]]:
        """Serialize reconcile/apply per user, preserving timeline semantics."""
        async with _reconcile_user_lock(request.user_id or "default"):
            return await self._reconcile_and_store_locked(
                new_memory_texts=new_memory_texts,
                new_memories_meta=new_memories_meta,
                request=request,
                vector_store=vector_store,
                req_id=req_id,
                source_raw_memory_id=source_raw_memory_id,
                precomputed_embeddings=precomputed_embeddings,
            )

    async def _reconcile_and_store_locked(
        self,
        new_memory_texts: List[str],
        new_memories_meta: List[Dict],
        request: WriteRequest,
        vector_store: VectorStoreBase,
        req_id: str,
        source_raw_memory_id: str = "",
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[List[str], Optional[str], Dict[str, int], Dict[str, int]]:
        """
        Reconcile the new memories, execute ADD (including EVOLVE), and write the WRITE_SUMMARY log.

        Returns:
            (stored_ids, error_message, ops_counts, recon_tokens)
            error_message=None means success; ops_counts holds add/supersede/update/total;
            recon_tokens holds prompt/completion/total (LLM usage of the reconcile pipeline)
        """
        stored_ids: List[str] = []
        # Episode: the internal session id this write belongs to (groups "one thing" across writes).
        # Only user-side memories are counted (matching rolling summary counting only pending_user_count):
        # letting agent-side memories count would fill the episode to its cap early, split too finely, and pay summary tokens for nothing.
        episode_id = await self._resolve_episode_id(
            request, _count_user_owned(new_memories_meta)
        )
        session_episode_id = self._session_episode_id(request)
        if session_episode_id:
            episode_id = session_episode_id
        # For the graph: collect the (node, op) pairs persisted by this write so the projection can build edges.
        # op is collected to get op.related (the typed relations annotated by the LLM).
        graph_payload: List[Tuple[Any, Any]] = []
        superseded_old_ids: List[str] = []
        tags_to_index: set[str] = set()
        current_time = request.memory_at.isoformat(timespec="seconds") if request.memory_at else ""

        recon_result = await self.reconciler.reconcile(
            new_memories=new_memory_texts,
            user_id=request.user_id,
            agent_id=request.agent_id or "default_agent",
            vector_store=vector_store,
            embed_service=self.embed_service,
            layers=[MemoryLayer.L2_FACT, MemoryLayer.L4_IDENTITY],
            cache=self._cache,
            request_id=req_id,
            current_time=current_time,
            new_memories_with_meta=new_memories_meta,
            session_id=request.session_id or "",
            graph_store=getattr(self, "_graph_store", None),
            new_memory_embeddings=precomputed_embeddings,
            source=request.source,
        )

        if not recon_result.success:
            return [], recon_result.error, {"add": 0, "supersede": 0, "update": 0, "total": 0}, {"prompt": 0, "completion": 0, "total": 0}

        # Counts by op type
        _op_counts = {}
        for op in recon_result.ops:
            _op_counts[op.op] = _op_counts.get(op.op, 0) + 1
        _op_counts_str = " ".join(f"{k}={v}" for k, v in sorted(_op_counts.items()))

        logger.info(
            f"TRACE_PERF [{req_id}] S1_RECONCILE_DONE "
            f"ops={len(recon_result.ops)} candidates={len(new_memory_texts)} | {_op_counts_str}"
            + (f" | DEGRADED({recon_result.degraded_reason})"
               if getattr(recon_result, "degraded", False) else "")
        )
        if getattr(recon_result, "degraded", False):
            # Took the ADD-all fallback: content is preserved, but dedup/evolution judgements were skipped. This must be observable,
            # otherwise production only shows "there are more memories" without saying why.
            logger.warning(
                f"[write] reconcile degraded → ADD-all "
                f"({recon_result.degraded_reason}), {len(recon_result.ops)} memories stored as-is"
            )
        # Per-op details are emitted by the reconciler as a single "[reconciler] ops detail" JSON list,
        # so they are not printed one by one here (avoiding duplicate log spam).

        # ── Batch embed: collect every content that needs embedding and do one batch call ──
        contents_to_embed: List[str] = []
        content_indices: List[int] = []  # matching index into ops
        for i, op in enumerate(recon_result.ops):
            if op.op in ("SUPERSEDE", "UPDATE"):
                if op.op == "UPDATE" and not op.content:
                    continue  # shadow-only, no embed needed
                if not op.memory_id:
                    continue
                contents_to_embed.append(op.content or "")
                content_indices.append(i)
            elif op.op == "ADD":
                contents_to_embed.append(op.content or "")
                content_indices.append(i)

        # One batch embed (no serial per-item calls)
        embeddings_map: Dict[int, List[float]] = {}
        reconciled_embeddings = getattr(
            recon_result, "new_memory_embeddings", None,
        ) or {}
        # Only reuse vectors for the exact reconcile input texts. If the LLM rewrites content into UPDATE/SUPERSEDE,
        # the content differs and misses, so the embedding is still computed for the new content.
        for idx, content in zip(content_indices, contents_to_embed):
            emb = reconciled_embeddings.get(content)
            if emb:
                embeddings_map[idx] = emb

        missing_pairs = [
            (idx, content)
            for idx, content in zip(content_indices, contents_to_embed)
            if idx not in embeddings_map
        ]
        if missing_pairs:
            try:
                batch_embeddings = await self.embed_service.embed_batch(
                    [content for _, content in missing_pairs]
                )
                for (idx, _), emb in zip(missing_pairs, batch_embeddings):
                    embeddings_map[idx] = emb
            except Exception as e:
                logger.warning(f"[reconciler] batch embed failed, falling back to sequential: {e}")
                # fallback: embed one by one
                for idx, content in missing_pairs:
                    try:
                        embeddings_map[idx] = await self.embed_service.embed_queued(content)
                    except Exception as e2:
                        logger.error(f"[reconciler] embed failed for op {idx}: {e2}")

        # Counters
        add_cnt = 0           # ADD (brand new)
        supersede_cnt = 0     # SUPERSEDE (contradictory evolution)
        update_cnt = 0        # UPDATE (merged refinement)

        # A fresh-user reconcile commonly produces an ADD-only batch (PMV2
        # chunks often contain 7-15 facts).  Every vector-store backend already
        # supports upsert_batch, but the writer historically sent one network
        # request per fact.  Batch only the dependency-free ADD-only case;
        # UPDATE/SUPERSEDE keep their original ordered application semantics.
        ops_to_apply = list(enumerate(recon_result.ops))
        batch_update = getattr(vector_store, "update_payload_batch", None)
        if (
            len(recon_result.ops) >= 2
            and all(
                op.op == "UPDATE" and op.memory_id and op.content
                for op in recon_result.ops
            )
            and not getattr(request, "client_meta", None)
            and callable(batch_update)
        ):
            update_rows = []
            update_nodes: List[Tuple[int, Any, Any]] = []
            # UPDATE refines one durable memory in place. Its logical event time
            # belongs to the original memory and must not jump to the latest write
            # merely because another occurrence/detail was reconciled into it.
            # Fetch all targets once so graph projection uses the same preserved
            # timestamp. If a legacy node has no time, the new known time may fill it.
            existing_update_nodes: Dict[str, MemoryNode] = {}
            try:
                existing_update_nodes = {
                    node.node_id: node
                    for node in await vector_store.get_by_ids([
                        op.memory_id for op in recon_result.ops if op.memory_id
                    ])
                    if node is not None
                }
            except Exception as e:
                logger.warning(
                    "[reconciler] batch UPDATE target lookup failed; "
                    "preserving stored memory_at implicitly: %s", e,
                )
            for op_idx, op in enumerate(recon_result.ops):
                content = op.content or ""
                layer = (
                    MemoryLayer.from_string(op.layer)
                    if op.layer else MemoryLayer.L2_FACT
                )
                updates = {
                    "content": content,
                    "source": request.source,
                    "embedding": (
                        embeddings_map.get(op_idx)
                        or await self.embed_service.embed_queued(content)
                    ),
                    "tags": list(op.tags or []),
                    "tag_ners": list(getattr(op, "tag_ners", None) or []),
                    "layer": layer.value,
                    "gmt_modified": int(datetime.now().timestamp()),
                    "is_latest": True,
                    "status": MemoryStatus.ACTIVE.value,
                }
                old_node = existing_update_nodes.get(op.memory_id)
                if (
                    old_node is not None
                    and old_node.memory_at is None
                    and request.memory_at is not None
                ):
                    updates["memory_at"] = int(request.memory_at.timestamp())
                if source_raw_memory_id:
                    updates["source_raw_memory_id"] = source_raw_memory_id
                if session_episode_id:
                    updates["episode_id"] = (
                        getattr(old_node, "episode_id", "") or session_episode_id
                    )
                merged_custom = self._merge_source_raw_ids(
                    old_node, source_raw_memory_id,
                )
                if getattr(
                    getattr(self.config, "llm", None),
                    "structure_metadata_enabled", False,
                ):
                    merged_custom = merge_structural_custom(
                        merged_custom,
                        topic_paths=getattr(op, "topic_paths", None),
                        event_facet=getattr(op, "event_facet", None),
                        topic_assignments=getattr(op, "topic_assignments", None),
                        topic_paths_provided=bool(
                            getattr(op, "topic_paths_provided", False)
                            or getattr(op, "topic_paths", None)
                        ),
                        event_facet_provided=bool(
                            getattr(op, "event_facet_provided", False)
                            or getattr(op, "event_facet", None)
                        ),
                        topic_assignments_provided=bool(
                            getattr(op, "topic_assignments_provided", False)
                            or getattr(op, "topic_assignments", None)
                        ),
                    )
                if session_episode_id:
                    merged_custom = add_parent_episode_membership(
                        merged_custom, session_episode_id,
                        similarity=1.0, method="write_session",
                    )
                if merged_custom:
                    updates["custom"] = merged_custom
                update_rows.append((op.memory_id, updates))
                update_nodes.append((op_idx, self._graph_stub_node(
                    op.memory_id, content, layer, list(op.tags or []), request,
                    memory_at=(
                        old_node.memory_at
                        if old_node is not None and old_node.memory_at is not None
                        else request.memory_at if old_node is not None else None
                    ),
                    custom=merged_custom,
                ), op))

            updated_ids = set(await batch_update(update_rows) or [])
            for _op_idx, node, op in update_nodes:
                if node.node_id not in updated_ids:
                    logger.warning(
                        "[reconciler] UPDATE batch did not return node id %s",
                        node.node_id,
                    )
                    continue
                stored_ids.append(node.node_id)
                graph_payload.append((node, op))
                update_cnt += 1
                await self._emit_reconcile_apply(
                    request, vector_store=vector_store, op_type="UPDATE",
                    memory_id=node.node_id, layer=node.layer.value,
                    content=node.content,
                )
                tags_to_index.update(t for t in (op.tags or []) if t)
                if self._cache:
                    try:
                        await self._cache.store_memory_operation(
                            request_id=req_id,
                            user_id=request.user_id,
                            agent_id=request.agent_id or "default_agent",
                            op="UPDATE",
                            memory_id=node.node_id,
                            content=node.content,
                            layer=node.layer.value,
                            reason=op.reason,
                        )
                    except Exception as e:
                        logger.warning(
                            "[reconciler] store_memory_operation(UPDATE) failed: %s", e,
                        )
            logger.info(
                "TRACE_PERF [%s] S1_UPDATE_BATCH_DONE nodes=%d",
                req_id, len(updated_ids),
            )
            ops_to_apply = []

        if (
            ops_to_apply
            and len(recon_result.ops) >= 2
            and all(op.op == "ADD" for op in recon_result.ops)
            and callable(getattr(vector_store, "upsert_batch", None))
        ):
            add_nodes: List[Tuple[Any, Any]] = []
            for op_idx, op in enumerate(recon_result.ops):
                content = op.content or ""
                layer = (
                    MemoryLayer.from_string(op.layer)
                    if op.layer else MemoryLayer.L2_FACT
                )
                add_nodes.append((MemoryNode(
                    user_id=request.user_id,
                    agent_id=request.agent_id or "default_agent",
                    session_id=request.session_id or "default_session",
                    source=request.source,
                    layer=layer,
                    content=content,
                    episode_id=episode_id,
                    owner=getattr(op, "owner", None),
                    speculate=op.speculate,
                    supersedes=None,
                    is_latest=True,
                    source_type=SourceType.INFERRED,
                    status=MemoryStatus.ACTIVE,
                    embedding=(
                        embeddings_map.get(op_idx)
                        or await self.embed_service.embed_queued(content)
                    ),
                    memory_at=request.memory_at,
                    source_raw_memory_id=source_raw_memory_id or None,
                    tags=list(op.tags or []),
                    tag_ners=list(getattr(op, "tag_ners", None) or []),
                    custom=self._fact_custom(op, req_id, request),
                ), op))

            batch_ids = await vector_store.upsert_batch(
                [node for node, _op in add_nodes]
            )
            accepted = set(batch_ids or [])
            for node, op in add_nodes:
                if node.node_id not in accepted:
                    logger.warning(
                        "[reconciler] ADD batch did not return node id %s",
                        node.node_id,
                    )
                    continue
                stored_ids.append(node.node_id)
                graph_payload.append((node, op))
                add_cnt += 1
                await self._emit_reconcile_apply(
                    request, vector_store=vector_store, op_type="ADD",
                    memory_id=node.node_id, layer=node.layer.value,
                    content=node.content,
                )
                await self._maybe_index_entities(vector_store, node, request)
                tags_to_index.update(t for t in (node.tags or []) if t)
                if self._cache:
                    try:
                        await self._cache.store_memory_operation(
                            request_id=req_id,
                            user_id=request.user_id,
                            agent_id=request.agent_id or "default_agent",
                            op="ADD",
                            memory_id=node.node_id,
                            content=node.content,
                            layer=node.layer.value,
                            reason=op.reason,
                            supersedes=[],
                        )
                    except Exception as e:
                        logger.warning(
                            "[reconciler] store_memory_operation failed: %s", e,
                        )
            logger.info(
                "TRACE_PERF [%s] S1_ADD_BATCH_DONE nodes=%d",
                req_id, len(stored_ids),
            )
            ops_to_apply = []

        for op_idx, op in ops_to_apply:
            # ------------------------------------------------
            # SUPERSEDE / UPDATE: mark the old node → create the new node
            # SUPERSEDE: old node status=SUPERSEDED (joins the evolution chain, still recallable and expandable),
            #            new node supersedes=[old_id]
            # UPDATE: old node status=SHADOW (logical delete, not recalled),
            #         new node supersedes=None (not in a chain)
            # ------------------------------------------------
            if op.op in ("SUPERSEDE", "UPDATE"):
                target_id = op.memory_id
                if not target_id:
                    logger.warning(f"[reconciler] {op.op} op missing memory_id, skipped")
                    continue

                # UPDATE with no content = legacy DELETE mapping (shadow-only)
                if op.op == "UPDATE" and not op.content:
                    try:
                        await vector_store.update_payload(
                            target_id,
                            {
                                "is_latest": False,
                                "status": MemoryStatus.SHADOW.value,
                            },
                        )
                        try:
                            await vector_store.delete_retrieval_clauses(target_id)
                        except Exception as clause_error:
                            logger.debug(
                                "[retrieval-clause] shadow cleanup failed for %s: %s",
                                target_id, clause_error,
                            )
                        logger.debug(f"[reconciler] shadow-only UPDATE: memory_id={target_id}")
                    except Exception as e:
                        logger.warning(f"[reconciler] failed to shadow node {target_id}: {e}")
                    continue

                content = op.content or ""
                layer = MemoryLayer.from_string(op.layer) if op.layer else MemoryLayer.L2_FACT

                # SUPERSEDE → the old node joins the evolution chain with status=SUPERSEDED (still recallable,
                #   and the whole chain expands both ways after a hit); new node supersedes=[old_id].
                # UPDATE → the old node stays out of the chain with status=SHADOW (a logical delete, not recalled);
                #   new node supersedes=None and stands alone.
                is_supersede = op.op == "SUPERSEDE"
                old_status = (
                    MemoryStatus.SUPERSEDED.value if is_supersede
                    else MemoryStatus.SHADOW.value
                )

                # ------------------------------------------------
                # Multi-node chain fold (SUPERSEDE only):
                # op.memory_ids = [E0, E1, ...] (ordered, old → new). These previously
                # unchained old nodes are first linked into one chain (E0 ← E1 ← ...), then the new head
                # node supersedes the newest one (target_id = memory_ids[-1]).
                # Every old node is marked SUPERSEDED (still recallable and expandable); only the head (the new node)
                # keeps is_latest=True.
                # ------------------------------------------------
                chain_ids = list(getattr(op, "memory_ids", None) or [])
                if is_supersede and len(chain_ids) > 1:
                    for _prev_id, _cur_id in zip(chain_ids, chain_ids[1:]):
                        # _cur_id supersedes _prev_id: build the supersedes / superseded_by two-way chain
                        try:
                            _cur_node = await vector_store.get_by_id(_cur_id)
                            if _cur_node is not None:
                                _sup = list(_cur_node.supersedes or [])
                                if _prev_id not in _sup:
                                    _sup.append(_prev_id)
                                await vector_store.update_payload(
                                    _cur_id, {"supersedes": _sup}
                                )
                        except Exception as e:
                            logger.warning(
                                f"[reconciler] chain-link supersedes on {_cur_id} failed: {e}"
                            )
                        try:
                            _prev_node = await vector_store.get_by_id(_prev_id)
                            _sb = list(_prev_node.superseded_by or []) if _prev_node else []
                            if _cur_id not in _sb:
                                _sb.append(_cur_id)
                            await vector_store.update_payload(
                                _prev_id,
                                {
                                    "superseded_by": _sb,
                                    "is_latest": False,
                                    "status": MemoryStatus.SUPERSEDED.value,
                                },
                            )
                        except Exception as e:
                            logger.warning(
                                f"[reconciler] chain-link superseded_by on {_prev_id} failed: {e}"
                            )
                    logger.debug(
                        f"[reconciler] SUPERSEDE chain fold: {chain_ids} "
                        f"(head target={target_id})"
                    )

                # Mark the newest target_id on the chain (the new head replaces it directly):
                # SUPERSEDE → SUPERSEDED (in the chain, recallable); UPDATE → SHADOW (logical delete)
                # ------------------------------------------------
                # UPDATE (has content, not SUPERSEDE): update the old node in place
                # ------------------------------------------------
                # UPDATE means "merge and refine the same topic"; it is essentially the continuation of one memory, so
                # target_id is updated in place: content + embedding + tags are replaced while the original
                # logical memory_at is kept. UPDATE is a refinement of the same durable memory or an aggregation of several events,
                # so the time of the whole history must not be overwritten with this write's time.
                # This naturally keeps access_count / last_accessed_at (no reset to cold start)
                # and produces no SHADOW garbage nodes. SUPERSEDE still uses the shadow-plus-new-chain path below.
                if not is_supersede:
                    new_emb = embeddings_map.get(op_idx) or await self.embed_service.embed_queued(content)
                    _old = None
                    _old_lookup_ok = False
                    try:
                        _old = await vector_store.get_by_id(target_id)
                        _old_lookup_ok = True
                    except Exception as e:
                        logger.warning(
                            f"[reconciler] read UPDATE target {target_id} failed; "
                            f"preserving stored memory_at implicitly: {e}"
                        )
                    _updates = {
                        "content": content,
                        "source": request.source,
                        "embedding": new_emb,
                        "tags": list(op.tags or []),
                        "tag_ners": list(getattr(op, "tag_ners", None) or []),
                        "layer": layer.value,
                        "gmt_modified": int(datetime.now().timestamp()),
                        "is_latest": True,
                        "status": MemoryStatus.ACTIVE.value,
                    }
                    if (
                        _old_lookup_ok
                        and _old is not None
                        and _old.memory_at is None
                        and request.memory_at is not None
                    ):
                        _updates["memory_at"] = int(request.memory_at.timestamp())
                    if getattr(op, "owner", None):
                        _updates["owner"] = _norm_owner(op.owner)
                    if source_raw_memory_id:
                        _updates["source_raw_memory_id"] = source_raw_memory_id
                    if session_episode_id:
                        _updates["episode_id"] = (
                            getattr(_old, "episode_id", "") or session_episode_id
                        )
                    try:
                        _merged_sources = self._merge_source_raw_ids(
                            _old, source_raw_memory_id,
                        )
                        if getattr(
                            getattr(self.config, "llm", None),
                            "structure_metadata_enabled", False,
                        ):
                            _merged_sources = merge_structural_custom(
                                _merged_sources,
                                topic_paths=getattr(op, "topic_paths", None),
                                event_facet=getattr(op, "event_facet", None),
                                topic_assignments=getattr(op, "topic_assignments", None),
                                topic_paths_provided=bool(
                                    getattr(op, "topic_paths_provided", False)
                                    or getattr(op, "topic_paths", None)
                                ),
                                event_facet_provided=bool(
                                    getattr(op, "event_facet_provided", False)
                                    or getattr(op, "event_facet", None)
                                ),
                                topic_assignments_provided=bool(
                                    getattr(op, "topic_assignments_provided", False)
                                    or getattr(op, "topic_assignments", None)
                                ),
                            )
                        if _merged_sources:
                            _updates["custom"] = _merged_sources
                    except Exception as e:
                        logger.warning(
                            f"[reconciler] merge source provenance on {target_id} failed: {e}"
                        )
                    # Merge semantics: UPDATE rewrites the same node in place, so this call's meta must be appended to the old
                    # meta list. custom is replaced wholesale, so the node is read back before writing it.
                    # The extra read happens only when this call carries meta; otherwise the original single write is kept.
                    if getattr(request, "client_meta", None):
                        try:
                            _merged = self._merge_client_meta(
                                getattr(_old, "custom", None), request,
                            )
                            if _merged is None:
                                _merged = dict(_updates.get("custom") or getattr(_old, "custom", None) or {})
                            elif _updates.get("custom"):
                                # `_merged` owns the newly appended client meta;
                                # `_updates.custom` owns fresh provenance and
                                # structural fields. Preserve both, with the new
                                # structure/source values taking precedence.
                                _incoming = dict(_updates["custom"])
                                _incoming.pop("meta", None)
                                _merged.update(_incoming)
                            if session_episode_id:
                                _merged = add_parent_episode_membership(
                                    _merged, session_episode_id,
                                    similarity=1.0, method="write_session",
                                )
                            if _merged is not None:
                                _updates["custom"] = _merged
                        except Exception as e:
                            # meta is extra payload; a read-back failure must not block the memory update itself
                            logger.warning(
                                f"[reconciler] merge client meta on {target_id} failed: {e}"
                            )
                    elif session_episode_id:
                        _updates["custom"] = add_parent_episode_membership(
                            _updates.get("custom") or getattr(_old, "custom", None),
                            session_episode_id,
                            similarity=1.0, method="write_session",
                        )
                    try:
                        await vector_store.update_payload(target_id, _updates)
                    except Exception as e:
                        logger.warning(f"[reconciler] in-place UPDATE failed on {target_id}: {e}")
                        continue
                    stored_ids.append(target_id)
                    graph_payload.append((
                        self._graph_stub_node(
                            target_id, content, layer, list(op.tags or []), request,
                            memory_at=(
                                _old.memory_at
                                if _old is not None and _old.memory_at is not None
                                else request.memory_at if _old_lookup_ok else None
                            ),
                            custom=_updates.get("custom") or getattr(_old, "custom", None),
                        ),
                        op,
                    ))
                    update_cnt += 1
                    await self._emit_reconcile_apply(
                        request, vector_store=vector_store, op_type="UPDATE",
                        memory_id=target_id, layer=layer.value, content=content,
                    )
                    logger.debug(
                        f"[reconciler] UPDATE(in-place): {content[:80]} → node_id={target_id} "
                        f"layer={layer.value} reason={op.reason}"
                    )
                    tags_to_index.update(t for t in (op.tags or []) if t)
                    # Write the memory_operations record
                    if self._cache:
                        try:
                            await self._cache.store_memory_operation(
                                request_id=req_id,
                                user_id=request.user_id,
                                agent_id=request.agent_id or "default_agent",
                                op=op.op,
                                memory_id=target_id,
                                content=content,
                                layer=layer.value,
                                reason=op.reason,
                            )
                        except Exception as e:
                            logger.warning(f"[reconciler] store_memory_operation(UPDATE) failed: {e}")
                    continue

                # ------------------------------------------------
                # SUPERSEDE: shadow (SUPERSEDED) the old node + create the new chain head
                # ------------------------------------------------
                try:
                    await vector_store.update_payload(
                        target_id,
                        {
                            "is_latest": False,
                            "status": old_status,
                        },
                    )
                    try:
                        await vector_store.delete_retrieval_clauses(target_id)
                    except Exception as clause_error:
                        logger.debug(
                            "[retrieval-clause] supersede cleanup failed for %s: %s",
                            target_id, clause_error,
                        )
                except Exception as e:
                    logger.warning(f"[reconciler] failed to mark old node {target_id} ({op.op}): {e}")
                    continue

                # Create the new head node (only SUPERSEDE can reach here; UPDATE was updated in place above and continued)
                new_node = MemoryNode(
                    user_id=request.user_id,
                    agent_id=request.agent_id or "default_agent",
                    session_id=request.session_id or "default_session",
                    source=request.source,
                    layer=layer,
                    content=content,
                    episode_id=episode_id,
                    owner=getattr(op, "owner", None),
                    speculate=op.speculate,
                    supersedes=[target_id],
                    is_latest=True,
                    source_type=SourceType.INFERRED,
                    status=MemoryStatus.ACTIVE,
                    embedding=embeddings_map.get(op_idx) or await self.embed_service.embed_queued(content),
                    memory_at=request.memory_at,
                    source_raw_memory_id=source_raw_memory_id or None,
                    tags=list(op.tags or []),
                    tag_ners=list(getattr(op, "tag_ners", None) or []),
                    # SUPERSEDE creates a new node carrying only this call's meta: meta means "the context when this
                    # node was written", and the superseded old node stays in the store with its meta untouched,
                    # queryable along the evolution chain, so nothing is lost. Inheriting the whole chain's meta would accumulate on every supersede,
                    # which would easily hit the 5KB cap on long chains.
                    custom=self._fact_custom(op, req_id, request),
                )
                nid = await vector_store.upsert(new_node)
                stored_ids.append(nid)
                graph_payload.append((new_node, op))
                if target_id:
                    superseded_old_ids.append(target_id)
                await self._emit_reconcile_apply(
                    request, vector_store=vector_store, op_type="SUPERSEDE",
                    memory_id=nid, layer=layer.value, content=content,
                    supersedes=[target_id],
                )
                await self._maybe_index_entities(vector_store, new_node, request)

                supersede_cnt += 1

                logger.debug(
                    f"[reconciler] SUPERSEDE: {content[:80]} → node_id={nid} "
                    f"layer={layer.value} supersedes=[{target_id}] reason={op.reason}"
                )

                # SUPERSEDE: write superseded_by back onto the old node (establishing the chain link)
                try:
                    old_node = await vector_store.get_by_id(target_id)
                    existing_superseded_by = (old_node.superseded_by or []) if old_node else []
                    if nid not in existing_superseded_by:
                        existing_superseded_by.append(nid)
                    await vector_store.update_payload(
                        target_id,
                        {"superseded_by": existing_superseded_by},
                    )
                except Exception as e:
                    logger.warning(f"[reconciler] failed to update superseded_by on {target_id}: {e}")

                tags_to_index.update(t for t in (new_node.tags or []) if t)

                # Write the memory_operations record
                if self._cache:
                    try:
                        await self._cache.store_memory_operation(
                            request_id=req_id,
                            user_id=request.user_id,
                            agent_id=request.agent_id or "default_agent",
                            op=op.op,
                            memory_id=nid,
                            content=content,
                            layer=layer.value,
                            reason=op.supersede_reason or op.reason,
                            supersedes=[target_id] if is_supersede else [],
                        )
                    except Exception as e:
                        logger.warning(f"[reconciler] store_memory_operation({op.op}) failed: {e}")
                continue

            # ------------------------------------------------
            # ADD op: purely new node
            # ------------------------------------------------
            if op.op != "ADD":
                logger.warning(f"[reconciler] unknown op '{op.op}', skipped")
                continue

            content = op.content or ""
            layer = MemoryLayer.from_string(op.layer) if op.layer else MemoryLayer.L2_FACT

            new_node = MemoryNode(
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                source=request.source,
                layer=layer,
                content=content,
                episode_id=episode_id,
                owner=getattr(op, "owner", None),
                speculate=op.speculate,
                supersedes=None,
                is_latest=True,
                source_type=SourceType.INFERRED,
                status=MemoryStatus.ACTIVE,
                embedding=embeddings_map.get(op_idx) or await self.embed_service.embed_queued(content),
                memory_at=request.memory_at,
                source_raw_memory_id=source_raw_memory_id or None,
                tags=list(op.tags or []),
                tag_ners=list(getattr(op, "tag_ners", None) or []),
                custom=self._fact_custom(op, req_id, request),
            )
            nid = await vector_store.upsert(new_node)
            stored_ids.append(nid)
            graph_payload.append((new_node, op))
            add_cnt += 1
            await self._emit_reconcile_apply(
                request, vector_store=vector_store, op_type="ADD",
                memory_id=nid, layer=layer.value, content=content,
            )
            await self._maybe_index_entities(vector_store, new_node, request)
            logger.debug(
                f"[reconciler] ADD: {content[:80]} → node_id={nid} "
                f"layer={layer.value} reason={op.reason}"
            )

            tags_to_index.update(t for t in (new_node.tags or []) if t)

            if self._cache:
                try:
                    await self._cache.store_memory_operation(
                        request_id=req_id,
                        user_id=request.user_id,
                        agent_id=request.agent_id or "default_agent",
                        op="ADD",
                        memory_id=nid,
                        content=content,
                        layer=layer.value,
                        reason=op.reason,
                        supersedes=[],
                    )
                except Exception as e:
                    logger.warning(f"[reconciler] store_memory_operation failed: {e}")

        # Hidden retrieval entry points for long aggregated L2 memories. Aliases live only in the auxiliary index and return the parent L2 on a hit.
        if graph_payload:
            try:
                await sync_retrieval_clause_index(
                    vector_store=vector_store,
                    embed_service=self.embed_service,
                    nodes=[node for node, _op in graph_payload],
                    enabled=bool(getattr(
                        self.config.recall, "retrieval_clauses_enabled", False
                    )),
                    min_words=int(getattr(
                        self.config.recall, "retrieval_clauses_min_words", 70
                    )),
                    clear_empty=False,
                )
                for node, op in graph_payload:
                    if (
                        getattr(op, "op", "") == "UPDATE"
                        and not eligible_retrieval_clauses(
                            node, min_words=int(getattr(
                            self.config.recall, "retrieval_clauses_min_words", 70
                        )))
                    ):
                        await vector_store.delete_retrieval_clauses(node.node_id)
            except Exception as e:
                logger.warning(
                    "[retrieval-clause] maintain failed (non-fatal): %s", e
                )

        # Merge tags across all ops of one write so each memory does not wait for its own tag embedding round.
        if tags_to_index:
            try:
                await _tag_index_helper.ensure_tag_embeddings_for_node(
                    vector_store=vector_store,
                    embed_service=self.embed_service,
                    user_id=request.user_id,
                    tags=sorted(tags_to_index),
                )
            except Exception as e:
                logger.debug(f"[tag-index] maintain failed (non-fatal): {e}")
            self._remember_existing_tags(
                request.user_id, request.agent_id, tags_to_index,
            )

        # WRITE_SUMMARY log
        if self._cache and recon_result.ops:
            try:
                import json as _json2
                total_ops = len(recon_result.ops)
                summary_data = {
                    "add_count": add_cnt,
                    "supersede_count": supersede_cnt,
                    "update_count": update_cnt,
                    "total_ops": total_ops,
                    "new_memories_input": len(new_memory_texts),
                }
                await self._cache.store_pipeline_log(
                    request_id=req_id,
                    user_id=request.user_id,
                    agent_id=request.agent_id or "default_agent",
                    session_id=request.session_id or "",
                    step="WRITE_SUMMARY",
                    prompt="",
                    response=_json2.dumps(summary_data, ensure_ascii=False),
                    parsed=_json2.dumps(summary_data, ensure_ascii=False),
                    memory_ids=stored_ids,
                )
            except Exception as e:
                logger.warning(f"[write] store WRITE_SUMMARY failed: {e}")

        # ── Graph projection (pro mode, only takes effect with enable_graph) ──
        # Asynchronous fire-and-forget: the graph is an extra signal and must not enter the write critical path.
        self._schedule_graph_projection(
            graph_payload=graph_payload,
            superseded_old_ids=superseded_old_ids,
            candidate_scores=recon_result.candidate_scores,
            request=request,
        )

        ops_counts = {
            "add": add_cnt,
            "supersede": supersede_cnt,
            "update": update_cnt,
            "total": add_cnt + supersede_cnt + update_cnt,
        }
        recon_tokens = {
            "prompt": recon_result.prompt_tokens,
            "completion": recon_result.completion_tokens,
            "total": recon_result.total_tokens,
        }
        return stored_ids, None, ops_counts, recon_tokens

    async def _direct_store(
        self,
        new_memories_meta: List[Dict],
        request: WriteRequest,
        vector_store: VectorStoreBase,
        req_id: str,
        source_raw_memory_id: str = "",
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[List[str], Optional[str], Dict[str, int], Dict[str, int]]:
        """
        Skip reconcile and insert the extractor's memories straight into the VDB.

        Activated by RECONCILE_ENABLED=false. Intended for eval scenarios:
        no dedup, merge or evolution; every extraction result is kept.
        """
        stored_ids: List[str] = []
        session_episode_id = self._session_episode_id(request)

        # Batch embed
        contents = [m.get("content", "") for m in new_memories_meta if m.get("content")]
        precomputed_embeddings = precomputed_embeddings or {}
        missing_contents = [c for c in contents if c not in precomputed_embeddings]
        computed_embeddings = (
            await self.embed_service.embed_batch(missing_contents)
            if missing_contents else []
        )
        embedding_by_content = dict(precomputed_embeddings)
        embedding_by_content.update(zip(missing_contents, computed_embeddings))

        stored_nodes: List[MemoryNode] = []
        for meta in new_memories_meta:
            content = meta.get("content", "")
            if not content:
                continue

            layer = MemoryLayer.from_string(meta.get("layer", "L2_FACT"))
            tags = meta.get("tags") or []
            speculate = meta.get("speculate")

            new_node = MemoryNode(
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "default_session",
                source=request.source,
                episode_id=session_episode_id,
                layer=layer,
                content=content,
                owner=_norm_owner(meta.get("owner")),
                speculate=speculate,
                is_latest=True,
                source_type=SourceType.INFERRED,
                status=MemoryStatus.ACTIVE,
                embedding=(
                    embedding_by_content.get(content)
                    or await self.embed_service.embed_queued(content)
                ),
                memory_at=request.memory_at,
                source_raw_memory_id=source_raw_memory_id or None,
                tags=list(tags),
                custom={
                    **self._fact_custom(None, req_id, request),
                    **(
                        structural_custom(
                            meta.get("topic_paths"), meta.get("event_facet"),
                            meta.get("topic_assignments"),
                        )
                        if getattr(
                            getattr(self.config, "llm", None),
                            "structure_metadata_enabled", False,
                        ) else {}
                    ),
                },
            )
            nid = await vector_store.upsert(new_node)
            stored_ids.append(nid)
            stored_nodes.append(new_node)
            await self._emit_reconcile_apply(
                request, vector_store=vector_store, op_type="ADD",
                memory_id=nid, layer=layer.value, content=content,
            )
            await self._maybe_index_entities(vector_store, new_node, request)

        if stored_nodes:
            try:
                await sync_retrieval_clause_index(
                    vector_store=vector_store,
                    embed_service=self.embed_service,
                    nodes=stored_nodes,
                    enabled=bool(getattr(
                        self.config.recall, "retrieval_clauses_enabled", False
                    )),
                    min_words=int(getattr(
                        self.config.recall, "retrieval_clauses_min_words", 70
                    )),
                    clear_empty=False,
                )
            except Exception as e:
                logger.warning(
                    "[retrieval-clause] direct-store maintain failed (non-fatal): %s", e
                )
            self._remember_existing_tags(
                request.user_id,
                request.agent_id,
                {tag for node in stored_nodes for tag in (node.tags or []) if tag},
            )

        logger.info(f"[direct_store] stored {len(stored_ids)} nodes (reconcile disabled)")
        n = len(stored_ids)
        return stored_ids, None, {"add": n, "supersede": 0, "update": 0, "total": n}, {"prompt": 0, "completion": 0, "total": 0}

    async def _emit_pipeline_step(
        self,
        request: WriteRequest,
        *,
        step: str,
        parsed: Union[str, Dict[str, Any], List[Any]],
        elapsed_ms: float = 0,
        response: str = "",
        prompt: str = "",
        memory_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Stage-level log/trace (via the client hook: file always, DB optional)."""
        if not self._cache:
            return
        parsed_str = (
            parsed
            if isinstance(parsed, str)
            else json.dumps(parsed, ensure_ascii=False, default=str)
        )
        try:
            await self._cache.store_pipeline_log(
                request_id=request.request_id or get_request_id(),
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step=step,
                prompt=prompt,
                response=response,
                parsed=parsed_str,
                memory_ids=memory_ids,
                elapsed_ms=elapsed_ms,
                **kwargs,
            )
        except Exception as e:
            logger.debug(f"[write] pipeline step {step} failed: {e}")

    async def _emit_reconcile_apply(
        self,
        request: WriteRequest,
        *,
        vector_store,
        op_type: str,
        memory_id: str,
        layer: str,
        content: str,
        supersedes: Optional[List[str]] = None,
    ) -> None:
        """Reconcile apply pipeline log (RECONCILE_APPLY).

        Immediately after upsert/update_payload the node is read back to confirm it really persisted, and the result is written into
        the pipeline trace. Used to pinpoint "reconcile decided ADD and the ledger recorded it, but the VDB has no such record",
        distinguishing "never written" from "written but not persisted" and "deleted after writing".
        """
        # This read-after-write check is diagnostic, not part of the persistence
        # contract. Running it unconditionally made a normal multi-memory write
        # pay one extra VDB round trip per extracted memory even when trace storage
        # was explicitly disabled. Only perform the stronger read-back audit when
        # the operator has enabled pipeline traces.
        verify_enabled = _write_apply_verify_enabled()
        verified = None
        verify_error = ""
        if verify_enabled:
            try:
                _node = await vector_store.get_by_id(memory_id)
                verified = _node is not None
                if not verified:
                    verify_error = "get_by_id returned None (upsert returned id but not readable)"
            except Exception as e:
                verified = False
                verify_error = f"{type(e).__name__}: {e}"
        parsed = {
            "op": op_type,
            "memory_id": memory_id,
            "layer": layer,
            "content": (content or "")[:200],
            "supersedes": list(supersedes or []),
            "verified": verified,
            "verification_skipped": not verify_enabled,
        }
        if verify_error:
            parsed["verify_error"] = verify_error
        if verify_enabled and not verified:
            logger.warning(
                f"[reconciler] RECONCILE_APPLY verify FAILED op={op_type} "
                f"mid={memory_id} layer={layer}: {verify_error}"
            )
        await self._emit_pipeline_step(
            request,
            step="RECONCILE_APPLY",
            parsed=parsed,
            memory_ids=[memory_id],
        )

    async def _emit_intention_upsert(
        self,
        request: WriteRequest,
        *,
        vector_store,
        memory_id: str,
        content: str,
        tags: Optional[List[str]] = None,
        valid_until=None,
    ) -> None:
        """Intention (L7) upsert pipeline log (INTENTION_UPSERT).

        Intentions are upserted straight into the VDB by System 1 without reconcile, so they are not part of RECONCILE_APPLY.
        This is instrumented separately: the node is read back after upsert to confirm persistence, and memory_id is included so production can trace by
        request_id / memory_id (distinguishing "never written", "written but not persisted" and "deleted after writing").
        """
        verify_enabled = _write_apply_verify_enabled()
        verified = None
        verify_error = ""
        if verify_enabled:
            try:
                _node = await vector_store.get_by_id(memory_id)
                verified = _node is not None
                if not verified:
                    verify_error = "get_by_id returned None (upsert returned id but not readable)"
            except Exception as e:
                verified = False
                verify_error = f"{type(e).__name__}: {e}"
        parsed = {
            "op": "INTENTION_UPSERT",
            "memory_id": memory_id,
            "layer": MemoryLayer.L7_INTENTION.value,
            "content": (content or "")[:200],
            "tags": list(tags or []),
            "valid_until": valid_until.isoformat() if hasattr(valid_until, "isoformat") else valid_until,
            "verified": verified,
            "verification_skipped": not verify_enabled,
        }
        if verify_error:
            parsed["verify_error"] = verify_error
        if verify_enabled and not verified:
            logger.warning(
                f"[intention] INTENTION_UPSERT verify FAILED mid={memory_id}: {verify_error}"
            )
        await self._emit_pipeline_step(
            request,
            step="INTENTION_UPSERT",
            parsed=parsed,
            memory_ids=[memory_id],
        )

    async def _emit_write_timeline(
        self,
        request: WriteRequest,
        tracer: PipelineTracer,
    ) -> None:
        """Write the request-level timeline into the Trace store (SQLite) for the Inspector."""
        if not self._cache or not is_pipeline_trace_enabled():
            return
        try:
            await self._cache.store_pipeline_log(
                request_id=request.request_id or get_request_id() or tracer.request_id,
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step="WRITE_TIMELINE",
                prompt="",
                response=tracer.to_summary_line(),
                parsed=json.dumps(tracer.to_dict(), ensure_ascii=False, default=str),
                elapsed_ms=0,
            )
        except Exception as e:
            logger.debug(f"[write] WRITE_TIMELINE failed: {e}")

    # ================================================================
    # Direct write of already-extracted facts (skips extract, goes straight to reconcile)
    # ================================================================

    async def write_extracted(
        self,
        request: WriteRequest,
        ctx: Optional[PipelineContext] = None,
        tracer: Optional[PipelineTracer] = None,
    ) -> WriteResponse:
        """Write memories the agent has already extracted: skip extract and go straight to reconcile.

        For callers (such as the openclaw memory_add tool) that already abstract a conversation into one or more
        conclusive memory contents. It skips LLM extract (saving one LLM call)
        but still runs the full reconcile: dedup against existing memories / merge / contradiction evolution, then persists as L2_FACT.

        Differences from write():
        - No L1_RAW source node is written (the content is itself the fact; there is no raw conversation to archive).
        - No extractor call (mem_agent.process_add).
        - The content list is fed directly to _reconcile_and_store (or to _direct_store when
          RECONCILE_ENABLED=false).

        Input contract: request.extra["extracted_contents"] = List[str] (one fact per entry).
        When empty it falls back to request.content (a single entry). The layer is always L2_FACT.
        """
        start_time = datetime.now()
        response = WriteResponse()

        if not request.request_id:
            request.request_id = get_request_id()

        if tracer is None:
            tracer = create_tracer(
                operation="write_extracted",
                pipeline_version="system1",
                uid=request.user_id,
                agent_id=request.agent_id,
                request_id=request.request_id,
                content_preview=request.content,
            )

        try:
            # Collect the extracted facts: prefer extra["extracted_contents"], otherwise fall back to content
            raw_contents = request.extra.get("extracted_contents")
            if not raw_contents:
                raw_contents = [request.content] if request.content else []
            contents = [c.strip() for c in raw_contents if isinstance(c, str) and c.strip()]

            if not contents:
                response.error_code = 400
                response.error_message = "extracted_contents or content is required"
                return response
            if not request.user_id:
                response.error_code = 400
                response.error_message = "user_id is required"
                return response

            if request.memory_at is None:
                request.memory_at = start_time

            _req_id = request.request_id or get_request_id()
            vector_store = await self._get_vector_store()

            # Build the reconcile input directly (extract skipped). The layer is always L2_FACT.
            new_memory_texts: List[str] = list(contents)
            new_memories_meta: List[Dict] = [
                {"content": c, "layer": "L2_FACT", "tags": []} for c in contents
            ]

            # Dedup the already-extracted entries against each other; the vectors computed here are passed on to reconcile / store.
            precomputed_embeddings: Dict[str, List[float]] = {}
            if len(new_memory_texts) >= 2:
                try:
                    (
                        new_memory_texts,
                        new_memories_meta,
                        precomputed_embeddings,
                    ) = await self._dedup_extracted(
                        new_memory_texts, new_memories_meta, request, _req_id,
                    )
                except Exception as e:
                    logger.warning(f"[write_extracted] dedup failed (non-fatal): {e}")

            _t_recon = datetime.now()
            if _RECONCILE_ENABLED:
                stored_ids, recon_error, ops_counts, recon_tokens = await self._reconcile_and_store(
                    new_memory_texts=new_memory_texts,
                    new_memories_meta=new_memories_meta,
                    request=request,
                    vector_store=vector_store,
                    req_id=_req_id,
                    precomputed_embeddings=precomputed_embeddings,
                )
            else:
                stored_ids, recon_error, ops_counts, recon_tokens = await self._direct_store(
                    new_memories_meta=new_memories_meta,
                    request=request,
                    vector_store=vector_store,
                    req_id=_req_id,
                    precomputed_embeddings=precomputed_embeddings,
                )
            _recon_ms = (datetime.now() - _t_recon).total_seconds() * 1000

            response.extra["reconcile_ops"] = ops_counts
            response.extra["reconcile_tokens"] = recon_tokens
            response.extra["extract_tokens"] = {"prompt": 0, "completion": 0, "total": 0}

            if recon_error is not None:
                response.success = False
                response.error_code = 502
                response.error_message = f"[RECONCILE_FAILED] {recon_error}"
                response.extra["agent_status"] = "failed"
                response.extra["agent_error"] = recon_error
                response.extra["agent_nodes"] = 0
                return response

            response.success = True
            response.layer = MemoryLayer.L2_FACT.value
            response.memory_id = stored_ids[0] if stored_ids else ""
            response.extra["agent_stored_ids"] = stored_ids
            response.extra["agent_status"] = "success"
            response.extra["agent_error"] = ""
            response.extra["agent_nodes"] = len(stored_ids)
            response.extra["agent_mode"] = "extracted"
            response.elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                f"[write_extracted] done: {len(stored_ids)} nodes stored from "
                f"{len(contents)} extracted contents, recon_ms={_recon_ms:.0f}"
            )
            return response

        except Exception as e:
            logger.error(f"[write_extracted] failed: {e}", exc_info=True)
            response.success = False
            response.error_code = 500
            response.error_message = str(e)
            return response
        finally:
            await self._await_graph_projection(request)
            response.extra["workload"] = workload_snapshot()
            self._attach_graph_executor_diagnostics(response)
            try:
                await self._emit_write_timeline(request, tracer)
            except Exception:
                pass

    # ================================================================
    # Main write flow
    # ================================================================

    async def write(
        self,
        request: WriteRequest,
        ctx: Optional[PipelineContext] = None,
        tracer: Optional[PipelineTracer] = None,
    ) -> WriteResponse:
        """Execute the Lite write flow."""
        start_time = datetime.now()
        response = WriteResponse()

        # request_id normalization: prefer the request.request_id explicitly passed by the client (contextvar-immune),
        # and fall back to the contextvar (protecting tests / other callers that call write() directly).
        # From here on the whole pipeline reads request.request_id for persistence and no longer relies on the contextvar.
        if not request.request_id:
            request.request_id = get_request_id()

        if tracer is None:
            tracer = create_tracer(
                operation="write",
                pipeline_version="system1",
                uid=request.user_id,
                agent_id=request.agent_id,
                request_id=request.request_id,
                content_preview=request.content,
            )

        try:
            # Validate parameters
            content = request.content
            if not content and request.has_messages():
                content = request.get_flat_content()
                request.content = content

            if not content:
                response.error_code = 400
                response.error_message = "content or messages is required"
                return response
            if not request.user_id:
                response.error_code = 400
                response.error_message = "user_id is required"
                return response

            # ── Ensure memory_at has a value: when absent, use the time the request was received ──
            if request.memory_at is None:
                request.memory_at = start_time

            total_tokens = 0

            # 1. Layer assignment
            with tracer.span("layer_assign") as s:
                layer_str = request.extra.get("layer", "")
                if layer_str:
                    suggested_layer = layer_str
                    s.set_output({"layer": suggested_layer, "source": "explicit"})
                else:
                    suggested_layer = MemoryLayer.L1_RAW.value
                    s.set_output({"layer": suggested_layer, "source": "default_raw"})

            response.layer = suggested_layer

            # ── Timing: sys1_waiting_ms ──
            # start_time is when write() was entered; real I/O starts now
            _t_process_start = datetime.now()
            _sys1_waiting_ms = (_t_process_start - start_time).total_seconds() * 1000
            _req_id_perf = request.request_id or get_request_id()
            logger.info(f"TRACE_PERF [{_req_id_perf}] S1_START waiting={_sys1_waiting_ms:.0f}ms user={request.user_id}")

            # Metrics: S1 start
            from ..metrics import MetricsCollector
            MetricsCollector.get().sys1_start()

            # 2. Vectorize + persist (L1_RAW)
            _t_embed = datetime.now()
            with tracer.span("embed") as s:
                embedding, embed_meta = await self.embed_service.embed_queued_with_meta(
                    request.content
                )
                s.set_output({"dims": len(embedding), "content_len": len(request.content)})
            _embed_ms = (datetime.now() - _t_embed).total_seconds() * 1000
            response.extra["embedding"] = embedding
            await self._emit_pipeline_step(
                request,
                step="S1_EMBED",
                parsed={
                    "dims": len(embedding),
                    "content_len": len(request.content),
                    "provider_meta": embed_meta,
                },
                elapsed_ms=_embed_ms,
            )

            # 3. Persist to the vector database (L1_RAW)
            _t_l1 = datetime.now()
            memory_id = ""
            _l1_error: Optional[str] = None
            vector_store = None
            mem_node = None
            with tracer.span("qdrant_upsert") as s:
                try:
                    vector_store = await self._get_vector_store()
                    layer_enum = MemoryLayer.from_string(suggested_layer)
                    _raw_custom: Dict[str, Any] = {}
                    self._attach_client_meta(_raw_custom, request)
                    mem_node = MemoryNode(
                        user_id=request.user_id,
                        agent_id=request.agent_id or "default_agent",
                        session_id=request.session_id or "default_session",
                        source=request.source,
                        layer=layer_enum,
                        content=request.content,
                        source_type=SourceType.EXPLICIT,
                        status=MemoryStatus.ACTIVE,
                        is_latest=True,
                        embedding=embedding,
                        memory_at=request.memory_at,
                        custom=_raw_custom,
                    )
                    memory_id = await vector_store.upsert(mem_node)
                    response.memory_id = memory_id
                    s.set_output({"memory_id": memory_id, "layer": suggested_layer})
                    logger.debug(f"[write] persisted: id={memory_id} layer={suggested_layer}")
                except Exception as persist_err:
                    _l1_error = str(persist_err)
                    s.set_error(_l1_error)
                    logger.error(f"V1 Write: Persist failed (non-fatal): {persist_err}", exc_info=True)
            _l1_ms = (datetime.now() - _t_l1).total_seconds() * 1000
            # Whether the backend has sparse full-text indexing enabled.
            _sparse_enabled = bool(getattr(vector_store, "supports_fulltext", False))
            await self._emit_pipeline_step(
                request,
                step="S1_L1_UPSERT",
                parsed={
                    "memory_id": memory_id or None,
                    "layer": suggested_layer,
                    "content": (mem_node.content if mem_node else request.content),
                    "tags": list(mem_node.tags) if (mem_node and mem_node.tags) else [],
                    "sparse_enabled": _sparse_enabled,
                    "error": _l1_error,
                },
                elapsed_ms=_l1_ms,
                memory_ids=[memory_id] if memory_id else None,
            )

            # ── Timing: sys1_l1_process_ms ──
            _sys1_l1_process_ms = (datetime.now() - _t_process_start).total_seconds() * 1000
            logger.info(f"TRACE_PERF [{_req_id_perf}] S1_L1_DONE l1_ms={_sys1_l1_process_ms:.0f}ms")

            # 4. MemAgent processing (optional)
            agent_mode = request.extra.get("agent_mode", "disabled")
            response.extra["agent_mode"] = agent_mode

            _t_workflow = datetime.now()
            if (agent_mode == "full" and vector_store is not None
                    and mem_node is not None and memory_id):
                # ── Fetch history context (the last k dialogue turns, for the extractor; k comes from config) ──
                _history_context = ""
                _recent_history_ms = 0.0
                _recent_history_count = 0
                try:
                    _hist_turns = getattr(self.config.extractor, "history_turns", 5) or 5
                    _t_recent_history = datetime.now()
                    _history_context = await self._get_recent_history(
                        vector_store, request.user_id, request.agent_id,
                        exclude_memory_id=memory_id,
                        max_turns=_hist_turns * 2,  # k dialogue turns = k*2 messages
                    )
                    _recent_history_ms = (_t_recent_history and (datetime.now() - _t_recent_history).total_seconds() * 1000) or 0.0
                    _recent_history_count = len([line for line in (_history_context or "").splitlines() if line.strip()])
                except Exception as _hist_err:
                    logger.warning(f"[write] get_recent_history failed: {_hist_err}")

                with tracer.span("mem_agent") as s:
                    await self._run_agent(
                        request=request,
                        response=response,
                        vector_store=vector_store,
                        mem_node=mem_node,
                        memory_id=memory_id,
                        tracer_span=s,
                        history_context=_history_context,
                        recent_history_ms=_recent_history_ms,
                        recent_history_count=_recent_history_count,
                    )
                    total_tokens += response.extra.get("_agent_tokens", 0)
                if response.extra.get("agent_status") == "failed":
                    await self._emit_pipeline_step(
                        request,
                        step="S1_AGENT_FAILED",
                        parsed={
                            "error_code": response.extra.get("agent_error_code", ""),
                            "error": response.extra.get("agent_error", ""),
                        },
                    )
                    await self._rollback_failed_l1(
                        request=request,
                        response=response,
                        vector_store=vector_store,
                        memory_id=memory_id,
                    )
                    response.tokens_used = total_tokens
                    _failed_workflow_ms = (
                        datetime.now() - _t_workflow
                    ).total_seconds() * 1000
                    response.extra["timing"] = {
                        "sys1_waiting_ms": round(_sys1_waiting_ms, 1),
                        "sys1_l1_process_ms": round(_sys1_l1_process_ms, 1),
                        "sys1_workflow_ms": round(_failed_workflow_ms, 1),
                        "sys1_ops_avg_ms": 0.0,
                    }
                    try:
                        MetricsCollector.get().sys1_end(
                            response.extra["timing"], success=False
                        )
                    except Exception:
                        pass
                    return response
                elif (
                    request.extra.get("finalize_session")
                    and (
                        getattr(self.config.recall, "session_episode_enabled", False)
                        or getattr(self.config.recall, "dynamic_episode_enabled", False)
                        or getattr(self.config.recall, "series_episode_enabled", False)
                    )
                ):
                    try:
                        from ..agent.episode_summary import (
                            finalize_dynamic_episode_capsules,
                            finalize_series_episode_containers,
                            finalize_session_episode,
                        )
                        finalize = (
                            finalize_dynamic_episode_capsules
                            if (
                                getattr(self.config.recall, "dynamic_episode_enabled", False)
                                or getattr(self.config.recall, "series_episode_enabled", False)
                            )
                            else finalize_session_episode
                        )
                        finalize_kwargs = dict(
                            vector_store=vector_store,
                            embed_service=self.embed_service,
                            llm_provider=self.llm_provider,
                            user_id=request.user_id,
                            agent_id=request.agent_id,
                            session_id=request.session_id or "default_session",
                        )
                        if finalize is finalize_dynamic_episode_capsules:
                            finalize_kwargs["summary_concurrency"] = getattr(
                                self.config.recall,
                                "dynamic_episode_summary_concurrency", 3,
                            )
                        async with episode_slot(), workload_scope("derived"):
                            response.extra["session_episode"] = await finalize(
                                **finalize_kwargs
                            )
                        if getattr(self.config.recall, "series_episode_enabled", False):
                            try:
                                await self._await_graph_projection(request)
                                response.extra["series_episode"] = (
                                    await finalize_series_episode_containers(
                                        vector_store=vector_store,
                                        graph_store=self._graph_store,
                                        user_id=request.user_id,
                                        agent_id=request.agent_id,
                                    )
                                )
                            except Exception as series_error:
                                logger.warning(
                                    "[series-episode] finalize failed: %s", series_error,
                                )
                                response.extra["series_episode"] = {
                                    "status": "failed", "error": str(series_error),
                                }
                    except Exception as episode_error:
                        logger.warning("[session-episode] finalize failed: %s", episode_error)
                        response.extra["session_episode"] = {
                            "status": "failed", "error": str(episode_error),
                        }
                        if (
                            getattr(self.config.recall, "dynamic_episode_enabled", False)
                            or getattr(self.config.recall, "series_episode_enabled", False)
                        ):
                            # L1/L2 already completed. Replaying the whole final
                            # chunk to repair a derived L3 can repeat reconcile
                            # mutations, so expose a non-retryable partial success.
                            response.success = False
                            response.error_code = 409
                            response.error_message = (
                                "[DYNAMIC_EPISODE_FINALIZE_FAILED] base memories "
                                "were persisted; automatic chunk retry is unsafe: "
                                f"{episode_error}"
                            )
                            response.extra["partial_agent_persist"] = True
                            await self._await_graph_projection(request)
                            return response
            elif agent_mode == "full":
                self._mark_agent_failure(
                    response,
                    error_code="L1_UPSERT_FAILED",
                    error_message=_l1_error or "L1 raw memory was not persisted",
                )
                await self._emit_pipeline_step(
                    request,
                    step="S1_AGENT_SKIPPED",
                    parsed={
                        "reason": "l1_upsert_failed_or_no_vector_store",
                        "error": _l1_error or "",
                    },
                )
                response.tokens_used = total_tokens
                return response

            # Graph starts after L2 persist and overlaps Dynamic Episode, but a
            # synchronous write does not return before its own projection ends.
            await self._await_graph_projection(request)

            # ── Timing: sys1_workflow_ms ──
            _sys1_workflow_ms = (datetime.now() - _t_workflow).total_seconds() * 1000
            logger.info(f"TRACE_PERF [{_req_id_perf}] S1_WORKFLOW_DONE workflow_ms={_sys1_workflow_ms:.0f}ms")

            # 5. Merge check (optional)
            enable_merge_check = request.extra.get("enable_merge_check", False)
            should_merge = response.extra.get("should_merge", False)

            if enable_merge_check and not should_merge and request.existing_memories:
                with tracer.span("merge_check") as s:
                    existing_for_merge = [
                        {"memory_id": mem.get("memory_id", ""), "content": mem.get("content", "")}
                        for mem in request.existing_memories
                    ]
                    merge_result = self.merger.check_merge(
                        new_content=request.content,
                        existing_memories=existing_for_merge,
                    )
                    response.extra["should_merge"] = merge_result.should_merge
                    response.extra["merge_target_id"] = merge_result.target_memory_id or ""
                    s.set_output({
                        "should_merge": merge_result.should_merge,
                        "target_id": merge_result.target_memory_id or "",
                    })

            response.success = True
            response.tokens_used = total_tokens

            # ── Aggregate sys1 timing ──
            _ops_count = len(response.extra.get("agent_stored_ids", []))
            _ops_avg_ms = _sys1_workflow_ms / _ops_count if _ops_count > 0 else 0
            response.extra["timing"] = {
                "sys1_waiting_ms": round(_sys1_waiting_ms, 1),
                "sys1_l1_process_ms": round(_sys1_l1_process_ms, 1),
                "sys1_workflow_ms": round(_sys1_workflow_ms, 1),
                "sys1_ops_avg_ms": round(_ops_avg_ms, 1),
            }

            # Metrics: S1 done + VDB ops
            _mc = MetricsCollector.get()
            _mc.sys1_end(response.extra["timing"], success=True)
            if _ops_count > 0:
                for _ in range(_ops_count):
                    _mc.record_vdb_op(_ops_avg_ms)

        except Exception as e:
            logger.error(f"MemoryWriter.write failed: {e}", exc_info=True)
            response.error_code = 500
            response.error_message = str(e)
            tracer.set_error(str(e))
            # Metrics: S1 failed
            try:
                MetricsCollector.get().sys1_end({}, success=False)
            except Exception:
                pass
        finally:
            await self._await_graph_projection(request)
            response.extra["workload"] = workload_snapshot()
            self._attach_graph_executor_diagnostics(response)
            response.elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            tracer.set_output({
                "success": response.success,
                "layer": response.layer,
                "memory_id": response.memory_id,
                "entities_count": len(response.entities),
                "entities": [
                    {"name": e.get("name", ""), "type": e.get("type", "")}
                    if isinstance(e, dict) else {"name": str(e)}
                    for e in response.entities
                ],
                "summary": response.extra.get("summary", ""),
                "content_stored": request.content,
                "tokens_used": response.tokens_used,
                "pipeline_ms": response.elapsed_ms,
                "agent_status": response.extra.get("agent_status", ""),
            })
            await self._emit_write_timeline(request, tracer)
            tracer.finish(write_file=False)

        return response

    async def _get_recent_history(
        self,
        vector_store: VectorStoreBase,
        user_id: str,
        agent_id: Optional[str] = None,
        exclude_memory_id: str = "",
        max_turns: int = 20,
        max_chars_assistant: int = 500,
    ) -> str:
        """
        Fetch this user's recent L1_RAW conversation records, split them by turn and assemble them into history context.

        Each L1_RAW content is a multi-turn dialogue ([user]: ...\n[assistant]: ...).
        After splitting into single messages, user messages are kept in full and assistant messages are truncated to the first max_chars_assistant characters.
        The result keeps the most recent max_turns messages (the unit is message count, not dialogue turns;
        callers pass k turns × 2).

        Returns:
            The formatted history context string; an empty string means no history.
        """
        try:
            # Only content/node_id/gmt_created are needed to build history; embeddings are not involved.
            # allow_raw_layer=True: the VDB excludes L1_RAW from every query by default (external read paths
            # never return raw conversations), and this is the one place where the internal pipeline reads raw conversations,
            # so without the override there would be no history at all and extraction quality would drop.
            nodes = await vector_store.list_by_user(
                user_id=user_id,
                agent_id=agent_id,
                layers=[MemoryLayer.L1_RAW],
                status_filter=[MemoryStatus.ACTIVE, MemoryStatus.SHADOW],
                limit=max(50, max_turns),
                with_vectors=False,
                allow_raw_layer=True,
            )
            if not nodes:
                return ""

            # Exclude the memory currently being processed (avoid self-reference)
            nodes = [n for n in nodes if n.node_id != exclude_memory_id]

            # Sort by gmt_created ascending (chronological order)
            nodes.sort(key=lambda n: n.gmt_created or datetime.min)

            # Split each L1_RAW content into single messages.
            # L1_RAW content format: "[user]: xxx\n[assistant]: yyy\n[user]: zzz..."
            # Multi-person conversations also add one line at the **very start**, "The user's name is Tim" (see WriteRequest.get_flat_content).
            # Key: a single message's content may itself contain newlines (multi-line user input),
            # so message boundaries must be detected by the "[label]: " prefix and unprefixed lines belong as continuations
            # of the previous message rather than being independent messages -- otherwise continuations become bare messages and the later
            # truncation by line count may cut through a multi-line message, losing the prefix line and leaving continuations at the start.
            #
            # Besides the 4 known roles, a label may also be the other real person's name ([John]:). A length cap keeps
            # ordinary body text such as "[Note]: xxx" from being mistaken for a message boundary.
            import re as _re
            from .base import parse_user_name_line
            _KNOWN_ROLES = {"user", "assistant", "system", "tool"}
            # Any bracketed label (including the other real person's name)
            _labeled_prefix = _re.compile(r"^\[([^\]\n]{1,32})\]:\s?")
            # Known roles without brackets (tolerant of legacy data, keeping the old regex's leniency `\[?...\]?`)
            _bare_role_prefix = _re.compile(r"^(user|assistant|system|tool):\s?")

            # List of (label, content) tuples; label is a role or the other real person's name
            parsed: List[Tuple[str, str]] = []
            user_name: str = ""
            for node in nodes:
                content = node.content or ""
                for _line_idx, line in enumerate(content.split("\n")):
                    m = _labeled_prefix.match(line) or _bare_role_prefix.match(line)
                    if m:
                        label = m.group(1)
                        body = line[m.end():]
                        parsed.append((label, body))
                        continue
                    # A name line can only be the first line of each node content (get_flat_content inserts it at the start).
                    # Decide per node rather than with `not parsed`: parsed accumulates across nodes, so from the second node
                    # on it is already non-empty and the name line would be wrongly appended as a continuation of the previous message.
                    # It is stripped here and emitted once at the start, avoiding max_turns quota use and repeated noise.
                    if _line_idx == 0:
                        _name = parse_user_name_line(line)
                        if _name:
                            user_name = _name
                            continue
                    if parsed:
                        # Continuation: append back onto the previous message (keeping the newline)
                        prev_label, prev_body = parsed[-1]
                        parsed[-1] = (prev_label, prev_body + "\n" + line)
                    else:
                        # Content starting with an unprefixed line (malformed/legacy): fall back to treating it as a user message
                        parsed.append(("user", line))

            # Take the most recent max_turns messages (truncated by message, never cut through the middle of one)
            recent = parsed[-max_turns:] if len(parsed) > max_turns else parsed
            if not recent:
                return ""

            # Formatting: everything not on the user side is truncated (including the other real person); user text is kept in full
            all_messages: List[str] = []
            if user_name:
                from .base import render_user_name_line
                from ..utils.lang_detect import is_chinese

                _name_line = render_user_name_line(
                    user_name,
                    is_chinese("\n".join(b for _, b in recent)),
                )
                if _name_line:
                    all_messages.append(_name_line)
            for label, body in recent:
                body = body.strip()
                # Truncation scope: assistant plus the other real person (any label that is not a known role).
                # user is kept in full; system / tool keep the historical behavior (no truncation).
                _truncate = label == "assistant" or label not in _KNOWN_ROLES
                if _truncate and len(body) > max_chars_assistant:
                    body = body[:max_chars_assistant] + "..."
                all_messages.append(f"[{label}]: {body}")

            return "\n".join(all_messages)
        except Exception as e:
            logger.debug(f"[write] _get_recent_history error: {e}")
            return ""

    async def _rollback_failed_l1(
        self,
        *,
        request: WriteRequest,
        response: WriteResponse,
        vector_store: Optional[VectorStoreBase],
        memory_id: str,
    ) -> None:
        """Delete this request's provisional L1_RAW after full-agent failure.

        A retryable write must not leave the same raw chunk in history. L2 changes
        are deliberately not deleted here: reconcile may have updated existing
        nodes, so treating all returned ids as newly-created would corrupt prior
        memory. The next retry/reconcile converges those partial high-level writes.
        """
        deleted = False
        rollback_error = ""
        if memory_id and vector_store is not None:
            try:
                delete_accepted = bool(await vector_store.delete(memory_id))
                if not delete_accepted:
                    rollback_error = "vector store did not confirm deletion"
                else:
                    remaining = await vector_store.get_by_id(memory_id)
                    deleted = remaining is None
                    if not deleted:
                        rollback_error = "raw memory still exists after deletion"
            except Exception as exc:
                rollback_error = f"{type(exc).__name__}: {exc}"
        elif memory_id:
            rollback_error = "vector store unavailable"

        response.extra["l1_rollback"] = {
            "memory_id": memory_id,
            "deleted": deleted,
            "error": rollback_error,
            "retry_safe": deleted,
        }
        if deleted:
            response.memory_id = ""
        partial_agent_persist = bool(response.extra.get("agent_stored_ids"))
        if partial_agent_persist:
            response.extra["partial_agent_persist"] = True
            response.extra["l1_rollback"]["retry_safe"] = False
        if rollback_error or partial_agent_persist:
            unsafe_reason = (
                f"L1 rollback failed for memory_id={memory_id}: {rollback_error}"
                if rollback_error
                else "agent partially persisted L2 changes"
            )
            if rollback_error:
                response.extra["l1_rollback_failed"] = True
            # Retrying after an unconfirmed L1 rollback or partial L2 mutation can
            # duplicate facts or reapply UPDATE/SUPERSEDE operations. Surface a
            # non-retryable conflict so recovery can inspect and reconcile first.
            response.error_code = 409
            response.error_message = (
                f"{response.error_message}; {unsafe_reason}; automatic retry is unsafe"
            ).strip("; ")
        await self._emit_pipeline_step(
            request,
            step="S1_L1_ROLLBACK",
            parsed=response.extra["l1_rollback"],
            memory_ids=[memory_id] if memory_id else None,
        )

    async def _collect_existing_tags(
        self,
        vector_store: VectorStoreBase,
        user_id: str,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Collect all tags this user already has in the VDB (deduplicated).

        Extracts the tags field from every vector_store.list_by_user result,
        merges them into a unique set, and lets the extract prompt steer the LLM to reuse existing tags first.
        On failure it silently returns an empty list (never blocking the main flow).
        """
        key = (user_id or "", agent_id or "")
        try:
            ttl = max(0.0, float(os.getenv("MEMORY_EXISTING_TAGS_CACHE_TTL_SEC", "1800")))
        except (TypeError, ValueError):
            ttl = 1800.0
        now = time.monotonic()
        cache = getattr(self, "_existing_tags_cache", None)
        if cache is None:
            cache = self._existing_tags_cache = {}
        cached = cache.get(key)
        if ttl > 0 and cached is not None and cached[0] >= now:
            return list(cached[1])

        locks = getattr(self, "_existing_tags_locks", None)
        if locks is None:
            locks = self._existing_tags_locks = {}
        lock = locks.get(key)
        if lock is None:
            lock = locks[key] = asyncio.Lock()

        async with lock:
            now = time.monotonic()
            cached = cache.get(key)
            if ttl > 0 and cached is not None and cached[0] >= now:
                return list(cached[1])
            try:
                nodes = await vector_store.list_by_user(
                    user_id=user_id,
                    agent_id=agent_id,
                    limit=10000,
                    with_vectors=False,
                )
                tag_set: set = set()
                for node in nodes:
                    if hasattr(node, "tags") and node.tags:
                        tag_set.update(node.tags)
                tags = tuple(sorted(tag_set))
                if ttl > 0:
                    cache[key] = (now + ttl, tags)
                if tags:
                    logger.debug(
                        f"[write] collected {len(tags)} existing tags for user={user_id}"
                    )
                return list(tags)
            except Exception as e:
                logger.warning(f"[write] _collect_existing_tags failed (non-fatal): {e}")
                return []

    def _remember_existing_tags(
        self,
        user_id: str,
        agent_id: Optional[str],
        tags,
    ) -> None:
        """Increment an already-warmed prompt tag cache after a successful write."""
        new_tags = {str(tag) for tag in (tags or []) if str(tag).strip()}
        if not new_tags:
            return
        cache = getattr(self, "_existing_tags_cache", None)
        if not cache:
            return
        key = (user_id or "", agent_id or "")
        cached = cache.get(key)
        if cached is None:
            return
        expires_at, old_tags = cached
        cache[key] = (expires_at, tuple(sorted(set(old_tags).union(new_tags))))

    async def _run_agent(
        self,
        request: WriteRequest,
        response: WriteResponse,
        vector_store: VectorStoreBase,
        mem_node: MemoryNode,
        memory_id: str,
        tracer_span,
        history_context: str = "",
        recent_history_ms: float = 0.0,
        recent_history_count: int = 0,
    ) -> None:
        """
        Full MemAgent processing: extract → reconcile & store → summary.
        Results are written back into response.extra.
        """
        mode = ProcessMode.FULL
        existing_memories = []
        for mem in (request.existing_memories or []):
            existing_memories.append({
                "memory_id": mem.get("memory_id", ""),
                "content": mem.get("content", ""),
                "layer": mem.get("layer", ""),
                "embedding": mem.get("embedding", []),
            })

        extract_timing = {
            "recent_history_ms": round(recent_history_ms, 2),
            "recent_history_count": recent_history_count,
            "collect_existing_tags_ms": 0.0,
            "existing_tags_count": 0,
            "extract_total_ms": 0.0,
        }

        try:
            # Collect all tags this user already has (for the extractor prompt to steer reuse)
            _t_existing_tags = datetime.now()
            existing_tags = await self._collect_existing_tags(
                vector_store, request.user_id, request.agent_id,
            )
            extract_timing["collect_existing_tags_ms"] = round(
                (datetime.now() - _t_existing_tags).total_seconds() * 1000, 2
            )
            extract_timing["existing_tags_count"] = len(existing_tags or [])

            # Build the extractor input: prefer the raw messages dump
            # When name is present it must be passed through: multi-person dialogues rely on it to tell "the other party is another real person" from the AI assistant
            # (role=assistant + name → real person), and it is also the authoritative source of basic_info.name.
            if request.messages:
                import json as _json_msgs
                _msgs_dump = []
                for m in request.messages:
                    _item = {"role": m.role, "content": m.content}
                    if getattr(m, "name", None):
                        _item["name"] = m.name
                    _msgs_dump.append(_item)
                _extract_content = _json_msgs.dumps(_msgs_dump, ensure_ascii=False)
            else:
                _extract_content = request.content

            # Basic profile schema fields ({name: description}), rendered into the prompt by the extractor
            _basic_profile_fields = self.config.basic_profile.effective_fields()

            # Effective summary switch for this call (per-call > config). Summaries are no longer produced in one mem_agent pass,
            # they are handled by the rolling buffer flow below (messages only), so mem_agent gets False here
            # and the old "summarize on every add" path naturally becomes a no-op (zero risk).

            _t_extract_total = datetime.now()
            agent_result = await self.mem_agent.process_add(
                content=_extract_content,
                context={"uid": request.user_id, "agent_id": request.agent_id},
                mode=mode,
                existing_memories=existing_memories,
                memory_at=request.memory_at,
                existing_tags=existing_tags,
                history_context=history_context,
                basic_profile_fields=_basic_profile_fields,
            )
            extract_timing["extract_total_ms"] = round(
                (datetime.now() - _t_extract_total).total_seconds() * 1000, 2
            )
        except Exception as agent_err:
            logger.error(f"[write] MemAgent process_add raised: {agent_err}", exc_info=True)
            self._mark_agent_failure(
                response,
                error_code="AGENT_EXCEPTION",
                error_message=str(agent_err),
            )
            response.extra["extract_timing"] = extract_timing
            tracer_span.set_output({"success": False, "error": str(agent_err)})
            return

        if agent_result is None or not agent_result.success:
            await self._handle_agent_failure(request, response, agent_result, tracer_span, extract_timing=extract_timing)
            return

        # ── Extraction succeeded ──
        _perf_req_id = request.request_id or get_request_id()
        logger.info(
            f"TRACE_PERF [{_perf_req_id}] S1_EXTRACT_DONE "
            f"extract_ms={agent_result.extract_elapsed_ms:.0f} "
            f"summary_ms={agent_result.summary_elapsed_ms:.0f} "
            f"memory={len((agent_result.extracted_info or {}).get('memory', (agent_result.extracted_info or {}).get('facts', [])))}"
        )

        # ── basic_info: prompt-driven schema, no LLM function-calling ──
        # The extractor puts the basic_info dict into extracted_info; the writer upserts the L0_BASIC_INFO
        # evolution chain here. Failures and invalid input never raise; they land in tool_results_summary and the original TOOL_CALLS pipeline log.
        basic_info_raw = None
        if isinstance(agent_result.extracted_info, dict):
            basic_info_raw = agent_result.extracted_info.pop("basic_info", None)

        tool_calls_raw = None  # v0.1.5.13+ has no real LLM tool_calls anymore; the field is kept for compatibility
        tool_results_summary: List[Dict[str, Any]] = []

        if isinstance(basic_info_raw, dict) and basic_info_raw:
            try:
                _bp_result = await upsert_basic_profile(
                    user_id=request.user_id,
                    kv=basic_info_raw,
                    vector_store=vector_store,
                    embed_service=self.embed_service,
                    allowed_fields=list(_basic_profile_fields.keys()),
                    client_meta=getattr(request, "client_meta", None),
                )
                # Wrap the upsert result into the original tool_results shape (backward-compatible pipeline log parsing)
                tool_results_summary.append({
                    "tool": "basic_profile_upsert",  # no longer an LLM tool name
                    "round": 1,
                    "success": _bp_result.success,
                    "data": _bp_result.to_dict(),
                    "error": _bp_result.error,
                })
                # The input side records the raw basic_info values produced by the LLM (traceable)
                tool_calls_raw = [{
                    "function": {
                        "name": "basic_profile_upsert",
                        "arguments": json.dumps(basic_info_raw, ensure_ascii=False, default=str),
                    }
                }]
            except Exception as bp_err:
                logger.error(f"[write] upsert_basic_profile raised: {bp_err}", exc_info=True)
                tool_results_summary.append({
                    "tool": "basic_profile_upsert",
                    "round": 1,
                    "success": False,
                    "error": str(bp_err),
                })

        extract_timing["extract_llm_ms"] = round(getattr(agent_result, "extract_elapsed_ms", 0) or 0, 2)
        response.extra["extract_timing"] = extract_timing
        response.extra["tool_results"] = tool_results_summary

        # Write the TOOL_CALLS pipeline log (observable tool invocation pipeline)
        if self._cache and (tool_calls_raw or tool_results_summary):
            try:
                import json as _json_tc
                _tc_req_id = request.request_id or get_request_id()
                tc_data = {
                    "tool_calls_input": (
                        [{"name": tc.get("function", {}).get("name", ""), "arguments": tc.get("function", {}).get("arguments", "")}
                         for tc in tool_calls_raw]
                        if isinstance(tool_calls_raw, list) else str(tool_calls_raw)[:500]
                    ) if tool_calls_raw else [],
                    "tool_results": tool_results_summary,
                    "tool_calls_only": bool(
                        tool_calls_raw
                        and not (agent_result.extracted_info or {}).get("memory")
                        and not (agent_result.extracted_info or {}).get("facts")
                        and not (agent_result.extracted_info or {}).get("identity")
                    ),
                }
                await self._cache.store_pipeline_log(
                    request_id=_tc_req_id,
                    user_id=request.user_id,
                    agent_id=request.agent_id or "default_agent",
                    session_id=request.session_id or "",
                    step="TOOL_CALLS",
                    prompt="",
                    response=_json_tc.dumps(tc_data, ensure_ascii=False, default=str),
                    parsed=_json_tc.dumps(tc_data, ensure_ascii=False, default=str),
                    memory_ids=[],
                )
            except Exception as e:
                logger.warning(f"[write] store TOOL_CALLS log failed: {e}")
        # Filter entities
        _CATEGORY_BLACKLIST = {
            "locations", "location", "persons", "person",
            "organizations", "organization", "events", "event",
            "relations", "relation", "products", "product",
            "animals", "animal", "foods", "food",
            "technologies", "technology", "tech",
            "attributes", "attribute", "others", "other",
        }
        if agent_result.extracted_info and isinstance(agent_result.extracted_info, dict):
            for entity in (agent_result.extracted_info.get("entities") or []):
                if isinstance(entity, dict):
                    name = entity.get("name", "")
                    if name and name.lower() not in _CATEGORY_BLACKLIST:
                        response.entities.append(entity)
                elif isinstance(entity, str):
                    if entity.lower() not in _CATEGORY_BLACKLIST:
                        response.entities.append({"name": entity})

        response.extra["summary"] = agent_result.summary or ""
        if agent_result.suggested_layer:
            response.layer = agent_result.suggested_layer

        conflicts = []
        for conflict in (agent_result.conflicts or []):
            conflicts.append({
                "type": conflict.get("type", ""),
                "target_id": conflict.get("target_id", ""),
                "description": conflict.get("description", ""),
                "resolution": conflict.get("resolution", ""),
            })
        response.extra["conflicts"] = conflicts
        response.extra["should_merge"] = agent_result.should_merge
        response.extra["merge_target_id"] = agent_result.merge_target_id or ""
        response.extra["_agent_tokens"] = agent_result.tokens_used
        # LLM token usage of the extract pipeline (prompt/completion/total), passed on to the caller by the client
        response.extra["extract_tokens"] = {
            "prompt": getattr(agent_result, "extract_prompt_tokens", 0) or 0,
            "completion": getattr(agent_result, "extract_completion_tokens", 0) or 0,
            "total": getattr(agent_result, "extract_tokens_used", 0) or 0,
        }

        # Write pipeline logs
        _req_id = request.request_id or get_request_id()
        await self._write_extract_log(
            request,
            agent_result,
            _req_id,
            tool_results=tool_results_summary,
            extract_timing=extract_timing,
        )
        await self._write_summary_log(request, agent_result, _req_id)

        # Reconcile & store
        stored_ids: List[str] = []
        _vs_logger = logging.getLogger("hy_memory.data.vector_store_chroma")
        _vs_level = _vs_logger.level
        _vs_logger.setLevel(logging.INFO)
        try:
            new_memory_texts, new_memories_meta = self._collect_new_memories(
                agent_result.extracted_info or {}
            )

            # Dedup extractor results: the newly extracted entries are compared with each other; the vectors keep flowing to
            # reconcile / store. Nothing is stored yet, so delete_from_store=False.
            precomputed_embeddings: Dict[str, List[float]] = {}
            if len(new_memory_texts) >= 2:
                try:
                    (
                        new_memory_texts,
                        new_memories_meta,
                        precomputed_embeddings,
                    ) = await self._dedup_extracted(
                        new_memory_texts, new_memories_meta, request, _req_id,
                    )
                except Exception as e:
                    logger.warning(f"[write] extractor dedup failed (non-fatal): {e}")

            if new_memory_texts:
                if _RECONCILE_ENABLED:
                    stored_ids, recon_error, ops_counts, recon_tokens = await self._reconcile_and_store(
                        new_memory_texts=new_memory_texts,
                        new_memories_meta=new_memories_meta,
                        request=request,
                        vector_store=vector_store,
                        req_id=_req_id,
                        source_raw_memory_id=memory_id,
                        precomputed_embeddings=precomputed_embeddings,
                    )
                else:
                    stored_ids, recon_error, ops_counts, recon_tokens = await self._direct_store(
                        new_memories_meta=new_memories_meta,
                        request=request,
                        vector_store=vector_store,
                        req_id=_req_id,
                        source_raw_memory_id=memory_id,
                        precomputed_embeddings=precomputed_embeddings,
                    )
                response.extra["reconcile_ops"] = ops_counts
                response.extra["reconcile_tokens"] = recon_tokens
                if recon_error is not None:
                    response.extra["agent_stored_ids"] = stored_ids
                    self._mark_agent_failure(
                        response,
                        error_code="RECONCILE_FAILED",
                        error_message=recon_error,
                        agent_nodes=len(stored_ids),
                    )
                    logger.warning(f"[write] reconcile failed: {recon_error}")
                    return

            # Intentions → L7 (direct upsert, no reconcile)
            # The whole block is skipped when MEMORY_ENABLE_INTENTION / LLMConfig.enable_intention is off
            if getattr(self.config.llm, "enable_intention", True):
                intentions = self._collect_intentions(agent_result.extracted_info or {})
                if intentions:
                    intention_ids = await self._store_intentions(
                        intentions=intentions,
                        request=request,
                        vector_store=vector_store,
                        req_id=_req_id,
                    )
                    stored_ids.extend(intention_ids)
                    logger.info(
                        f"TRACE_PERF [{_perf_req_id}] S1_INTENTION_DONE "
                        f"intentions={len(intention_ids)}"
                    )
            else:
                logger.debug(
                    "[write] enable_intention=false, skip L7 intention store"
                )

            response.extra["agent_stored_ids"] = stored_ids
            response.extra["agent_status"] = "success"
            response.extra["agent_error"] = ""
            response.extra["agent_nodes"] = len(stored_ids)
            _vs_logger.setLevel(_vs_level)
            logger.info(
                f"TRACE_PERF [{_perf_req_id}] S1_PERSIST_DONE "
                f"ops={len(stored_ids)} nodes to vector store"
            )
            logger.debug(f"[agent] persisted {len(stored_ids)} nodes to vector store")

            # L1_RAW → SHADOW
            #
            # The condition is "the agent finished and this node really is L1_RAW", and it **does not look at how many facts were extracted**:
            # when the extractor concludes "this conversation holds no extractable user facts" (e.g. a pure technical Q&A)
            # it returns memory=[], which is a **normal conclusion**, not a failure; the raw conversation should also step back to the backup layer.
            # The condition used to include stored_ids, so an empty extraction left it un-downgraded with zero instrumentation,
            # leaving L1_RAW in ACTIVE for a long time and recallable by list/search/keyword_search.
            #
            # A failed extraction never reaches here: an agent exception, agent_result.success=False or
            # a reconcile failure all return early above, so failure cases still keep
            # ACTIVE (the data may still be salvageable and stays archived for a retry).
            #
            # The layer guard is not optional: callers may pass extra["layer"] via metadata to explicitly specify
            # the layer (see the Layer assignment block in this file); then mem_node is not a raw conversation node, and
            # downgrading it would be a logical delete of data the caller just wrote.
            if memory_id and mem_node.layer == MemoryLayer.L1_RAW:
                try:
                    mem_node.status = MemoryStatus.SHADOW
                    await vector_store.upsert(mem_node)
                    logger.debug(f"[agent] L1 raw {memory_id} status → SHADOW")
                except Exception as shadow_err:
                    logger.warning(f"[agent] failed to shadow L1 raw: {shadow_err}")

        except Exception as persist_err:
            _vs_logger.setLevel(_vs_level)
            logger.error(f"[agent] persist failed: {persist_err}", exc_info=True)
            response.extra["agent_stored_ids"] = stored_ids
            self._mark_agent_failure(
                response,
                error_code="AGENT_PERSIST_FAILED",
                error_message=f"agent persist failed: {persist_err}",
                agent_nodes=len(stored_ids),
            )
            return

        tracer_span.set_output({
            "success": True,
            "entities_count": len(response.entities),
            "summary": agent_result.summary or "",
            "conflicts_count": len(conflicts),
            "tokens_used": agent_result.tokens_used,
            "should_merge": agent_result.should_merge,
            "agent_stored_count": len(stored_ids),
        })

    def _extract_llm_diag(self, agent_result) -> Dict[str, Any]:
        """Pull diagnostic fields from the EXTRACT LLM response to debug EMPTY_RESPONSE / long-latency empty returns."""
        _er = getattr(agent_result, "_extract_result", None) if agent_result is not None else None
        if _er is None:
            return {}
        return {
            "llm_finish_reason": getattr(_er, "llm_finish_reason", "") or "",
            "llm_model": getattr(_er, "llm_model", "") or "",
            "llm_response_id": getattr(_er, "llm_response_id", "") or "",
            "llm_choices_count": getattr(_er, "llm_choices_count", 0) or 0,
            "llm_content_len": getattr(_er, "llm_content_len", 0) or 0,
            "llm_tool_calls_count": getattr(_er, "llm_tool_calls_count", 0) or 0,
            "llm_prompt_tokens": getattr(_er, "prompt_tokens", 0) or 0,
            "llm_completion_tokens": getattr(_er, "completion_tokens", 0) or 0,
            "llm_total_tokens": getattr(_er, "tokens_used", 0) or 0,
        }

    @staticmethod
    def _mark_agent_failure(
        response: WriteResponse,
        *,
        error_code: str,
        error_message: str,
        agent_nodes: int = 0,
    ) -> None:
        """Make an internal agent failure visible to every write caller.

        L1_RAW may already have been persisted, but a full-mode write has not
        completed its memory contract when extraction/reconcile/persist fails.
        The caller must see ``success=False`` so checkpointed ingestion can retry.
        """
        code = error_code or "AGENT_FAILED"
        message = error_message or "agent processing failed"
        response.extra["agent_status"] = "failed"
        response.extra["agent_error"] = message
        response.extra["agent_error_code"] = code
        response.extra["agent_nodes"] = int(agent_nodes or 0)
        response.success = False
        response.error_code = 502
        response.error_message = f"[{code}] {message}"

    async def _handle_agent_failure(self, request, response, agent_result, tracer_span, extract_timing: Optional[Dict[str, Any]] = None) -> None:
        """Handle the case where agent extraction failed."""
        if agent_result is None:
            self._mark_agent_failure(
                response,
                error_code="AGENT_NO_RESULT",
                error_message="agent processing returned no result",
            )
            tracer_span.set_output({"success": False, "error": "agent processing failed"})
            return

        _error_code = getattr(agent_result, "error_code", "") or "AGENT_FAILED"
        _error_msg = agent_result.error or "agent processing failed"
        self._mark_agent_failure(
            response,
            error_code=_error_code,
            error_message=_error_msg,
        )
        logger.warning(f"[write] agent extract failed: code={_error_code} error={_error_msg}")

        _raw_resp = getattr(agent_result, "extract_raw_response", "") or ""
        # Rebuild the real prompt ([SYSTEM]/[USER]) the same way the success path _write_extract_log does,
        # so EXTRACT logs for failures (EMPTY_RESPONSE / JSON_PARSE_FAILED / LLM_ERROR) also show the actual input.
        _extract_prompt = ""
        _attempts = []
        _er = getattr(agent_result, "_extract_result", None)
        if _er is not None:
            if getattr(_er, "_actual_prompt", None):
                _sys = getattr(_er, "_actual_system_prompt", "") or ""
                _usr = getattr(_er, "_actual_prompt", "") or ""
                _extract_prompt = f"[SYSTEM]\n{_sys}\n\n[USER]\n{_usr}"
            _attempts = list(getattr(_er, "attempts", []) or [])
        _parsed = {"error": _error_code, "message": _error_msg}
        if extract_timing:
            _parsed.update(extract_timing)
        _parsed.update(self._extract_llm_diag(agent_result))
        _parsed["extract_llm_ms"] = round(getattr(agent_result, "extract_elapsed_ms", 0) or 0, 2)
        if _attempts:
            # Failure detail for each LLM retry (type/message/HTTP status code and so on), for troubleshooting
            _parsed["attempts"] = _attempts
        await self._emit_pipeline_step(
            request,
            step="EXTRACT",
            parsed=_parsed,
            prompt=_extract_prompt,
            response=_raw_resp,
            elapsed_ms=getattr(agent_result, "extract_elapsed_ms", 0) or 0,
            prompt_tokens=getattr(agent_result, "extract_prompt_tokens", 0) or 0,
            completion_tokens=getattr(agent_result, "extract_completion_tokens", 0) or 0,
            total_tokens=getattr(agent_result, "extract_tokens_used", 0) or 0,
        )

        tracer_span.set_output({"success": False, "error": _error_msg})

    async def _write_extract_log(
        self,
        request,
        agent_result,
        req_id: str,
        tool_results: List[Dict[str, Any]] = None,
        extract_timing: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write the EXTRACT pipeline log."""
        if not self._cache:
            return
        try:
            import json as _json

            # Use the real prompt returned by the extractor, when there is one
            _extract_prompt = ""
            _extract_result = getattr(agent_result, '_extract_result', None)
            if _extract_result and hasattr(_extract_result, '_actual_prompt') and _extract_result._actual_prompt:
                _sys = _extract_result._actual_system_prompt or ""
                _usr = _extract_result._actual_prompt or ""
                _extract_prompt = f"[SYSTEM]\n{_sys}\n\n[USER]\n{_usr}"
            else:
                # fallback: build an approximate prompt the old way (not fully accurate)
                from ..agent.extractor import EXTRACT_PROMPT
                _current_time = request.memory_at.isoformat(timespec="seconds") if request.memory_at else ""
                _extract_prompt = EXTRACT_PROMPT.format(content=request.content, current_date="", memory_at=_current_time or "", existing_tags="(see actual prompt)", last_messages="")

            # response field: prefer raw_response (the LLM's raw output)
            _raw_resp = agent_result.extract_raw_response or ""

            # parsed field: extracted_info plus tool_results metadata
            _parsed_data = dict(agent_result.extracted_info) if agent_result.extracted_info else {}
            if extract_timing:
                _parsed_data.update(extract_timing)
            _parsed_data.update(self._extract_llm_diag(agent_result))
            _parsed_data["extract_llm_ms"] = round(getattr(agent_result, "extract_elapsed_ms", 0) or 0, 2)
            if tool_results:
                _parsed_data["_tool_results"] = tool_results

            await self._cache.store_pipeline_log(
                request_id=req_id,
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step="EXTRACT",
                prompt=_extract_prompt,
                response=_raw_resp if _raw_resp else _json.dumps(_parsed_data, ensure_ascii=False, default=str),
                parsed=_json.dumps(_parsed_data, ensure_ascii=False, default=str) if _parsed_data else "",
                elapsed_ms=agent_result.extract_elapsed_ms,
                prompt_tokens=agent_result.extract_prompt_tokens,
                completion_tokens=agent_result.extract_completion_tokens,
                total_tokens=agent_result.extract_tokens_used,
            )
        except Exception as e:
            logger.warning(f"[write] store EXTRACT log failed: {e}")

    async def _write_summary_log(self, request, agent_result, req_id: str) -> None:
        """Write the SUMMARY pipeline log (when there is a summary)."""
        if not self._cache or not agent_result.summary or not agent_result.summary_tokens_used:
            return
        try:
            import json as _json_s

            # Use the real prompt returned by the summarizer
            _summary_result = getattr(agent_result, '_summary_result', None)
            if _summary_result and hasattr(_summary_result, '_actual_prompt') and _summary_result._actual_prompt:
                _summary_prompt = _summary_result._actual_prompt
            else:
                # fallback
                from ..agent.summarizer import SUMMARY_PROMPT
                from datetime import date as _date_now
                _current_time = request.memory_at.isoformat(timespec="seconds") if request.memory_at else ""
                _memory_date = _current_time[:10] if _current_time and len(_current_time) >= 10 else _date_now.today().isoformat()
                _current_date = _date_now.today().isoformat()
                _summary_prompt = SUMMARY_PROMPT.format(
                    content=request.content or "",
                    memory_date=_memory_date,
                    current_date=_current_date,
                )
            await self._cache.store_pipeline_log(
                request_id=req_id,
                user_id=request.user_id,
                agent_id=request.agent_id or "default_agent",
                session_id=request.session_id or "",
                step="SUMMARY",
                prompt=_summary_prompt,
                response=agent_result.summary,
                parsed=_json_s.dumps({"summary": agent_result.summary}, ensure_ascii=False),
                elapsed_ms=agent_result.summary_elapsed_ms,
                prompt_tokens=agent_result.summary_prompt_tokens,
                completion_tokens=agent_result.summary_completion_tokens,
                total_tokens=agent_result.summary_tokens_used,
            )
        except Exception as e:
            logger.warning(f"[write] store SUMMARY log failed: {e}")

    async def close(self) -> None:
        provider = (
            self._mem_agent.llm_provider if self._mem_agent is not None else None
        )
        if provider is not None and hasattr(provider, "close"):
            await provider.close()
        logger.debug("MemoryWriter closed")
