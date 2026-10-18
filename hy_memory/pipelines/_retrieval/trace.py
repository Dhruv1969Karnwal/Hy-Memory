"""
Read-side pipeline trace logger.

Writes the stages of one search call (embed / recall / bm25 / tag-match / rrf / evolution /
summary) into the shared `pipeline_logs` table. Step strings all carry the `READ_` prefix, mirroring the
writer side (EXTRACT / RECONCILE / SUMMARY / WRITE_SUMMARY).

External contract:
  - Every instrumentation call is best-effort: an empty cache or a failed write is ignored silently and never affects the main flow
  - Step names are constants, so the inspector can search by step prefix (e.g. `step="READ_RECALL_VEC"`)
  - The parsed field of each log holds JSON-serialized structured metadata (so the inspector UI can parse it)

Design trade-offs:
  - No new table; the existing `store_pipeline_log` is reused
  - The logger is a stateless helper, bound to a request/user/agent and then reused across steps
  - Best-effort try/except is fully contained internally, so callers do not need their own try/except
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _best_effort_log(method):
    """Wrap log_* coroutine methods: any exception, including an argument signature error, is swallowed silently.

    The contract guarantees instrumentation never affects the main flow. Previously a wrong kwarg from a caller raised during
    argument binding (before the internal try/except) and took the whole search down → total_found=0.
    Catching everything at the outermost level makes best-effort actually hold.
    """
    @functools.wraps(method)
    async def _wrapper(*args, **kwargs):
        try:
            return await method(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — a trace must never affect the main flow
            logger.debug(f"[read-trace] {getattr(method, '__name__', '?')} swallowed: {e}")
            return None
    return _wrapper


# ================================================================
# Step constants
# ================================================================

# Reader common
STEP_READ_SCENE_JUDGE   = "READ_SCENE_JUDGE"      # productivity (coding) vs normal scene decision (first search step, persisted by client)
STEP_READ_REQUEST       = "READ_REQUEST"          # input snapshot
STEP_READ_EMBED_QUERY   = "READ_EMBED_QUERY"      # query embedding
STEP_READ_INTENT        = "READ_INTENT"           # intent classification result (hybrid/*_tag)

# Recall channels
STEP_READ_RECALL_VEC        = "READ_RECALL_VEC"        # Channel A: main vector recall
STEP_READ_HIDDEN_L0_CONTEXT = "READ_HIDDEN_L0_CONTEXT" # L0 used internally, not returned by Search
STEP_READ_KEYWORD_EMBED     = "READ_KEYWORD_EMBED"     # Channel B: keyword batch embed
STEP_READ_TAG_MATCH         = "READ_TAG_MATCH"         # Channel B: tag_index similarity match
STEP_READ_RECALL_TAG        = "READ_RECALL_TAG"        # Channel B: tag filter vector recall

# Fusion / ranking / synthesis
STEP_READ_BM25          = "READ_BM25"             # Channel C: BM25 rerank
STEP_READ_RRF           = "READ_RRF"              # RRF fusion
STEP_READ_MERGE_SCHEMA  = "READ_MERGE_SCHEMA"     # schema merge + truncation
STEP_READ_EVOLUTION     = "READ_EVOLUTION"        # evolution chain walk-back and synthesis
STEP_READ_INTENTION     = "READ_INTENTION"        # L7 intention recall (Proactive channel, incl. expiry conversion / embedding backfill / dedup)

# Overview
STEP_READ_GRAPH_RERANK  = "READ_GRAPH_RERANK"    # graph rerank/expansion (usages A/B/C + episode bounding)
STEP_READ_SUMMARY       = "READ_SUMMARY"          # aggregate overview (mirrors WRITE_SUMMARY on the write side)

# Coding (productivity) recall path
STEP_READ_CODING_EMBED  = "READ_CODING_EMBED"     # coding query embedding
STEP_READ_CODING_VDB    = "READ_CODING_VDB"       # coding store vector recall (before/after min_score filtering)
STEP_READ_CODING_RESULT = "READ_CODING_RESULT"    # final coding recall result overview


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def _preview_node(item: Dict[str, Any], *, content_limit: int = 120) -> Dict[str, Any]:
    """Compress one hit into a compact, readable preview structure."""
    node = item.get("node")
    content = ""
    layer = ""
    tags: List[str] = []
    if node is not None:
        content = (getattr(node, "content", "") or "")[:content_limit]
        try:
            layer = node.layer.value if node.layer else ""
        except Exception:
            layer = ""
        try:
            tags = list(getattr(node, "tags", None) or [])
        except Exception:
            tags = []
    out = {
        "node_id": item.get("node_id", ""),
        "score": round(float(item.get("score", 0.0)), 6),
        "layer": layer,
        "content": content,
        "tags": tags,
    }
    if "_semantic" in item:
        out["_semantic"] = round(float(item["_semantic"]), 6)
    if "_bm25" in item:
        out["_bm25"] = round(float(item["_bm25"]), 6)
    if item.get("source"):
        out["source"] = item["source"]
    return out


def _preview_hits(items: List[Dict[str, Any]], *, top: int = 10) -> List[Dict[str, Any]]:
    return [_preview_node(it) for it in (items or [])[:top]]


# ================================================================
# Logger
# ================================================================

def _wrap_log_methods(cls):
    """Put every log_* coroutine method behind the best-effort wrapper (argument signature errors included)."""
    import inspect
    for name, attr in list(vars(cls).items()):
        if name.startswith("log_") and inspect.iscoroutinefunction(attr):
            setattr(cls, name, _best_effort_log(attr))
    return cls


@_wrap_log_methods
class ReadTraceLogger:
    """
    Lightweight trace writer for the read pipeline.

    Usage:
        tracer_log = ReadTraceLogger(cache, request_id, user_id, agent_id)
        await tracer_log.log_embed_query(query, dims=384, cache_hit=False, elapsed_ms=12.3)
        ...

    Every log_* method is best-effort (cache is None / write exception / argument error → silently skipped,
    never affecting the main search flow).
    """


    def __init__(
        self,
        cache: Any,
        request_id: str,
        user_id: str,
        agent_id: str,
        reader_version: str = "",
        session_id: str = "",
    ):
        self._cache = cache
        self._request_id = request_id or ""
        self._user_id = user_id or ""
        self._agent_id = agent_id or "default_agent"
        self._reader_version = reader_version or ""
        self._session_id = session_id or ""

    @property
    def enabled(self) -> bool:
        return self._cache is not None and bool(self._request_id)

    async def _write(
        self,
        step: str,
        *,
        prompt: str = "",
        response: str = "",
        parsed: Optional[Dict[str, Any]] = None,
        memory_ids: Optional[List[str]] = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        try:
            # parsed uniformly carries the reader tag so the inspector can tell which reader version ran
            p_dict = dict(parsed or {})
            if self._reader_version and "_reader" not in p_dict:
                p_dict["_reader"] = self._reader_version
            await self._cache.store_pipeline_log(
                request_id=self._request_id,
                user_id=self._user_id,
                agent_id=self._agent_id,
                session_id=self._session_id,
                step=step,
                prompt=prompt,
                response=response,
                parsed=_safe_json(p_dict),
                memory_ids=memory_ids or None,
                elapsed_ms=float(elapsed_ms or 0.0),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
        except Exception as e:
            logger.debug(f"[read-trace] store_pipeline_log step={step} failed: {e}")

    # ------------------------------------------------------------------
    # Per-step instrumentation
    # ------------------------------------------------------------------

    async def log_scene_judge(
        self,
        *,
        query: str,
        scene: str,
        mode: str,
        is_coding: bool,
        rewrite_query: Optional[str] = None,
        rewrite_used: bool = False,
        created_after: Optional[str] = None,
        elapsed_ms: float = 0.0,
        llm_ms: Optional[float] = None,
        gc_delta: Optional[int] = None,
        throttle_ms: Optional[float] = None,
    ) -> None:
        """Productivity (coding) vs normal scene decision. First step of search, persisted by the client before the reader runs.

        mode: "explicit" (scene specified explicitly, no LLM call) / "llm" (multi-query LLM classification + rewrite)
              / "default" (single query or pipeline disabled, defaults to normal)
        created_after: lower time bound extracted by query rewrite, already converted to a readable date string (e.g. "2026-06-24"),
                       None when there is none.

        Timing field semantics (keep these apart; they were once confused):
        - elapsed_ms → persisted as `parsed.step_ms`: **total wall-clock time of this stage** (t0 → this call),
          regardless of whether an LLM was called. For mode=default/explicit it is the cost of a purely synchronous block.
        - llm_ms: **only set for mode="llm"**, the net duration of that judge/rewrite LLM call.
          mode=default/explicit calls no LLM → not passed → the field is absent from parsed.

        Runtime stall attribution fields (not persisted when None, keeping parsed lean):
        - gc_delta: increase in the cumulative GC count within this segment (t0 → this call). >=1 means GC ran in between.
        - throttle_ms: how long (milliseconds) this segment spent under cgroup CPU throttling. ≈step_ms means container
          CPU throttling caused the stall.
        All three are evaluated synchronously on the business side like elapsed_ms, so they carry equal weight; they separate
        GIL contention / long GC pauses / cgroup throttling as the three kinds of process-level stalls.
        """
        _parsed = {
            "scene": scene,
            "mode": mode,
            "is_coding": is_coding,
            "rewrite_query": rewrite_query,
            "rewrite_used": rewrite_used,
            "created_after": created_after,
        }
        if llm_ms is not None:
            _parsed["llm_ms"] = round(float(llm_ms), 2)
        if gc_delta is not None:
            _parsed["gc_delta"] = int(gc_delta)
        if throttle_ms is not None:
            _parsed["throttle_ms"] = round(float(throttle_ms), 2)
        await self._write(
            STEP_READ_SCENE_JUDGE,
            prompt=query or "",
            response=_safe_json({
                "scene": scene,
                "mode": mode,
                "is_coding": is_coding,
                "rewrite_used": rewrite_used,
                "created_after": created_after,
            }),
            parsed=_parsed,
            elapsed_ms=elapsed_ms,
        )

    async def log_coding_embed(
        self,
        *,
        query: str,
        dims: int = 0,
        elapsed_ms: float = 0.0,
        provider_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """coding path: query embedding."""
        parsed: Dict[str, Any] = {"dims": dims}
        if provider_meta:
            parsed["provider_meta"] = provider_meta
        await self._write(
            STEP_READ_CODING_EMBED,
            prompt=query or "",
            response=_safe_json({"dims": dims}),
            parsed=parsed,
            elapsed_ms=elapsed_ms,
        )

    async def log_coding_vdb(
        self,
        *,
        query: str,
        raw_hits: int,
        kept_hits: int,
        min_score: float,
        top_scores: Optional[List[float]] = None,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        """coding path: coding store vector recall + min_score filtering.

        raw_hits: oversampled recall count; kept_hits: count left after min_score filtering.
        top_scores: top scores before filtering, useful to diagnose "scores too low or collection empty".
        """
        await self._write(
            STEP_READ_CODING_VDB,
            prompt=query or "",
            response=_safe_json({
                "raw_hits": raw_hits,
                "kept_hits": kept_hits,
                "min_score": min_score,
            }),
            parsed={
                "raw_hits": raw_hits,
                "kept_hits": kept_hits,
                "min_score": min_score,
                "top_scores": [round(float(s), 4) for s in (top_scores or [])[:10]],
                "workspace_id": workspace_id,
                "branch": branch,
            },
            elapsed_ms=elapsed_ms,
        )

    async def log_coding_result(
        self,
        *,
        query: str,
        returned: int,
        memory_ids: Optional[List[str]] = None,
        previews: Optional[List[Dict[str, Any]]] = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        """coding path: overview of the final returned results (search wrap-up)."""
        await self._write(
            STEP_READ_CODING_RESULT,
            prompt=query or "",
            response=_safe_json({"returned": returned}),
            parsed={"returned": returned, "hits": previews or [],
                    "elapsed_ms": round(float(elapsed_ms or 0.0), 2)},
            memory_ids=memory_ids or None,
            # Wrap-up overview: elapsed_ms is the total duration of the whole coding search (an aggregate, not one stage),
            # so the persisted elapsed_ms column is 0 and the total stays in parsed.elapsed_ms.
            elapsed_ms=0,
        )

    async def log_request(
        self,
        *,
        query: str,
        limit: int,
        layers: Optional[List[str]],
        min_score: float = 0.0,
        schema_min_score: Optional[float] = None,
        schema_limit: Optional[int] = None,
        user_ids: List[str],
        agent_ids: List[str],
        session_ids: List[str],
        intention_limit: Optional[int] = None,
    ) -> None:
        # intention_limit: the L7 intention recall cap passed by the caller. Both 0 and None make the
        # reader skip the intention channel; only by persisting it explicitly into READ_REQUEST.parsed can
        # the logs distinguish "upstream passed nothing" from "reader skipped it internally", two completely different failure modes
        # (the direct reason the 0805 incident took so long to trace).
        parsed = {
            "query": query,
            "limit": limit,
            "layers": layers,
            "min_score": min_score,
            "schema_min_score": schema_min_score,
            "schema_limit": schema_limit,
            "intention_limit": intention_limit,
            "user_ids": user_ids,
            "agent_ids": agent_ids,
            "session_ids": session_ids,
        }
        await self._write(
            STEP_READ_REQUEST,
            prompt=query or "",
            response=_safe_json({
                "limit": limit,
                "min_score": min_score,
                "user_ids": user_ids,
                "agent_ids": agent_ids,
                "session_ids": session_ids,
            }),
            parsed=parsed,
        )

    async def log_embed_query(
        self,
        *,
        query: str,
        dims: int,
        cache_hit: bool,
        elapsed_ms: float = 0.0,
        provider_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        parsed: Dict[str, Any] = {"dims": dims, "cache_hit": cache_hit}
        if provider_meta:
            parsed["provider_meta"] = provider_meta
        await self._write(
            STEP_READ_EMBED_QUERY,
            prompt=query or "",
            response=_safe_json({"dims": dims, "cache_hit": cache_hit}),
            parsed=parsed,
            elapsed_ms=elapsed_ms,
        )

    async def log_intent(self, *, query: str, intent: str, keywords: List[str]) -> None:
        await self._write(
            STEP_READ_INTENT,
            prompt=query or "",
            response=intent,
            parsed={"intent": intent, "keywords": keywords},
        )

    async def log_recall_vec(
        self,
        *,
        pool_size: int,
        hits: List[Dict[str, Any]],
        elapsed_ms: float = 0.0,
        sparse_query: Optional[Dict[str, Any]] = None,
    ) -> None:
        preview = _preview_hits(hits)
        parsed = {
            "pool_size": pool_size,
            "returned": len(hits),
            "hits": preview,
        }
        if sparse_query is not None:
            parsed["sparse_query"] = sparse_query
        await self._write(
            STEP_READ_RECALL_VEC,
            prompt=f"limit={pool_size}",
            response=_safe_json({"count": len(hits), "top": preview[:5]}),
            parsed=parsed,
            memory_ids=[h.get("node_id", "") for h in hits if h.get("node_id")],
            elapsed_ms=elapsed_ms,
        )

    async def log_hidden_l0_context(
        self,
        *,
        hits: List[Dict[str, Any]],
        elapsed_ms: float = 0.0,
    ) -> None:
        preview = _preview_hits(hits)
        parsed = {
            "returned": len(hits),
            "hits": preview,
        }
        await self._write(
            STEP_READ_HIDDEN_L0_CONTEXT,
            prompt="",
            response=_safe_json({"count": len(hits), "top": preview[:5]}),
            parsed=parsed,
            memory_ids=[h.get("node_id", "") for h in hits if h.get("node_id")],
            elapsed_ms=elapsed_ms,
        )

    async def log_keyword_embed(
        self,
        *,
        keywords: List[str],
        vec_count: int,
        elapsed_ms: float = 0.0,
        provider_metas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        parsed: Dict[str, Any] = {"keywords": keywords, "vec_count": vec_count}
        if provider_metas is not None:
            parsed["provider_metas"] = provider_metas
        await self._write(
            STEP_READ_KEYWORD_EMBED,
            prompt=_safe_json(keywords),
            response=_safe_json({"vec_count": vec_count}),
            parsed=parsed,
            elapsed_ms=elapsed_ms,
        )

    async def log_tag_match(
        self,
        *,
        keywords: List[str],
        hit_tags: List[str],
        topk: int,
        min_score: float,
        elapsed_ms: float = 0.0,
    ) -> None:
        parsed = {
            "keywords": keywords,
            "hit_tags": hit_tags,
            "topk": topk,
            "min_score": min_score,
        }
        await self._write(
            STEP_READ_TAG_MATCH,
            prompt=_safe_json(keywords),
            response=_safe_json(hit_tags),
            parsed=parsed,
            elapsed_ms=elapsed_ms,
        )

    async def log_recall_tag(
        self,
        *,
        hit_tags: List[str],
        pool_size: int,
        hits: List[Dict[str, Any]],
        elapsed_ms: float = 0.0,
    ) -> None:
        preview = _preview_hits(hits)
        parsed = {
            "hit_tags": hit_tags,
            "pool_size": pool_size,
            "returned": len(hits),
            "hits": preview,
        }
        await self._write(
            STEP_READ_RECALL_TAG,
            prompt=_safe_json(hit_tags),
            response=_safe_json({"count": len(hits), "top": preview[:5]}),
            parsed=parsed,
            memory_ids=[h.get("node_id", "") for h in hits if h.get("node_id")],
            elapsed_ms=elapsed_ms,
        )

    async def log_bm25(
        self,
        *,
        pool_size: int,
        query_terms: List[str],
        hits: List[Dict[str, Any]],
        raw_hits: Optional[List[Dict[str, Any]]] = None,
        bm25_scores: Optional[Dict[str, float]] = None,
        normalize_method: Optional[str] = None,
        sigmoid_midpoint: Optional[float] = None,
        sigmoid_steepness: Optional[float] = None,
        has_bm25: Optional[bool] = None,
        elapsed_ms: float = 0.0,
        sparse_query: Optional[Dict[str, Any]] = None,
    ) -> None:
        """BM25/keyword channel instrumentation.

        Beyond the fused `hits`, it exposes as much of BM25's actual behaviour as possible:
          - keyword_count / has_bm25: whether backend keyword recall hit for this query (decides whether weighting kicks in);
          - normalize_method: `passthrough` (backend scores already in [0,1]) /
            `sigmoid` (raw classic BM25 scores → sigmoid normalization);
          - sigmoid_midpoint / sigmoid_steepness: sigmoid normalization parameters (adaptive to the query term count);
          - bm25_detail: the raw → normalized transform for each keyword hit (shows how a single BM25 score feeds the ranking);
          - sparse_query: the sparse vector produced by BM25Encoder.encode_queries (nnz + the full [[token_id, weight], ...]).
        """
        preview = _preview_hits(hits)
        # raw→normalized detail for keyword hits (sorted by raw score descending, easier to debug)
        bm25_detail: List[Dict[str, Any]] = []
        _scores = bm25_scores or {}
        _raw = raw_hits if raw_hits is not None else []
        for r in sorted(_raw, key=lambda x: x.get("score", 0.0), reverse=True)[:20]:
            nid = r.get("node_id", "")
            bm25_detail.append({
                "node_id": nid,
                "raw": round(float(r.get("score", 0.0)), 6),
                "norm": round(float(_scores.get(nid, 0.0)), 6),
            })
        keyword_count = len(_raw)
        _has = has_bm25 if has_bm25 is not None else bool(_scores or _raw)
        parsed = {
            "pool_size": pool_size,
            "query_terms": query_terms,
            "keyword_count": keyword_count,
            "has_bm25": _has,
            "normalize_method": normalize_method,
            "sigmoid_midpoint": round(float(sigmoid_midpoint), 6) if sigmoid_midpoint is not None else None,
            "sigmoid_steepness": round(float(sigmoid_steepness), 6) if sigmoid_steepness is not None else None,
            "bm25_detail": bm25_detail,
            "returned": len(hits),
            "hits": preview,
        }
        if sparse_query is not None:
            parsed["sparse_query"] = sparse_query
        await self._write(
            STEP_READ_BM25,
            prompt=_safe_json(query_terms),
            response=_safe_json({
                "keyword_count": keyword_count,
                "has_bm25": _has,
                "normalize_method": normalize_method,
                "top_bm25": bm25_detail[:5],
                "sparse_nnz": (sparse_query or {}).get("nnz") if sparse_query else None,
            }),
            parsed=parsed,
            memory_ids=[h.get("node_id", "") for h in hits if h.get("node_id")],
            elapsed_ms=elapsed_ms,
        )

    async def log_rrf(
        self,
        *,
        channels: Dict[str, int],          # {"vec": 30, "tag": 18, "bm25": 12}
        weights: Dict[str, float],
        fused: List[Dict[str, Any]],
        confidence: float,
        is_low_confidence: bool,
        top_n: int = 10,
    ) -> None:
        preview: List[Dict[str, Any]] = []
        for item in (fused or [])[:top_n]:
            preview.append({
                "node_id": item.get("node_id", ""),
                "rrf_score": round(float(item.get("rrf_score", 0.0)), 6),
                "rank_by_channel": item.get("rrf_rank_by_channel", {}),
                "per_channel_score": {
                    k: round(float(v), 6)
                    for k, v in (item.get("per_channel_score") or {}).items()
                },
            })
        parsed = {
            "channels": channels,
            "weights": weights,
            "confidence": round(float(confidence), 4),
            "is_low_confidence": bool(is_low_confidence),
            "top": preview,
        }
        await self._write(
            STEP_READ_RRF,
            prompt=_safe_json(weights),
            response=_safe_json({
                "confidence": round(float(confidence), 4),
                "is_low_confidence": bool(is_low_confidence),
                "top": preview[:5],
            }),
            parsed=parsed,
            memory_ids=[p["node_id"] for p in preview if p.get("node_id")],
        )

    async def log_merge_schema(
        self,
        *,
        fused_size: int,
        schema_size: int,
        merged_size: int,
        final_limit: int,
    ) -> None:
        await self._write(
            STEP_READ_MERGE_SCHEMA,
            prompt="",
            response=_safe_json({
                "fused_size": fused_size,
                "schema_size": schema_size,
                "merged_size": merged_size,
                "final_limit": final_limit,
            }),
            parsed={
                "fused_size": fused_size,
                "schema_size": schema_size,
                "merged_size": merged_size,
                "final_limit": final_limit,
            },
        )

    async def log_graph_rerank(
        self,
        *,
        stats: Optional[Dict[str, Any]] = None,
        pool_size: int = 0,
        admitted_ids: Optional[List[str]] = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        """Graph rerank / expansion (usages A/B/C + episode bounding).

        The effect of this step **must be separable from "the pool got bigger"**, so stats carries the population at every funnel level:
        edges / propagated / reachable / episode_candidates / prefiltered / admitted / paths。
        Without it, production only shows "the results changed" with no way to attribute which level admitted them.
        """
        st = dict(stats or {})
        st["pool_size"] = pool_size
        if admitted_ids:
            st["admitted_ids"] = list(admitted_ids)[:50]
        await self._write(
            STEP_READ_GRAPH_RERANK,
            prompt="",
            response=_safe_json(st),
            parsed=st,
            memory_ids=list(admitted_ids or [])[:50] or None,
            elapsed_ms=elapsed_ms,
        )

    async def log_evolution(
        self,
        *,
        input_size: int,
        evolved_count: int,
        elapsed_ms: float = 0.0,
    ) -> None:
        await self._write(
            STEP_READ_EVOLUTION,
            prompt="",
            response=_safe_json({
                "input_size": input_size,
                "evolved_count": evolved_count,
            }),
            parsed={
                "input_size": input_size,
                "evolved_count": evolved_count,
            },
            elapsed_ms=elapsed_ms,
        )

    async def log_intention(
        self,
        *,
        stats: Optional[Dict[str, Any]] = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        """Stage timings and counts for L7 intention recall (Proactive channel).

        This step had **no instrumentation at all** before, which left an unexplained gap between
        READ_EVOLUTION → READ_SUMMARY (426ms seen in production). stats is backfilled by
        `_retrieval.intention.recall_intentions()` through stats_out, split into
        sub-stages so the slow segment can be located directly:

        - `vdb_search_ms`      VDB vector search (oversampled L7 fetch, fetch_limit=limit×6)
        - `expire_ms`          lazy expiry conversion (per-node update_payload into L2_FACT)
        - `backfill_embed_ms`  backfilling missing embeddings before dedup (get_embeddings)
        - `dedup_ms`           dedup (pairwise cosine comparison, pure CPU)
        - `total_ms`           end to end for recall_intentions
        Counts: `fetched` / `expired` / `survivors` / `dedup_dropped` / `returned`

        `stats.total_ms` is the function's own end-to-end duration, suitable for comparing across call paths.
        """
        await self._write(
            STEP_READ_INTENTION,
            parsed=dict(stats or {}),
            elapsed_ms=elapsed_ms,
        )

    async def log_summary(
        self,
        *,
        query: str,
        intent: Optional[str],
        confidence: Optional[float],
        is_low_confidence: Optional[bool],
        channels: Dict[str, int],
        total_found: int,
        elapsed_ms: float,
        returned_memories: List[Dict[str, Any]],
    ) -> None:
        """Overview log for one search, the read-side counterpart of WRITE_SUMMARY on the writer side."""
        parsed = {
            "query": query,
            "reader": self._reader_version,
            "intent": intent,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "is_low_confidence": is_low_confidence,
            "channels": channels,
            "total_found": total_found,
            "elapsed_ms": round(float(elapsed_ms or 0.0), 2),
            "returned": [
                {
                    "memory_id": m.get("memory_id", ""),
                    "layer": m.get("layer", ""),
                    "score": round(float(m.get("score", 0.0)), 6),
                    "content": (m.get("content", "") or "")[:160],
                    "tags": m.get("tags") or [],
                }
                for m in (returned_memories or [])[:20]
            ],
        }
        await self._write(
            STEP_READ_SUMMARY,
            prompt=query or "",
            response=_safe_json({
                "reader": self._reader_version,
                "intent": intent,
                "confidence": parsed["confidence"],
                "total_found": total_found,
                "elapsed_ms": parsed["elapsed_ms"],
            }),
            parsed=parsed,
            memory_ids=[m.get("memory_id") for m in (returned_memories or []) if m.get("memory_id")],
            # Overview of one search: elapsed_ms is the total duration of the whole search (an aggregate, not one stage),
            # so the persisted elapsed_ms column is 0 and the total stays in parsed.elapsed_ms / response.
            elapsed_ms=0,
        )
