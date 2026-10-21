"""
Agent Memory - EmbedService embedding service

Converts text into vector representations.

Features:
- Single-text embedding
- Batch embedding
- Multi-model support
- Cache mechanism
- Write-path batching queue (embed_queued): collects high-concurrency single embed requests
  into one batch and sends them in a single call, so a flood of concurrent write requests cannot

Example:
    embed_service = EmbedService(config)
    
    # Single embed (search path, direct concurrency)
    vector = await embed_service.embed("The user likes Sichuan food")
    
    # Batched embed (write path, collected in the queue and then sent)
    vector = await embed_service.embed_queued("The user likes Sichuan food")
    
    # Batch embed
    vectors = await embed_service.embed_batch([
        "The user likes Sichuan food",
        "The user lives in Beijing",
    ])
"""

from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import asyncio
import hashlib
import importlib.util
import json
from json import JSONDecodeError
import logging
import threading
import time

from ..config import MemoryConfig
from .workload import embedding_slot
from ..utils.log_setup import get_request_id

logger = logging.getLogger(__name__)

# ── embedding input length limit ─────────────────────────────────────────
#
# text-embedding-3-small hard-limits input at 8192 tokens; one token over and the whole call 400s:
#   Invalid 'input[0]': maximum input length is 8192 tokens
#
# Why this layer has to absorb it: retrieval can hand the embedding endpoint a whole query
# carrying long context, formulas or code. A deterministic oversize never recovers on retry,
# so truncation must happen here, where all embedding entry points converge.
#
# Truncate rather than raise: query meaning lives mostly in the head, so cutting the tail beats
# failing the whole search; same on the write side -- losing a fact is worse than a truncated one.
#
# Keep a 64-token margin: we estimate with cl100k_base, the server tokenizer may differ, and
# cutting right at 8192 would still hit occasional 400s.
_EMBED_MAX_TOKENS = 8192 - 64

# With tiktoken installed we cut exactly by token; without it we fall back to a character estimate.
# 1 token ~= 4 chars is the English rule of thumb; Chinese is closer to 1:1, so 2 is used as a
_CHARS_PER_TOKEN_FALLBACK = 2

# In-process truncation counters, quantifying how often and how much input gets truncated.
_TRUNCATION_STATS: Dict[str, int] = {"calls": 0, "truncated": 0, "tokens_dropped": 0}


def embed_truncation_stats() -> Dict[str, int]:
    """Cumulative embedding truncation counters for this process (calls / truncated / tokens_dropped)."""
    return dict(_TRUNCATION_STATS)


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text) // _CHARS_PER_TOKEN_FALLBACK + 1


def _truncate_for_embedding(text: str, limit: int = _EMBED_MAX_TOKENS) -> str:
    """Truncate one text to a length embedding accepts; returned as-is when within the limit.

    Prefers an exact token cut (with tiktoken available it decodes back to a string, so boundaries
    are exact), otherwise cuts by a conservative character estimate. Both paths trim the tail only.
    """
    _TRUNCATION_STATS["calls"] += 1
    if not text:
        return text
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        if len(ids) <= limit:
            return text
        dropped = len(ids) - limit
        out = enc.decode(ids[:limit])
    except Exception:
        est = len(text) // _CHARS_PER_TOKEN_FALLBACK + 1
        if est <= limit:
            return text
        keep = limit * _CHARS_PER_TOKEN_FALLBACK
        dropped = est - limit
        out = text[:keep]
    _TRUNCATION_STATS["truncated"] += 1
    _TRUNCATION_STATS["tokens_dropped"] += dropped
    logger.warning(
        f"[embed] input truncated to {limit} tokens "
        f"(dropped ~{dropped}; total truncated={_TRUNCATION_STATS['truncated']})"
    )
    return out


# Per-call endpoint override (contextvar -> concurrency safe, propagates across asyncio tasks).
# Only overrides the access-endpoint fields (base_url / api_key / extra_headers / timeout);
# **never model / embedding_dims / extra_body** -- those decide the vectors themselves and changing
# them would make new writes incompatible with stored vectors. Typical use: the app layer resolves
# a Venus direct ip:port and reaches the same embedding model through a faster endpoint. Empty falls back to the EmbedService's own config.
_embed_endpoint_override: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "hy_embed_endpoint_override", default=None
)


@contextmanager
def embed_endpoint_override(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
):
    """Within this context, embedding HTTP requests from every EmbedService use the given endpoint.

    Only the fields you pass take effect (None keeps the EmbedService's own config). model / dims never change.
    Implemented with a contextvar: set on entry, reset on exit, so it is concurrency safe by nature and
    propagates across await / new tasks into the embed calls inside writer / reader, no call site has to change.
    """
    payload: Dict[str, Any] = {}
    if base_url is not None:
        payload["base_url"] = base_url
    if api_key is not None:
        payload["api_key"] = api_key
    if extra_headers is not None:
        payload["extra_headers"] = extra_headers
    if timeout is not None:
        payload["timeout"] = timeout
    token = _embed_endpoint_override.set(payload or None)
    try:
        yield
    finally:
        _embed_endpoint_override.reset(token)

_SENSITIVE_HEADER_KEYS = frozenset({
    "authorization", "set-cookie", "cookie", "api-key", "x-api-key",
    "x-auth-token", "proxy-authorization",
})


def _openai_version() -> str:
    """openai SDK version string (best-effort, for debug logs)."""
    try:
        import openai
        return getattr(openai, "__version__", "unknown")
    except Exception:
        return "unknown"


def _redact_headers(headers: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Strip/redact auth-related headers before logging."""
    if not headers:
        return headers
    out: Dict[str, Any] = {}
    for k, v in headers.items():
        if str(k).lower() in _SENSITIVE_HEADER_KEYS:
            out[k] = "***"
        else:
            out[k] = v
    return out


def _embeddings_api_url(base_url: str) -> str:
    """OpenAI SDK posts to ``{base_url}/embeddings`` (base_url usually ends with /v1)."""
    base = (base_url or "").rstrip("/")
    if not base:
        return "/embeddings"
    return f"{base}/embeddings"


@dataclass
class EmbedServiceConfig:
    """
    Embedding service configuration
    """
    provider: str = "openai"
    model: str = "text-embedding-ada-002"
    api_key: str = ""
    base_url: str = ""
    embedding_dims: int = 1536
    batch_size: int = 100
    enable_cache: bool = True
    cache_size: int = 10000
    max_retries: int = 5
    retry_delay: float = 3.0
    timeout: int = 60
    # Write-path batching queue parameters
    # Semantics: within one effective endpoint the window closes once and sends every request
    # collected inside it as **one batched HTTP call**; a full batch_size is sent immediately without waiting.
    # The window is 100ms (not 1000ms): it is pure added latency, batching only merges requests and
    # must not make users wait an extra second. A 1000ms window with per-item serial sends once produced queue_wait 6.8s.
    queue_batch_size: int = 8              # Send immediately once this many items pile up (a cap, not a target)
    queue_batch_window_ms: float = 100.0   # Closing window; flush on expiry with whatever has accumulated
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None


class EmbedService:
    """
    Embedding service
    
    Converts text into vectors.
    - embed(): sends directly, fits the search path (low latency first)
    - embed_queued(): sends through the batching queue, fits the write path (fewer concurrent calls, protects the service)
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig.from_env()
        self._embed_config = self._build_config()
        
        # Cache
        # LRU rather than a fill-once dict. The old cache stopped accepting new
        # entries after 10k unique texts, so long eval runs lost all locality:
        # a vector computed during dedup/reconcile was requested again at store.
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        
        # Stats
        self._embed_count = 0
        self._cache_hits = 0

        # Write-path batching queue (lazily initialized on the first embed_queued call).
        #
        # **Bucketed by endpoint**: when requests share one EmbedService and use per-call endpoint
        # overrides, the queue must be isolated per effective endpoint, otherwise:
        #   1) correctness: a batch mixing endpoints runs _send_batch under only one of the
        #      embed_endpoint_override contextvars, so the other texts go to the **wrong
        #      endpoint** (wrong base_url/api_key);
        #   2) performance: one slow endpoint holds up all the others;
        #   3) observability: batch_size / queue_depth mix endpoints and stop being truthful.
        # The bucket key is the currently effective (base_url, api_key).
        self._queue_locks: Dict[tuple, asyncio.Lock] = {}
        self._queue_pending: Dict[tuple, List[tuple]] = {}   # bucket -> [(text, Future, enter_at, lock_wait_ms, depth)]
        self._queue_flush_tasks: Dict[tuple, Optional[asyncio.Task]] = {}

        # Last embed response metadata (non-sensitive headers + body metadata)
        self._last_embed_meta: Dict[str, Any] = {}

        # AsyncOpenAI client cache: reused per (loop_id, base_url, api_key, timeout).
        # Constructing AsyncOpenAI eagerly builds an SSLContext + httpx connection pool (tens of ms), and
        # rebuilding it per embed was the main cost (client_build_ms). Reusing it makes only the first request pay for the connection.
        # The key carries loop_id: httpx AsyncClient binds to an event loop, and the SDK's loop threads (write/search)
        # cannot reuse across loops; it also carries the endpoint, since per-call overrides swap base_url/api_key.
        # On retry (retryable error) that key is evicted and rebuilt, so a wedged connection is never reused.
        self._client_cache: Dict[tuple, Any] = {}

        logger.info(f"EmbedService initialized, model={self._embed_config.model}")
    
    def _build_config(self) -> EmbedServiceConfig:
        """Build the embedding configuration from settings"""
        embedder = self.config.embedder
        import os

        return EmbedServiceConfig(
            provider=embedder.provider,
            model=embedder.model,
            api_key=embedder.api_key,
            base_url=embedder.base_url,
            embedding_dims=embedder.embedding_dims,
            max_retries=embedder.max_retries if embedder.max_retries is not None else 5,
            retry_delay=embedder.retry_delay if embedder.retry_delay is not None else 3.0,
            timeout=embedder.timeout if embedder.timeout is not None else 60,
            queue_batch_size=int(os.getenv("MEMORY_EMBED_BATCH_SIZE", "8")),
            queue_batch_window_ms=float(os.getenv("MEMORY_EMBED_BATCH_WINDOW_MS", "100")),
            extra_headers=embedder.extra_headers,
            extra_body=embedder.extra_body,
        )
    
    def _get_cache_key(self, text: str) -> str:
        """Build the cache key"""
        return hashlib.md5(text.encode()).hexdigest()

    def _cache_get(self, cache_key: str) -> Optional[List[float]]:
        with self._cache_lock:
            vector = self._cache.get(cache_key)
            if vector is not None:
                self._cache.move_to_end(cache_key)
            return vector

    def _cache_put(self, cache_key: str, vector: List[float]) -> None:
        if self._embed_config.cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[cache_key] = vector
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._embed_config.cache_size:
                self._cache.popitem(last=False)

    def _endpoint_awareness(self) -> Dict[str, Any]:
        """Currently effective provider / endpoint awareness (honouring per-call contextvar overrides).

        Reused by the cache-hit path -- a cache hit issues no HTTP, so it cannot obtain the meta computed in _embed_openai,
        but provider/endpoint are still determinable (config plus current override); they are merged into meta so
        S1_EMBED / DEDUP / READ_EMBED_QUERY stay endpoint-aware on cache hits as well.
        """
        _ov = _embed_endpoint_override.get()
        base_url = self._embed_config.base_url
        if _ov and _ov.get("base_url") is not None:
            base_url = _ov["base_url"]
        return {
            "provider": self._embed_config.provider,
            "endpoint": _embeddings_api_url(base_url),
            "endpoint_overridden": bool(_ov),
        }

    def _cache_hit_meta(self) -> Dict[str, Any]:
        """Meta for a cache hit: cache_hit=True plus endpoint awareness."""
        meta = {"cache_hit": True}
        meta.update(self._endpoint_awareness())
        return meta

    def _client_cache_key(self, base_url: Optional[str], api_key: Optional[str], timeout: int) -> tuple:
        """Client cache key: bound to the current event loop plus the endpoint parameters."""
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = 0
        return (loop_id, base_url or "", api_key or "", timeout)

    def _get_client(self, base_url: Optional[str], api_key: Optional[str], timeout: int):
        """Get (or lazily build) the reusable AsyncOpenAI client. The same (loop, endpoint) hits the cache with zero connect cost.

        The OpenAI SDK ships default retries (2 in the current version) while this class already has an explicit,
        observable retry loop. Without disabling the SDK's inner retries, one outer attempt configured for 3 seconds
        could really issue 3 HTTP requests, after which HY Memory runs the next attempt, nesting retries and
        amplifying latency. Only this class's retry policy is kept, so attempts, logs and latency budgets match one to one.
        """
        from openai import AsyncOpenAI

        key = self._client_cache_key(base_url, api_key, timeout)
        client = self._client_cache.get(key)
        if client is None:
            client_options = {
                "api_key": api_key,
                "base_url": base_url or None,
                "timeout": timeout,
                "max_retries": 0,
            }
            transport_factory = getattr(self.config, "http_transport_factory", None)
            if callable(transport_factory):
                import httpx

                client_options["http_client"] = httpx.AsyncClient(
                    transport=transport_factory()
                )
            client = AsyncOpenAI(
                **client_options,
            )
            self._client_cache[key] = client
        return client

    def _evict_client(self, base_url: Optional[str], api_key: Optional[str], timeout: int) -> None:
        """Evict a given endpoint's client and close it best-effort (call before retry, avoids reusing a wedged connection)."""
        key = self._client_cache_key(base_url, api_key, timeout)
        client = self._client_cache.pop(key, None)
        if client is not None:
            try:
                _close = getattr(client, "close", None)
                if _close is not None:
                    _res = _close()
                    if asyncio.iscoroutine(_res):
                        asyncio.ensure_future(_res)  # fire-and-forget, does not block the retry
            except Exception:
                pass

    # ----------------------------------------------------------------
    # Write-path batching queue
    # ----------------------------------------------------------------

    async def embed_queued(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Embed for the write path: merges concurrent requests into one batch call via the batching queue.
        Fits the write path (avoids a swarm of concurrent embed requests under heavy write load).
        Use embed() for the search path.
        """
        vector, _meta = await self.embed_queued_with_meta(text, use_cache=use_cache)
        return vector

    async def embed_queued_with_meta(
        self, text: str, use_cache: bool = True
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Embed for the write path and also return provider metadata (the non-vector payload).

        Batching semantics: within one effective endpoint the window closes every queue_batch_window_ms,
        sending everything collected as **one batched HTTP call** (however many there are); a full
        queue_batch_size is sent at once without waiting. Endpoints never interfere (see the bucketing note in __init__).
        """
        # Cache hit returns directly without entering the queue
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached, self._cache_hit_meta()

        # Locate the request bucket for the current effective endpoint; the lock is lazy (must be created inside the event loop)
        bucket = self._queue_bucket_key()
        lock = self._queue_locks.get(bucket)
        if lock is None:
            lock = asyncio.Lock()
            self._queue_locks[bucket] = lock

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        enter_at = time.perf_counter()
        _t_lock_wait = time.perf_counter()

        # Inside the lock **memory operations only**: enqueue, decide whether to flush, detach the batch.
        # The HTTP send must live outside the lock (see _send_batch), otherwise one batch's network time
        # would serialize all concurrent writes.
        batch_to_send = None
        async with lock:
            lock_wait_ms = round((time.perf_counter() - _t_lock_wait) * 1000, 2)
            pending = self._queue_pending.setdefault(bucket, [])
            queue_depth_at_enqueue = len(pending) + 1
            pending.append((text, fut, enter_at, lock_wait_ms, queue_depth_at_enqueue))

            # Batch full: detach it inside the lock, send it outside
            if len(pending) >= self._embed_config.queue_batch_size:
                batch_to_send = self._detach_batch(bucket)
            else:
                # Otherwise start this bucket's periodic flush task (if not already running)
                task = self._queue_flush_tasks.get(bucket)
                if task is None or task.done():
                    self._queue_flush_tasks[bucket] = asyncio.ensure_future(
                        self._delayed_flush(
                            self._embed_config.queue_batch_window_ms / 1000.0,
                            bucket,
                        )
                    )

        # ── lock released ── other coroutines may keep enqueueing, not blocked by this batch's HTTP
        if batch_to_send is not None:
            await self._send_batch(batch_to_send)

        result = await fut
        if isinstance(result, tuple) and len(result) == 2:
            vector, meta = result
        else:
            vector, meta = result, {}

        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            self._cache_put(cache_key, vector)

        self._embed_count += 1
        out_meta = dict(meta or {})
        out_meta["cache_hit"] = False
        return vector, out_meta

    async def _delayed_flush(self, delay: float, bucket: tuple) -> None:
        """Flush one endpoint bucket once its window elapses (the closing mechanism for batches that never fill batch_size).

        The window defaults to 100ms: flush on expiry with whatever is queued. It is pure added latency, so it should stay small;
        production once ran a 1000ms window on top of per-item serial sends and saw queue_wait 6.8s.
        """
        await asyncio.sleep(delay)
        lock = self._queue_locks.get(bucket)
        if lock is None:
            return
        # Same contract: detach inside the lock, send HTTP outside it
        async with lock:
            batch = self._detach_batch(bucket)
            # Key point: release this bucket's timer identity before the HTTP starts. Otherwise requests
            # enqueued while this batch is in flight would see the old task still unfinished and never arm
            # the next timer, so requests below batch_size would wait until later traffic filled them by chance.
            current = asyncio.current_task()
            if self._queue_flush_tasks.get(bucket) is current:
                self._queue_flush_tasks[bucket] = None
        if batch is not None:
            await self._send_batch(batch)

    def _queue_bucket_key(self) -> tuple:
        """Current effective endpoint -> batching bucket key.

        Takes the per-call override from the current contextvar and falls back to the client's own config.
        Endpoints sharing one client therefore get separate buckets: texts from different endpoints
        are never mixed into one batch request (correctness) and never slow each other down (performance).
        """
        _ov = _embed_endpoint_override.get()
        base_url = self._embed_config.base_url
        api_key = self._embed_config.api_key
        if _ov:
            if _ov.get("base_url") is not None:
                base_url = _ov["base_url"]
            if _ov.get("api_key") is not None:
                api_key = _ov["api_key"]
        return (base_url or "", api_key or "")

    def _detach_batch(self, bucket: tuple) -> Optional[List[tuple]]:
        """Detach and clear the whole pending queue of one bucket; returns None when it is empty.

        **Must be called while holding that bucket's lock**, and this function is pure memory work with no await:
        it is one half of the "memory operations only inside the lock" contract, the other being _send_batch outside the lock.
        """
        pending = self._queue_pending.get(bucket)
        if not pending:
            return None
        batch = pending[:]
        pending.clear()
        return batch

    async def _send_batch(self, batch: List[tuple]) -> None:
        """Send an already detached batch and fill in each of its futures.

        **Must be called outside the lock.** An early implementation kept this block (HTTP included) inside the lock, so one
        batch's network time serialized every concurrent write: production measured lock_wait_ms of 34.7s for a single embed and
        S1_EMBED reporting 48s while the real HTTP took only 1.78s. The lock now covers only the few lines of enqueue/detach
        memory operations, fully decoupling lock contention from network latency.

        One HTTP call carries every text in the batch (see _embed_texts_with_meta); no per-item serial sends remain inside a batch.
        """
        if not batch:
            return

        flush_started_at = time.perf_counter()
        texts = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        enter_times = [item[2] if len(item) > 2 else flush_started_at for item in batch]
        lock_waits = [item[3] if len(item) > 3 else 0.0 for item in batch]
        queue_depths = [item[4] if len(item) > 4 else len(batch) for item in batch]
        batch_size = len(texts)
        # The same batching window often holds repeated tags (several memories all carrying preferences).
        # The provider computes it once and the result fans back out to the original positions, saving tokens and queue pressure.
        unique_texts = list(dict.fromkeys(texts))

        logger.debug(f"[embed_queued] flushing batch of {batch_size}")
        try:
            unique_vectors, unique_metas = await self._embed_texts_with_meta(unique_texts)
            batch_flush_ms = round((time.perf_counter() - flush_started_at) * 1000, 2)

            # In batch mode the provider must return exactly one vector per input item. On a count mismatch we
            # must not silently truncate with zip -- that would leave the extra futures **hung forever** and the caller
            # (the write pipeline) blocked until something upstream timed out. Detect it explicitly and set an
            # exception on the remaining futures, surfacing the failure immediately.
            if len(unique_vectors) != len(unique_texts):
                raise RuntimeError(
                    f"embed batch size mismatch: requested {len(unique_texts)} unique texts, "
                    f"provider returned {len(unique_vectors)} vectors"
                )

            by_text = {
                text: (unique_vectors[i], unique_metas[i])
                for i, text in enumerate(unique_texts)
            }

            for idx, (text, fut) in enumerate(zip(texts, futures)):
                vec, meta = by_text[text]
                enriched_meta = dict(meta or {})
                enriched_meta.update({
                    "batch_size": batch_size,
                    "batch_unique_count": len(unique_texts),
                    "position_in_batch": idx + 1,
                    "queue_depth": queue_depths[idx],
                    "queue_wait_ms": round((flush_started_at - enter_times[idx]) * 1000, 2),
                    "lock_wait_ms": lock_waits[idx],
                    "batch_flush_ms": batch_flush_ms,
                    "queue_batch_size_config": self._embed_config.queue_batch_size,
                    "queue_batch_window_ms": self._embed_config.queue_batch_window_ms,
                })
                if not fut.done():
                    fut.set_result((vec, enriched_meta))
        except Exception as e:
            logger.error(f"[embed_queued] batch failed: {e}")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
        finally:
            # Fallback: whether it succeeded or raised, never allow a future to hang -- otherwise the write pipeline
            # would await forever. The normal path never reaches this branch.
            for fut in futures:
                if not fut.done():
                    fut.set_exception(
                        RuntimeError("embed batch future left unresolved")
                    )
    
    async def embed(
        self,
        text: str,
        use_cache: bool = True,
    ) -> List[float]:
        """
        Embed a single text
        
        Args:
            text: the text content
            use_cache: whether to use the cache
        
        Returns:
            The vector list
        """
        # Check the cache
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                logger.debug(f"[embed] cache hit (total hits={self._cache_hits})")
                return cached

        # Call the embedding service
        vectors = await self._embed_texts([text])
        vector = vectors[0]
        
        # Update the cache
        if use_cache and self._embed_config.enable_cache:
            self._cache_put(cache_key, vector)
        
        self._embed_count += 1
        return vector
    
    async def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        Embed a batch of texts
        
        Args:
            texts: the list of texts
            use_cache: whether to use the cache
        
        Returns:
            The vector list
        """
        vectors, _metas = await self.embed_batch_with_meta(texts, use_cache=use_cache)
        return vectors

    async def embed_batch_with_meta(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """Embed a batch and also return each item's provider metadata (the non-vector part)."""
        if not texts:
            return [], []

        results: List[Any] = [None] * len(texts)
        metas: List[Dict[str, Any]] = [{} for _ in texts]
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            if use_cache and self._embed_config.enable_cache:
                cache_key = self._get_cache_key(text)
                cached = self._cache_get(cache_key)
                if cached is not None:
                    results[i] = cached
                    metas[i] = self._cache_hit_meta()
                    self._cache_hits += 1
                    continue

            texts_to_embed.append(text)
            indices_to_embed.append(i)

        if texts_to_embed:
            vectors, batch_metas = await self._embed_texts_with_meta(texts_to_embed)

            for j, vector in enumerate(vectors):
                idx = indices_to_embed[j]
                results[idx] = vector
                m = dict(batch_metas[j] if j < len(batch_metas) else {})
                m["cache_hit"] = False
                metas[idx] = m

                if use_cache and self._embed_config.enable_cache:
                    cache_key = self._get_cache_key(texts_to_embed[j])
                    self._cache_put(cache_key, vector)

            self._embed_count += len(texts_to_embed)

        return results, metas
    
    async def _embed_texts_with_meta(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """The real embedding call: **one HTTP request carries the whole batch**; returns (vectors, metas).

        The early implementation was `for text in texts: await _embed_openai([text])`, one item at a time.
        The intent was to dodge batch size limits, but the cost was huge: in production a batch of 8 items at ~5.6s each
        accumulated to batch_flush_ms=44.7s, throwing away every benefit of batching and then some
        (S1_EMBED reported 51.7s, where a real single HTTP call was only 2.8s).

        The OpenAI embedding protocol natively accepts `input` as an array and returns all vectors in one round trip,
        so we send the batch directly: latency drops from the sum over items to a single round trip.

        Handling of metas: a batch produces a single response meta (http_status / traceId / usage are
        request-level); it is copied onto every item with `batch_mode`/`batch_index` added to record provenance,
        keeping the existing per-item meta contract intact (callers still index by position).

        Truncation happens here (each text passes `_truncate_for_embedding` before sending): every embedding
        path (embed / embed_batch / embed_queued / *_with_meta) ultimately lands in this method -- putting
        it at the entry points would miss some. Rationale and threshold: see `_EMBED_MAX_TOKENS` at the top of this module.
        """
        if not texts:
            return [], []

        safe_texts = [_truncate_for_embedding(t) for t in texts]
        vectors, meta = await self._embed_openai(safe_texts)
        base_meta = dict(meta or {})
        base_meta["batch_mode"] = "single_request"
        base_meta["batch_input_count"] = len(safe_texts)

        all_metas: List[Dict[str, Any]] = []
        for idx in range(len(safe_texts)):
            m = dict(base_meta)
            m["batch_index"] = idx
            all_metas.append(m)

        self._last_embed_meta = all_metas[-1] if all_metas else {}
        return vectors, all_metas

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """The real embedding call (always via the OpenAI-compatible protocol), one item at a time to dodge batch size limits"""
        vectors, _metas = await self._embed_texts_with_meta(texts)
        return vectors
    
    def _log_embed_call(
        self,
        *,
        status: str,
        api_url: str,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> None:
        """
        Emit one INFO-level embedding request log (request_id / provider / model / api / request / response).

        The response part carries only vector dimensions, never vector values. The request_id is also written into the
        message and relies on the log formatter's %(request_id)s (effective once setup_logging has run).
        """
        payload = {
            "request_id": get_request_id(),
            "status": status,
            "provider": self._embed_config.provider,
            "model": self._embed_config.model,
            "api": api_url,
            "request": request_payload,
            "response": response_payload,
        }
        try:
            body = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            body = str(payload)
        logger.info(f"[embed] {body}")

    async def _embed_openai(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Embed using the OpenAI-compatible API

        Supports every OpenAI-compatible embedding service:
        - OpenAI official
        - Alibaba Cloud DashScope (text-embedding-v3)
        - DeepSeek
        - Local vLLM / Ollama and the like

        Request parameters: model + input (+ dimensions) + extra_headers + extra_body.
        When `embedding_dims` is empty or 0, dimensions is not sent.

        Returns:
            (vectors, meta): meta is the response headers plus the non-vector body fields
            (http_status / model / usage / attempts and so on). It is also written to
            self._last_embed_meta (racy under concurrent calls; callers should use the returned meta).
        """
        if importlib.util.find_spec("openai") is None:
            raise ImportError("openai is required. Install with: pip install openai")

        # Only retry the errors below ("suspected network / server-side flakiness"); business errors (400/401 etc.) raise immediately.
        # String matching avoids importing openai's concrete exception types (their names move between versions).
        _RETRYABLE = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            # HTTP 200 with an empty `data` list is still "no usable response".
            # The gateway has returned this transiently for longer episode texts;
            # treating it as a permanent ValueError wastes the preceding summary
            # generation and leaves the session without L3.
            ValueError,
            # Some OpenAI-compatible gateways return an empty/non-JSON body for
            # a transient upstream failure.
            # openai surfaces that as JSONDecodeError without an HTTP status.
            JSONDecodeError,
        )
        # Gateway flakiness is judged by HTTP status code, not by keywords in the message.
        #
        # The status code comes from the exception instance (APIStatusError instances carry .status_code, the class does not,
        # hence getattr instead of isinstance, which also avoids importing concrete exception classes).
        #   5xx        server / gateway side, retry is meaningful (502/503/504 were observed here)
        #   408 / 429  request timeout and rate limiting, same retry semantics
        #   other 4xx  business errors (400 bad input, 401 auth); retrying only amplifies the failure, raise at once
        _RETRYABLE_STATUS = {408, 429}

        kwargs: Dict[str, Any] = {
            "model": self._embed_config.model,
            "input": texts,
        }
        if self._embed_config.embedding_dims:
            kwargs["dimensions"] = self._embed_config.embedding_dims

        # ── Per-call endpoint override (contextvar) ──────────────────────
        # Only the access endpoint is swapped (base_url/api_key/extra_headers/timeout); model/dims stay.
        # The override is not written into self._embed_config and applies to this request only, so the shared instance is not polluted.
        _ov = _embed_endpoint_override.get()
        _eff_base_url = self._embed_config.base_url
        _eff_api_key = self._embed_config.api_key
        _eff_timeout = self._embed_config.timeout or 15
        _eff_extra_headers = self._embed_config.extra_headers
        if _ov:
            if _ov.get("base_url") is not None:
                _eff_base_url = _ov["base_url"]
            if _ov.get("api_key") is not None:
                _eff_api_key = _ov["api_key"]
            if _ov.get("timeout") is not None:
                _eff_timeout = _ov["timeout"]
            if _ov.get("extra_headers") is not None:
                _eff_extra_headers = _ov["extra_headers"]

        if _eff_extra_headers:
            kwargs["extra_headers"] = _eff_extra_headers
        if self._embed_config.extra_body:
            kwargs["extra_body"] = self._embed_config.extra_body

        api_url = _embeddings_api_url(_eff_base_url)
        # Log the complete request payload (without api_key); auth fields in headers are redacted
        request_payload: Dict[str, Any] = {
            "model": kwargs.get("model"),
            "input": texts,
            "input_count": len(texts),
            "dimensions": kwargs.get("dimensions"),
            "extra_body": self._embed_config.extra_body,
            "extra_headers": _redact_headers(_eff_extra_headers),
            "timeout": _eff_timeout,
            "endpoint_overridden": bool(_ov),
        }

        _max_retries = max(0, int(self._embed_config.max_retries or 0))
        _retry_delay = max(0.0, float(self._embed_config.retry_delay or 0.0))
        last_exc: Optional[Exception] = None

        for attempt in range(_max_retries + 1):
            # Reuse the client (cached per loop + endpoint): constructing AsyncOpenAI eagerly builds an SSLContext +
            # httpx pool (tens of ms), and rebuilding it every time is the main source of client_build_ms. Reusing it means
            # only the first request pays the connect cost and later ones use the pool. client_build_ms / embed_request_ms
            # are measured separately, to tell "connection setup" apart from "embedding service handling + network round trip".
            # A retryable failure in the previous attempt already evicted the client, so a miss here builds a clean one.
            _t_client = time.perf_counter()
            client = self._get_client(_eff_base_url, _eff_api_key, _eff_timeout)
            _client_build_ms = round((time.perf_counter() - _t_client) * 1000, 2)
            try:
                # Use with_raw_response to capture HTTP headers (provider traceId etc.)
                # embed_request_ms brackets only this one HTTP await: the connection (a first-use client handshakes here)
                # plus request send, server-side handling and response receive. It already excludes upstream loop scheduling,
                # cache lookups, the asyncio.wait_for wrapper and other SDK-peripheral costs.
                _t_req = time.perf_counter()
                async with embedding_slot():
                    raw = await client.embeddings.with_raw_response.create(**kwargs)
                _embed_request_ms = round((time.perf_counter() - _t_req) * 1000, 2)
                response = raw.parse()

                # Store metadata for trace logging (headers + body fields except vectors)
                resp_headers = {
                    k: v for k, v in raw.headers.items()
                    if k.lower() not in _SENSITIVE_HEADER_KEYS
                }
                # Sort by index before reading vectors: the OpenAI embedding protocol gives each item an
                # index (its position in the input array) but **never guarantees the data array itself is ordered**.
                # For batch requests (several texts at once), reading in data order could pair vectors with the wrong texts.
                # For a single-item request the sort is a no-op with no extra cost.
                _data = sorted(
                    response.data,
                    key=lambda it: getattr(it, "index", 0) or 0,
                )
                vectors = [item.embedding for item in _data]
                if len(vectors) != len(texts) or any(not vector for vector in vectors):
                    raise ValueError(
                        "No embedding data received"
                        if not vectors else
                        f"Embedding count mismatch: expected={len(texts)} "
                        f"got={len(vectors)}"
                    )
                dims = [len(v) for v in vectors]
                meta = {
                    "http_status": raw.status_code,
                    "response_headers": resp_headers,
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "attempts": attempt + 1,
                    # Client-side self-measured timings (the SDK cannot see the peer service; pure HTTP accounting):
                    "embed_request_ms": _embed_request_ms,   # this embedding HTTP round trip (connection setup + server handling)
                    "client_build_ms": _client_build_ms,     # cost of constructing a new AsyncOpenAI
                    # provider / endpoint actually used for this call (flows into S1_EMBED / DEDUP / search embed_meta,
                    # letting the inspector tell a gateway call from a per-call override direct connection):
                    "provider": self._embed_config.provider,
                    "endpoint": api_url,                       # actual request URL (a direct ip:port once overridden)
                    "endpoint_overridden": bool(_ov),          # whether a per-call endpoint override was used
                }
                self._last_embed_meta = meta
                # INFO: the complete embed request chain (provider/model/api/request params/output dims)
                self._log_embed_call(
                    status="ok",
                    api_url=api_url,
                    request_payload=request_payload,
                    response_payload={
                        "http_status": raw.status_code,
                        "model": response.model,
                        "usage": meta.get("usage"),
                        "attempts": attempt + 1,
                        "vector_count": len(vectors),
                        "vector_dims": dims,
                        "embed_request_ms": _embed_request_ms,
                        "client_build_ms": _client_build_ms,
                        # Log dimensions only, never the vector itself
                        "response_headers": resp_headers,
                    },
                )
                return vectors, meta
            except Exception as e:
                last_exc = e
                _err_name = type(e).__name__
                _err_msg = str(e)
                # Retryability decision: HTTP status code, explicit exception type, or message keywords.
                # The status code wins -- it is the only one of the three unaffected by how the message is worded.
                _status = getattr(e, "status_code", None)
                _retryable = (
                    (isinstance(_status, int)
                     and (_status >= 500 or _status in _RETRYABLE_STATUS))
                    or isinstance(e, _RETRYABLE)
                    or any(
                        kw in _err_msg.lower()
                        for kw in ("timeout", "timed out", "connection", "read operation")
                    )
                )
                meta = {
                    "error": _err_name,
                    "error_msg": _err_msg[:300],
                    "attempts": attempt + 1,
                    "retryable": _retryable,
                    # The status code gets its own field: chasing a 504 used to mean fishing HTML out of error_msg,
                    # while the error field was always InternalServerError, indistinguishable between 502/503/504.
                    "http_status": _status if isinstance(_status, int) else None,
                    # Endpoint awareness: attached on failures too, to pinpoint which endpoint (gateway or direct) misbehaved
                    "provider": self._embed_config.provider,
                    "endpoint": api_url,
                    "endpoint_overridden": bool(_ov),
                }
                self._last_embed_meta = meta
                if _retryable and attempt < _max_retries:
                    # Evict the possibly wedged client so the next attempt builds a clean connection (restores the original defence)
                    self._evict_client(_eff_base_url, _eff_api_key, _eff_timeout)
                    logger.warning(
                        f"[embed] attempt {attempt + 1}/{_max_retries + 1} failed "
                        f"({_err_name}): {_err_msg[:200]}; retry in {_retry_delay}s"
                    )
                    if _retry_delay > 0:
                        await asyncio.sleep(_retry_delay)
                    continue
                self._log_embed_call(
                    status="error",
                    api_url=api_url,
                    request_payload=request_payload,
                    response_payload={
                        "attempts": attempt + 1,
                        "error": _err_name,
                        "error_msg": _err_msg[:500],
                        "retryable": _retryable,
                        "http_status": _status if isinstance(_status, int) else None,
                    },
                )
                raise

        # Theoretically unreachable; re-raise as a fallback
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("embed_openai: unreachable state")

    async def embed_with_meta(
        self,
        text: str,
        use_cache: bool = True,
    ) -> tuple:
        """
        Embed a single text and also return provider metadata (including the Venus traceId and so on).

        Concurrency safe: meta comes from the _embed_openai return value, not from self._last_embed_meta.

        Returns:
            (vector, meta_dict)
            meta_dict contains http_status, response_headers, model, usage and so on.
            On a cache hit, meta_dict is {"cache_hit": True}.
        """
        # Check cache
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached, self._cache_hit_meta()

        vectors, meta = await self._embed_openai([_truncate_for_embedding(text)])
        vector = vectors[0]

        # Update cache
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            self._cache_put(cache_key, vector)

        self._embed_count += 1
        out_meta = dict(meta or {})
        out_meta["cache_hit"] = False
        return vector, out_meta


    async def embed_with_cache_info(
        self,
        text: str,
        use_cache: bool = True,
    ) -> tuple:
        """
        Embed a single text and report whether the cache was hit
        
        Args:
            text: the text content
            use_cache: whether to use the cache
        
        Returns:
            (vector list, whether the cache was hit)
        """
        # Check the cache
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached, True
        
        # Call the embedding service
        vectors = await self._embed_texts([text])
        vector = vectors[0]
        
        # Update the cache
        if use_cache and self._embed_config.enable_cache:
            self._cache_put(cache_key, vector)
        
        self._embed_count += 1
        return vector, False
    
    def get_embedding_dims(self) -> int:
        """Get the vector dimensions"""
        return self._embed_config.embedding_dims
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("EmbedService cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        cache_size = self._cache_len()
        return {
            "embed_count": self._embed_count,
            "cache_hits": self._cache_hits,
            "cache_size": cache_size,
            "cache_hit_rate": (
                self._cache_hits / (self._embed_count + self._cache_hits)
                if (self._embed_count + self._cache_hits) > 0 else 0
            ),
            "model": self._embed_config.model,
            "embedding_dims": self._embed_config.embedding_dims,
        }

    def _cache_len(self) -> int:
        with self._cache_lock:
            return len(self._cache)
