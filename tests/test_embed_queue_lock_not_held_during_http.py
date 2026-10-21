# -*- coding: utf-8 -*-
"""Embed batching queue: HTTP must run outside _queue_lock.

Regression background (production incident 2026-08-20):
Early `_flush_queue` awaited HTTP while still holding `_queue_lock`. Because
`_embed_texts_with_meta` sent one request per item, a batch of 8 x ~1.7s ~= 13.5s held the lock throughout,
so later `embed_queued` calls all piled up on `async with self._queue_lock` at the enqueue point.
Measured in production: a single embed showed `lock_wait_ms` of 34.7s and the `S1_EMBED` step reported 48s,
while the real HTTP took only 1.78s -- the alert wrongly read this as "Embed timeout".

Fix: the lock covers in-memory work only (enqueue / `_detach_batch`); `_send_batch` is called outside the lock.
This test pins that contract so it cannot regress.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.core.embed_service import EmbedService


HTTP_MS = 0.05  # Simulated per-item HTTP time (real ~1.7s, scaled down to keep the test fast)


def _make_service(*, batch_size: int = 8, window_ms: float = 1000.0) -> EmbedService:
    cfg = MemoryConfig.from_env()
    cfg.embedder.base_url = "http://fake/v1"
    cfg.embedder.api_key = "K"
    cfg.embedder.embedding_dims = None
    svc = EmbedService(cfg)
    svc._embed_config.queue_batch_size = batch_size
    svc._embed_config.queue_batch_window_ms = window_ms
    svc._embed_config.enable_cache = False  # Force every item through the queue
    return svc


class _FakeRaw:
    """Simulates a provider response: returns as many vectors as there are input items, each with an index.

    Under the real OpenAI embedding protocol every data item carries an index (the input position)
    and the count matches the input. The indexes are deliberately reversed here to check that the SDK sorts by index
    instead of relying on the physical order of data.
    """

    status_code = 200
    headers: dict = {}

    def __init__(self, input_count: int = 1):
        self._input_count = max(1, input_count)

    def parse(self):
        resp = MagicMock()
        items = []
        for i in range(self._input_count):
            item = MagicMock()
            # The first element of item i's vector is i, so the order can be asserted
            item.embedding = [float(i), 0.1, 0.2, 0.3]
            item.index = i
            items.append(item)
        resp.data = list(reversed(items))   # Deliberately reversed to exercise sorting by index
        resp.model = "fake-model"
        resp.usage = None
        return resp


def _slow_client_factory(**_kwargs):
    """embeddings.create sleeps HTTP_MS seconds to simulate one batch request round trip."""

    class _Embeddings:
        class with_raw_response:
            @staticmethod
            async def create(**kwargs):
                await asyncio.sleep(HTTP_MS)
                texts = kwargs.get("input") or []
                return _FakeRaw(input_count=len(texts) if isinstance(texts, list) else 1)

    client = MagicMock()
    client.embeddings = _Embeddings()
    return client


class TestEmbedQueueLockNotHeldDuringHttp:
    """The lock must never span HTTP: lock_wait_ms under concurrent writes has to stay far below one batch of HTTP time."""

    @pytest.mark.asyncio
    async def test_lock_wait_far_below_batch_http_time(self):
        batch_size = 8
        svc = _make_service(batch_size=batch_size)
        n = batch_size * 2  # Two batches

        with patch("openai.AsyncOpenAI", side_effect=_slow_client_factory):
            results = await asyncio.gather(
                *[svc.embed_queued_with_meta(f"text-{i}") for i in range(n)]
            )

        assert len(results) == n
        assert all(len(vec) == 4 for vec, _ in results)

        lock_waits = [meta["lock_wait_ms"] for _, meta in results]
        one_batch_http_ms = HTTP_MS * 1000 * batch_size

        # If HTTP were still inside the lock, latecomers' lock_wait would approach a full batch of HTTP time.
        assert max(lock_waits) < one_batch_http_ms * 0.25, (
            f"lock_wait_ms max={max(lock_waits):.1f} is close to one batch of HTTP "
            f"{one_batch_http_ms:.0f}ms -- HTTP may have moved back inside the lock"
        )

    @pytest.mark.asyncio
    async def test_detach_batch_is_pure_memory_op(self):
        """_detach_batch must be a synchronous function (no await), so the lock contains no I/O."""
        svc = _make_service()
        assert not asyncio.iscoroutinefunction(svc._detach_batch)

        bucket = ("http://fake/v1", "K")
        svc._queue_pending[bucket] = [("a", MagicMock(), 0.0, 0.0, 1)]
        batch = svc._detach_batch(bucket)
        assert batch is not None and len(batch) == 1
        assert svc._queue_pending[bucket] == []       # Emptied once the batch is detached
        assert svc._detach_batch(bucket) is None      # An empty queue returns None
        assert svc._detach_batch(("other", "x")) is None  # An unknown bucket is safe too


class TestEmbedQueueBehaviourPreserved:
    """The fix must not break the existing batching semantics."""

    @pytest.mark.asyncio
    async def test_timer_flush_still_works_when_batch_not_full(self):
        """When batch_size is never reached the timed flush still has to act as fallback (otherwise requests hang forever)."""
        window_ms = 200.0
        svc = _make_service(batch_size=8, window_ms=window_ms)

        started = time.perf_counter()
        with patch("openai.AsyncOpenAI", side_effect=_slow_client_factory):
            results = await asyncio.wait_for(
                asyncio.gather(*[svc.embed_queued_with_meta(f"t{i}") for i in range(3)]),
                timeout=5.0,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert all(len(vec) == 4 for vec, _ in results)
        assert results[0][1]["batch_size"] == 3      # Batched by the actual item count
        assert elapsed_ms >= window_ms * 0.8         # It really waited for the timer window

    @pytest.mark.asyncio
    async def test_http_failure_propagates_to_all_futures(self):
        """When a batch fails, every future must receive the exception instead of hanging (otherwise write deadlocks)."""
        batch_size = 4
        svc = _make_service(batch_size=batch_size, window_ms=200.0)

        def _failing_client(**_kwargs):
            class _Embeddings:
                class with_raw_response:
                    @staticmethod
                    async def create(**_k):
                        raise ValueError("boom")

            client = MagicMock()
            client.embeddings = _Embeddings()
            return client

        with patch("openai.AsyncOpenAI", side_effect=_failing_client):
            results = await asyncio.gather(
                *[svc.embed_queued_with_meta(f"bad-{i}") for i in range(batch_size)],
                return_exceptions=True,
            )

        assert len(results) == batch_size
        assert all(isinstance(r, Exception) for r in results)

    @pytest.mark.asyncio
    async def test_queue_diagnostics_fields_present(self):
        """provider_meta must still carry every queue diagnostic field (production debugging depends on them)."""
        svc = _make_service(batch_size=2, window_ms=200.0)

        with patch("openai.AsyncOpenAI", side_effect=_slow_client_factory):
            (_vec, meta), _ = await asyncio.gather(
                svc.embed_queued_with_meta("x1"),
                svc.embed_queued_with_meta("x2"),
            )

        for key in (
            "batch_size",
            "position_in_batch",
            "queue_depth",
            "queue_wait_ms",
            "lock_wait_ms",
            "batch_flush_ms",
            "queue_batch_size_config",
            "queue_batch_window_ms",
            "embed_request_ms",
        ):
            assert key in meta, f"provider_meta is missing the diagnostic field {key}"
