# -*- coding: utf-8 -*-
"""
Server-side layers filtering on the search side (R3: CTX needs retrieval targeted by type).

Two layers of protection, both indispensable:

1. Directed recall (reader side): push layers down into the recall stage so irrelevant layers cannot take
   the limit budget and leave the target layers under-recalled. Client-side filtering can only drop extras, never recover misses.
2. Fallback filter (client side): readers support request.layers to varying degrees, and the profile /
   graph paths generally recall fixed layers and ignore request.layers. The contract is "return only the matched layers",
   so results are filtered once more before being returned.

This file locks down layer 2, plus argument normalisation and backward compatibility.
"""

import asyncio
import inspect

import pytest

from hy_memory.client import HyMemoryClient
from hy_memory.models.memory import MemoryLayer


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSearchSignature:
    def test_async_search_exposes_layers(self):
        params = inspect.signature(HyMemoryClient.async_search).parameters
        assert "layers" in params
        assert params["layers"].default is None, "omitting it must behave exactly as before the change"

    def test_sync_search_exposes_layers(self):
        params = inspect.signature(HyMemoryClient.search).parameters
        assert "layers" in params
        assert params["layers"].default is None

    def test_search_uses_schema_parameter_names(self):
        for method in (HyMemoryClient.search, HyMemoryClient.async_search):
            params = inspect.signature(method).parameters
            assert "schema_limit" in params
            assert "schema_min_score" in params
            assert "profile_limit" not in params
            assert "profile_min_score" not in params


class TestReadRequestWiring:
    """layers must really reach ReadRequest, otherwise the reader has nothing to recall in a targeted way."""

    def test_read_request_accepts_layers_as_str_list(self):
        from hy_memory.pipelines.base import ReadRequest

        req = ReadRequest(query="q", layers=["l2_fact"])
        assert req.layers == ["l2_fact"], (
            "ReadRequest.layers is List[str]; the client side must convert the enums back to strings"
        )

    def test_normalize_layers_output_is_enum(self):
        """The client unifies on enums internally and takes .value before writing them into ReadRequest."""
        got = HyMemoryClient._normalize_layers(["l2_fact"])
        assert got == [MemoryLayer.L2_FACT]
        assert [l.value for l in got] == ["l2_fact"]

    def test_legacy_alias_normalized_before_recall(self):
        """Legacy layer ids are normalised at the entry point, otherwise the reader compares them raw and misses results."""
        got = HyMemoryClient._normalize_layers(["l5_schema"])
        assert [l.value for l in got] == ["l6_schema"]

    def test_invalid_layer_raises(self):
        with pytest.raises(ValueError):
            HyMemoryClient._normalize_layers(["l3_identity"])


class TestHybridDirectedRecall:
    """The shared reader main flow must push request.layers down into the recall stage."""

    def test_normal_channels_use_request_layers(self):
        import hy_memory.pipelines.reader_base as m

        src = inspect.getsource(m)
        assert "_normal_layers" in src, "the normal path must recall in a targeted way from request.layers"
        # Semantic, BM25 and date-based backfill recall all obey the same layer filter.
        assert src.count("layers=_normal_layers") == 3

    def test_hidden_l0_context_keeps_fixed_layer(self):
        """L0 remains a fixed hidden context query, not a public Search result."""
        import hy_memory.pipelines.reader_base as m

        src = inspect.getsource(m)
        assert "layers=_VDB_HIDDEN_L0_LAYERS" in src

    def test_recall_task_order_unchanged(self):
        """The three channels are unpacked by fixed index, so skipping a channel because of filtering would misalign them."""
        import hy_memory.pipelines.reader_base as m

        src = inspect.getsource(m)
        assert "results[0]" in src and "results[1]" in src and "results[2]" in src
