# -*- coding: utf-8 -*-
"""
Regression: list_by_user must filter by layers; evolution_chain must not expose L1_RAW.

Background bug (0.1.5.30):
- the faiss backend's list_by_user once accepted a layers parameter but did not filter, causing
  upsert_basic_profile._find_latest_l0(layers=[L0]) to get cross-layer nodes (raw),
  so a new L0 node wrongly superseded raw -> the search's evolution_chain leaked raw.
"""

import asyncio
import uuid

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


DIMS = 32


def _node(user_id, layer, content="x", is_latest=True):
    import random
    n = MemoryNode(
        node_id=str(uuid.uuid4()),
        user_id=user_id,
        agent_id="a1",
        session_id="s1",
        content=content,
        layer=layer,
        status=MemoryStatus.ACTIVE,
        is_latest=is_latest,
    )
    vec = [random.gauss(0, 1) for _ in range(DIMS)]
    norm = sum(v * v for v in vec) ** 0.5
    n.embedding = [v / norm for v in vec]
    return n


@pytest.fixture
def faiss_store(tmp_path):
    config = MemoryConfig.from_dict({
        "vector_store": {
            "provider": "faiss",
            "collection_name": "test_layer_filter",
            "persist_directory": str(tmp_path / "faiss_data"),
            "embedding_dims": DIMS,
        },
    })
    from hy_memory.data.vector_store_faiss import FaissVectorStore
    vs = FaissVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


class TestListByUserLayerFilter:
    def test_faiss_list_by_user_filters_layers(self, faiss_store):
        u = "user_layer_test"
        run(faiss_store.upsert_batch([
            _node(u, MemoryLayer.L1_RAW, "raw dialogue"),
            _node(u, MemoryLayer.L0_BASIC_INFO, "The user's name is ivanye."),
            _node(u, MemoryLayer.L2_FACT, "likes badminton"),
        ]))

        # only L0: raw must never sneak in
        l0 = run(faiss_store.list_by_user(user_id=u, layers=[MemoryLayer.L0_BASIC_INFO]))
        assert len(l0) == 1
        assert l0[0].layer == MemoryLayer.L0_BASIC_INFO

        # multiple layers
        multi = run(faiss_store.list_by_user(
            user_id=u, layers=[MemoryLayer.L0_BASIC_INFO, MemoryLayer.L2_FACT]
        ))
        got = {n.layer for n in multi}
        assert MemoryLayer.L1_RAW not in got
        assert got == {MemoryLayer.L0_BASIC_INFO, MemoryLayer.L2_FACT}

        # no layers passed: return everything except raw (raw is invisible to external read paths)
        all_nodes = run(faiss_store.list_by_user(user_id=u))
        assert len(all_nodes) == 2
        assert MemoryLayer.L1_RAW not in {n.layer for n in all_nodes}

        # naming only raw: gets nothing (not "caller decides", it is uniformly blocked)
        only_raw = run(faiss_store.list_by_user(user_id=u, layers=[MemoryLayer.L1_RAW]))
        assert only_raw == []

        # only internal pipelines that opt in explicitly can get it (writer's dialogue-history assembly relies on this path)
        internal = run(faiss_store.list_by_user(
            user_id=u, layers=[MemoryLayer.L1_RAW], allow_raw_layer=True,
        ))
        assert len(internal) == 1
        assert internal[0].layer == MemoryLayer.L1_RAW

        # opted in but no layers passed: back to the "all layers" semantics (clone relies on it)
        internal_all = run(faiss_store.list_by_user(user_id=u, allow_raw_layer=True))
        assert len(internal_all) == 3


class TestEvolutionChainExcludesRaw:
    def test_chain_drops_raw_nodes(self):
        from hy_memory.pipelines._retrieval.evolution import _expand_one_chain

        raw = _node("u", MemoryLayer.L1_RAW, "You are an AI capability platform", is_latest=False)
        head = _node("u", MemoryLayer.L0_BASIC_INFO, "The user's name is ivanye.", is_latest=True)
        head.supersedes = [raw.node_id]
        raw.superseded_by = [head.node_id]

        class _FakeVS:
            async def get_by_ids(self, ids):
                m = {raw.node_id: raw, head.node_id: head}
                return [m[i] for i in ids if i in m]

        exp = run(_expand_one_chain(_FakeVS(), {"node": head, "score": 0.9}))
        chain = exp.get("evolution_chain", [])
        # the chain head is kept, raw is filtered out
        layers = [c["layer"] for c in chain]
        assert "l1_raw" not in layers
        assert any(c["node_id"] == head.node_id for c in chain)
        assert all(c["node_id"] != raw.node_id for c in chain)


class TestFindLatestL0Validates:
    """_find_latest_l0 must accept only valid L0 chain heads (layer + kv structure); dirty nodes are not treated as chain heads."""

    def test_raw_node_not_treated_as_l0_head(self):
        from hy_memory.agent.tools.basic_profile import _find_latest_l0, _is_valid_l0_head

        # an is_latest raw node (no basic_info_kv), must not be treated as the L0 chain head
        raw = _node("u", MemoryLayer.L1_RAW, "You are an AI capability platform", is_latest=True)
        assert _is_valid_l0_head(raw) is False

        # a valid L0 node
        l0 = _node("u", MemoryLayer.L0_BASIC_INFO, "The user's name is ivanye.", is_latest=True)
        l0.custom = {"basic_info_kv": {"name": "ivanye"}}
        assert _is_valid_l0_head(l0) is True

        class _FakeVS:
            def __init__(self, nodes):
                self._nodes = nodes
            async def list_by_user(self, **kwargs):
                # simulate a backend with layer filtering broken: return raw too
                return self._nodes
            async def get_by_id(self, _id):
                return None

        # the store has only a raw node (broken layers filter) -> cannot be a chain head, return None
        head = run(_find_latest_l0(_FakeVS([raw]), "u"))
        assert head is None

        # the store has raw + a valid L0 -> must pick out L0, never return raw
        head2 = run(_find_latest_l0(_FakeVS([raw, l0]), "u"))
        assert head2 is not None
        assert head2.node_id == l0.node_id
