# -*- coding: utf-8 -*-
"""
list returns evolution chains (R4: the update history of a structured fact must be queryable and displayable).

search already returned evolution_chain while list did not. This reuses the same expand_evolution_chains
so both sides share one structure (the frontend can render them identically).

This file pins down three things:
1. Off by default -- expansion re-reads chain by chain and must not slow down the default list.
2. When enabled it only attaches fields and never adds or drops items. expand_evolution_chains
   is built for search and dedups within a chain; list taking it as-is would lose rows vs total.
3. Only the current page is expanded, not the whole table.
"""

import asyncio
import uuid

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.client import HyMemoryClient
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


DIMS = 32


def _node(content, *, node_id=None, is_latest=True, status=MemoryStatus.ACTIVE,
          supersedes=None, superseded_by=None, user_id="u1"):
    import random
    n = MemoryNode(
        node_id=node_id or str(uuid.uuid4()),
        user_id=user_id,
        agent_id="a1",
        session_id="s1",
        content=content,
        layer=MemoryLayer.L2_FACT,
        status=status,
        is_latest=is_latest,
    )
    n.supersedes = supersedes or []
    n.superseded_by = superseded_by or []
    vec = [random.gauss(0, 1) for _ in range(DIMS)]
    norm = sum(v * v for v in vec) ** 0.5
    n.embedding = [v / norm for v in vec]
    return n


@pytest.fixture
def faiss_store(tmp_path):
    config = MemoryConfig.from_dict({
        "vector_store": {
            "provider": "faiss",
            "collection_name": "test_evolution",
            "persist_directory": str(tmp_path / "faiss_data"),
            "embedding_dims": DIMS,
        },
    })
    from hy_memory.data.vector_store_faiss import FaissVectorStore
    vs = FaissVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


def _client_with(store):
    c = HyMemoryClient.__new__(HyMemoryClient)
    c._vector_store = store
    return c


def _seed_chain(store):
    """One two-node evolution chain plus one memory that has no chain.

    The head (ACTIVE/latest) supersedes old (SUPERSEDED), matching the structure the
    writer produces for a contradiction-supersede decision.
    """
    old_id, head_id = str(uuid.uuid4()), str(uuid.uuid4())
    run(store.upsert_batch([
        _node("v1 used Vue", node_id=old_id, is_latest=False,
              status=MemoryStatus.SUPERSEDED, superseded_by=[head_id]),
        _node("v2 switched to React", node_id=head_id, supersedes=[old_id]),
        _node("Standalone memory"),
    ]))
    return old_id, head_id


class TestDefaultOff:
    def test_no_chain_field_by_default(self, faiss_store):
        _seed_chain(faiss_store)
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
        ))
        assert all("evolution_chain" not in m for m in got["memories"]), (
            "the default must not return evolution chains: expansion costs re-reads and slows the default list"
        )


class TestWithEvolution:
    def test_chain_attached_to_head(self, faiss_store):
        old_id, head_id = _seed_chain(faiss_store)
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
            with_evolution=True,
        ))

        by_id = {m["memory_id"]: m for m in got["memories"]}
        chain = by_id[head_id].get("evolution_chain")
        assert chain, "the chain head must carry its evolution chain"
        assert [c["content"] for c in chain] == ["v2 switched to React", "v1 used Vue"], (
            "the order must be latest -> oldest, same as on the search side"
        )
        assert {"node_id", "content", "memory_at", "layer"} <= set(chain[0].keys())

    def test_unlinked_memory_has_no_chain_field(self, faiss_store):
        _seed_chain(faiss_store)
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
            with_evolution=True,
        ))
        solo = next(m for m in got["memories"] if m["content"] == "Standalone memory")
        assert "evolution_chain" not in solo, "a memory without a chain must not get an empty field"

    def test_item_count_and_total_unchanged(self, faiss_store):
        """The switch must not change item counts: history versions are SUPERSEDED and never listed."""
        _seed_chain(faiss_store)
        c = _client_with(faiss_store)
        base = run(c._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
        ))
        with_evo = run(c._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
            with_evolution=True,
        ))
        assert with_evo["total"] == base["total"] == 2
        assert len(with_evo["memories"]) == len(base["memories"])
        assert ({m["memory_id"] for m in with_evo["memories"]}
                == {m["memory_id"] for m in base["memories"]})

    def test_only_current_page_expanded(self, faiss_store):
        """Expansion happens after paging: page 2 chains must not be traced while page 1 is fetched."""
        _seed_chain(faiss_store)
        calls = []
        orig = faiss_store.get_by_ids

        async def _counting(ids):
            calls.append(list(ids))
            return await orig(ids)

        faiss_store.get_by_ids = _counting
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=1, offset=0, order="desc",
            with_evolution=True,
        ))
        assert len(got["memories"]) == 1
        # Without a chain head on page 1 no re-read should happen at all; with one, at most that chain
        assert len(calls) <= 1

    def test_expand_failure_does_not_break_list(self, faiss_store):
        """Evolution chains are extra info: the list must still come back normally without them."""
        _seed_chain(faiss_store)

        async def _boom(ids):
            raise RuntimeError("vdb down")

        faiss_store.get_by_ids = _boom
        got = run(_client_with(faiss_store)._list_vdb_bucket(
            user_id="u1", agent_id=None, limit=100, offset=0, order="desc",
            with_evolution=True,
        ))
        assert got["total"] == 2
        assert all("evolution_chain" not in m for m in got["memories"])
