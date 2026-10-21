# -*- coding: utf-8 -*-
"""SHADOW nodes are filtered out while expanding evolution chains.

After dedup changed from hard deletion to flagging SHADOW (logical delete, recoverable), one leak path appeared:

  _trace_full_chain fetches by node_id through get_by_ids and **ignores status**; a deduped
  node keeps its supersedes / superseded_by links too, so as soon as another node on the
  chain is recalled, the just-deduped "deleted" content comes back to the user via evolution_chain.

Worse: when dedup finds a duplicate chain head it also handles the whole chain (chain_node_ids),
and once the whole chain is SHADOW every is_latest is False, so _trace_full_chain picks
its head as `max(all_nodes, key=_time_key)` -- handing back a SHADOW node as representative.

This file pins down:
  - individual SHADOW nodes on a chain never appear in evolution_chain;
  - the representative node is the head of the visible chain and is never SHADOW;
  - when the whole chain is invisible the hit is dropped, instead of falling back to the raw hit and returning SHADOW;
  - normal chains (no SHADOW) behave exactly as before.
"""

import asyncio
import random
import uuid

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.pipelines._retrieval.evolution import expand_evolution_chains


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _emb(dims=32):
    v = [random.gauss(0, 1) for _ in range(dims)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


@pytest.fixture
def config(tmp_path):
    return MemoryConfig.from_dict({
        "vector_store": {
            "provider": "chroma",
            "collection_name": f"test_shadowchain_{uuid.uuid4().hex[:8]}",
            "persist_directory": str(tmp_path / "chroma"),
            "embedding_dims": 32,
        },
        "cache": {"backend": "sqlite"},
    })


@pytest.fixture
def store(config):
    from hy_memory.data.vector_store_chroma import ChromaVectorStore
    vs = ChromaVectorStore(config)
    run(vs.initialize())
    yield vs
    run(vs.close())


def _node(store, *, content, status, is_latest, supersedes=None, emb=None):
    n = MemoryNode(
        user_id="u1", agent_id="a1", session_id="s1",
        layer=MemoryLayer.L2_FACT, content=content, owner="user",
        status=status, is_latest=is_latest,
        supersedes=supersedes,
        embedding=emb or _emb(),
    )
    return run(store.upsert(n)), n


def _link(store, old_id, new_id):
    """Backfills superseded_by to build the two-way chain (same as the SUPERSEDE path in writer.py)."""
    run(store.update_payload(old_id, {"superseded_by": [new_id]}))


def _hit(store, node_id, score=0.9):
    node = run(store.get_by_id(node_id))
    return {"node_id": node_id, "score": score, "node": node}


class TestShadowFilteredFromChain:
    def test_shadow_body_not_exposed_in_chain(self, store):
        """A SHADOW chain body (e.g. a deduped duplicate that is not the head) stays out of the chain."""
        e = _emb()
        old_id, _ = _node(store, content="v1 old", status=MemoryStatus.SHADOW,
                          is_latest=False, emb=e)
        mid_id, _ = _node(store, content="v2 mid", status=MemoryStatus.SUPERSEDED,
                          is_latest=False, supersedes=[old_id], emb=e)
        head_id, _ = _node(store, content="v3 head", status=MemoryStatus.ACTIVE,
                           is_latest=True, supersedes=[mid_id], emb=e)
        _link(store, old_id, mid_id)
        _link(store, mid_id, head_id)

        out = run(expand_evolution_chains(store, [_hit(store, head_id)]))
        assert len(out) == 1
        contents = [c["content"] for c in out[0]["evolution_chain"]]
        assert "v3 head" in contents
        assert "v2 mid" in contents
        assert "v1 old" not in contents, "a SHADOW chain body must not appear in evolution_chain"

    def test_representative_node_is_never_shadow(self, store):
        """With a SHADOW head and an ACTIVE body, the representative must be the visible one."""
        e = _emb()
        body_id, _ = _node(store, content="visible body", status=MemoryStatus.ACTIVE,
                           is_latest=False, emb=e)
        head_id, _ = _node(store, content="shadowed head", status=MemoryStatus.SHADOW,
                           is_latest=False, supersedes=[body_id], emb=e)
        _link(store, body_id, head_id)

        out = run(expand_evolution_chains(store, [_hit(store, body_id)]))
        assert len(out) == 1
        rep = out[0]["node"]
        assert rep.status != MemoryStatus.SHADOW, "the representative node must not be SHADOW"
        assert rep.content == "visible body"
        assert all(
            c["content"] != "shadowed head"
            for c in out[0]["evolution_chain"]
        )

    def test_fully_shadowed_chain_is_dropped(self, store):
        """When dedup SHADOWs the whole chain the hit is dropped, with no SHADOW content returned.

        This is the critical case: after the whole chain is SHADOWed every is_latest is False,
        so _trace_full_chain falls back to taking the newest head and returns a SHADOW node.
        Falling back to the raw hit would hand logically deleted content back to the user.
        """
        e = _emb()
        old_id, _ = _node(store, content="dup old", status=MemoryStatus.SHADOW,
                          is_latest=False, emb=e)
        new_id, _ = _node(store, content="dup new", status=MemoryStatus.SHADOW,
                          is_latest=False, supersedes=[old_id], emb=e)
        _link(store, old_id, new_id)

        out = run(expand_evolution_chains(store, [_hit(store, old_id)]))
        assert out == [], f"a fully SHADOWed chain must drop this hit, got {out}"

    def test_fully_shadowed_chain_dropped_even_when_hit_is_head(self, store):
        """A hit from the head side must be dropped too (so the filter works in both directions)."""
        e = _emb()
        old_id, _ = _node(store, content="d1", status=MemoryStatus.SHADOW,
                          is_latest=False, emb=e)
        new_id, _ = _node(store, content="d2", status=MemoryStatus.SHADOW,
                          is_latest=False, supersedes=[old_id], emb=e)
        _link(store, old_id, new_id)

        out = run(expand_evolution_chains(store, [_hit(store, new_id)]))
        assert out == []

    def test_mixed_hits_only_shadowed_chain_dropped(self, store):
        """A batch mixing a normal chain and a fully SHADOWed chain drops only the latter."""
        e1, e2 = _emb(), _emb()
        # Normal chain
        ok_old, _ = _node(store, content="ok v1", status=MemoryStatus.SUPERSEDED,
                          is_latest=False, emb=e1)
        ok_head, _ = _node(store, content="ok v2", status=MemoryStatus.ACTIVE,
                           is_latest=True, supersedes=[ok_old], emb=e1)
        _link(store, ok_old, ok_head)
        # Fully SHADOWed chain
        bad_old, _ = _node(store, content="bad v1", status=MemoryStatus.SHADOW,
                           is_latest=False, emb=e2)
        bad_new, _ = _node(store, content="bad v2", status=MemoryStatus.SHADOW,
                           is_latest=False, supersedes=[bad_old], emb=e2)
        _link(store, bad_old, bad_new)

        out = run(expand_evolution_chains(
            store, [_hit(store, ok_head), _hit(store, bad_old)]
        ))
        assert len(out) == 1
        assert out[0]["node"].content == "ok v2"

    def test_normal_chain_unaffected(self, store):
        """Chains without SHADOW behave unchanged (the filter must not hit the normal path)."""
        e = _emb()
        old_id, _ = _node(store, content="n v1", status=MemoryStatus.SUPERSEDED,
                          is_latest=False, emb=e)
        head_id, _ = _node(store, content="n v2", status=MemoryStatus.ACTIVE,
                           is_latest=True, supersedes=[old_id], emb=e)
        _link(store, old_id, head_id)

        # Hit on the chain body: it must expand and represent the chain by its head
        out = run(expand_evolution_chains(store, [_hit(store, old_id)]))
        assert len(out) == 1
        assert out[0]["node"].node_id == head_id
        assert out[0]["is_evolved"] is True
        assert [c["content"] for c in out[0]["evolution_chain"]] == ["n v2", "n v1"]

    def test_non_chain_hit_passes_through(self, store):
        """A plain node that is not on a chain is returned as-is, unaffected."""
        nid, _ = _node(store, content="standalone", status=MemoryStatus.ACTIVE,
                       is_latest=True)
        out = run(expand_evolution_chains(store, [_hit(store, nid)]))
        assert len(out) == 1
        assert out[0]["node"].content == "standalone"
        assert not out[0].get("is_evolved")
