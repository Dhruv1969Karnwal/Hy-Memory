"""Idempotency of graph writes, plus the cleanup that has to happen before a rebuild.

`add_edge` used `CREATE`, so the same pair of nodes could get edges of the same type created **repeatedly**:

- `UPDATE` is an in-place update (`node_id` unchanged), so a memory merged several times takes part in projection several times;
  if two old nodes show up in the same batch again, `co_occurs` gets created again. Production measured one memory being UPDATEd 7 times.
- The System 2 agent loop may call `add_edge` repeatedly for the same pair.

Measured in production (eval database): **204k duplicate pairs, the worst pair holding 32 copies**, 99% of which are
untyped `related` written by System 2. Duplicate edges do not amplify scoring (`build_adjacency` folds them into the
strongest one), but they do eat the `max_edges` budget of `get_related_edges` -- Cypher's `LIMIT` truncates first and
deduplication happens on the Python side, so duplicates crowd out the real edges that should have been returned.
"""

import asyncio
import tempfile

import pytest

from hy_memory.config import MemoryConfig
from hy_memory.models.memory import MemoryLayer, MemoryNode


def _cfg(path: str) -> MemoryConfig:
    cfg = MemoryConfig()
    cfg.graph_store.provider = "kuzu"
    cfg.graph_store.db_path = path
    return cfg


async def _store():
    from hy_memory.data.graph_store_kuzu import KuzuGraphStore

    gs = KuzuGraphStore(_cfg(tempfile.mkdtemp() + "/g"))
    await gs.initialize()
    return gs


def _node(content: str, *, session="s1", layer=MemoryLayer.L2_FACT, tags=None):
    return MemoryNode(
        user_id="u", agent_id="a", session_id=session,
        layer=layer, content=content, tags=tags or [],
    )


class TestAddEdgeIdempotent:
    @pytest.mark.asyncio
    async def test_repeated_add_edge_yields_one_edge(self):
        gs = await _store()
        try:
            a, b = _node("m0"), _node("m1")
            await gs.upsert_memory_node(a)
            await gs.upsert_memory_node(b)
            for w in (0.5, 0.9, 0.7):
                await gs.add_edge(
                    a.node_id, b.node_id, "RELATED_TO",
                    {"relation_type": "co_occurs", "weight": w},
                )
            rows = await gs._run(
                "MATCH (x:Memory)-[r:RELATED_TO]->(y:Memory) RETURN count(*);"
            )
            # one per direction = 2; with CREATE it would be 6
            assert int(rows[0][0]) == 2
        finally:
            await gs.close()

    @pytest.mark.asyncio
    async def test_on_match_keeps_max_weight(self):
        """ON MATCH keeps the larger weight, matching the read-side build_adjacency semantics of folding to the strongest."""
        gs = await _store()
        try:
            a, b = _node("m0"), _node("m1")
            await gs.upsert_memory_node(a)
            await gs.upsert_memory_node(b)
            for w in (0.5, 0.9, 0.7):
                await gs.add_edge(
                    a.node_id, b.node_id, "RELATED_TO",
                    {"relation_type": "co_occurs", "weight": w},
                )
            edges = await gs.get_related_edges([a.node_id])
            co = [e for e in edges if e["relation_type"] == "co_occurs"]
            assert len(co) == 1
            assert co[0]["weight"] == pytest.approx(0.9)
        finally:
            await gs.close()

    @pytest.mark.asyncio
    async def test_different_relation_types_coexist(self):
        """The same pair of nodes **may** carry several relations -- deduplication must not merge them away."""
        gs = await _store()
        try:
            a, b = _node("m0"), _node("m1")
            await gs.upsert_memory_node(a)
            await gs.upsert_memory_node(b)
            for rt in ("co_occurs", "causes", "elaborates"):
                await gs.add_edge(
                    a.node_id, b.node_id, "RELATED_TO",
                    {"relation_type": rt, "weight": 0.8},
                )
            edges = await gs.get_related_edges([a.node_id])
            assert {e["relation_type"] for e in edges} == {
                "co_occurs", "causes", "elaborates",
            }
        finally:
            await gs.close()

    @pytest.mark.asyncio
    async def test_reprojecting_same_batch_is_stable(self):
        """Regression on a production case: the same batch of nodes projected twice (UPDATE puts old nodes back into a batch)."""
        from hy_memory.pipelines._graph.projector import project_memories_to_graph

        gs = await _store()
        try:
            nodes = [_node(f"m{i}", tags=["health"]) for i in range(3)]
            for _ in range(2):
                await project_memories_to_graph(
                    graph_store=gs, embed_service=None, nodes=nodes,
                )
            rows = await gs._run(
                "MATCH (x:Memory)-[r:RELATED_TO]->(y:Memory) RETURN count(*);"
            )
            # 3 nodes fully connected = 3 pairs, both directions = 6 edges. Projecting twice must still yield 6.
            assert int(rows[0][0]) == 6
        finally:
            await gs.close()


class TestClearProjection:
    @pytest.mark.asyncio
    async def test_clears_l2_and_orphan_topics(self):
        gs = await _store()
        try:
            for i in range(3):
                n = _node(f"m{i}", session=f"s{i}", tags=["health"])
                await gs.upsert_memory_node(n)
                # deliberately building Topics with the old (session-scoped) key to mimic fragmented legacy data
                await gs.add_topic_tag(n.node_id, "health", n.get_isolation_key())
            res = await gs.clear_projection("u", "a", layer="l2_fact")
            assert res["memories"] == 3
            assert res["topics"] == 3
            rows = await gs._run("MATCH (t:Topic) RETURN count(*);")
            assert int(rows[0][0]) == 0
        finally:
            await gs.close()

    @pytest.mark.asyncio
    async def test_does_not_touch_l6_schema(self):
        """Key guardrail: one user's Memory nodes still include l6_schema written by System 2, so
        an unconditional delete would destroy ultra data."""
        gs = await _store()
        try:
            await gs.upsert_memory_node(_node("fact"))
            await gs.upsert_memory_node(_node("schema", layer=MemoryLayer.L6_SCHEMA))
            await gs.clear_projection("u", "a", layer="l2_fact")
            rows = await gs._run(
                "MATCH (m:Memory {layer:'l6_schema'}) RETURN count(*);"
            )
            assert int(rows[0][0]) == 1
            rows = await gs._run("MATCH (m:Memory) RETURN count(*);")
            assert int(rows[0][0]) == 1, "only the single l6_schema node should remain"
        finally:
            await gs.close()

    @pytest.mark.asyncio
    async def test_other_users_untouched(self):
        gs = await _store()
        try:
            await gs.upsert_memory_node(_node("mine"))
            other = MemoryNode(
                user_id="u2", agent_id="a", session_id="s1",
                layer=MemoryLayer.L2_FACT, content="theirs",
            )
            await gs.upsert_memory_node(other)
            await gs.clear_projection("u", "a", layer="l2_fact")
            rows = await gs._run(
                "MATCH (m:Memory {user_id:'u2'}) RETURN count(*);"
            )
            assert int(rows[0][0]) == 1
        finally:
            await gs.close()

    def test_base_default_is_noop(self):
        """A backend that does not implement this capability does nothing and must not raise."""
        import inspect

        from hy_memory.data.graph_store_base import GraphStoreBase

        src = inspect.getsource(GraphStoreBase.clear_projection)
        assert "l2_fact" in src and "l6_schema" in src

    def test_all_backends_implement_it(self):
        from hy_memory.data.graph_store_base import GraphStoreBase
        from hy_memory.data.graph_store_kuzu import KuzuGraphStore
        from hy_memory.data.graph_store_neo4j import Neo4jGraphStore

        for cls in (KuzuGraphStore, Neo4jGraphStore):
            assert cls.clear_projection is not GraphStoreBase.clear_projection, (
                f"{cls.__name__} does not implement clear_projection itself and would silently skip the cleanup"
            )


class TestNoCreateLeftInAddEdge:
    """Static guard: going back to CREATE would reintroduce duplicate edges."""

    @pytest.mark.parametrize("module_path", [
        "hy_memory.data.graph_store_kuzu",
        "hy_memory.data.graph_store_neo4j",
    ])
    def test_add_edge_uses_merge(self, module_path):
        import importlib
        import inspect

        mod = importlib.import_module(module_path)
        cls = next(
            v for k, v in vars(mod).items()
            if isinstance(v, type) and hasattr(v, "add_edge")
            and v.__module__ == module_path
        )
        src = inspect.getsource(cls.add_edge)
        assert "MERGE" in src, f"{module_path}.add_edge does not use MERGE"
        assert "CREATE (a)-[:RELATED_TO" not in src, (
            f"{module_path}.add_edge still creates edges with CREATE"
        )
        assert "ON MATCH" in src, "ON MATCH is missing, so repeated calls would not update the weight"


class TestBuildMemoryGraphClearFirst:
    def test_client_exposes_clear_first(self):
        import inspect

        from hy_memory.client import HyMemoryClient

        for fn in ("build_memory_graph", "async_build_memory_graph"):
            params = inspect.signature(getattr(HyMemoryClient, fn)).parameters
            assert "clear_first" in params, f"{fn} is missing clear_first"
            assert params["clear_first"].default is False, "the default must be False"

    def test_clear_scoped_to_l2_fact(self):
        import inspect

        from hy_memory.client import HyMemoryClient

        src = inspect.getsource(HyMemoryClient.async_build_memory_graph)
        assert "clear_projection(" in src
        assert "L2_FACT" in src, "the cleanup must be narrowed by layer, otherwise it would delete l6_schema"
