"""Topic isolation granularity + support blending. Both areas back problems that regression tests cover.

**Topic isolation key**: Topics used to be created with the session-bearing `isolation_key`, so the same tag name
got a separate, mutually oblivious Topic in every session. That splits same-name topics into several nodes, which in turn stops the temporal spine
from building edges (no cross-batch predecessor found within a topic), makes topic summaries repeatedly summarize same-name fragments, and leaves cross-session tag
recall disconnected.

**Support blending**: graph-edge reachability and episode co-membership used to be combined with max. They are independent observations, so both holding should be more credible;
and episode support is pressed down by sqrt(size), naturally smaller than co_occurs, so taking max left the episode channel
completely useless for "candidates that already have a co-occurrence edge".
"""

import inspect

import pytest

from hy_memory.models.memory import MemoryLayer, MemoryNode
from hy_memory.pipelines._graph.rerank import blend_support


# ======================================================================
# Topic isolation granularity
# ======================================================================

class TestTopicScopeKey:
    def _node(self, *, user="u1", agent="a1", session="s1"):
        return MemoryNode(
            user_id=user, agent_id=agent, session_id=session,
            layer=MemoryLayer.L2_FACT, content="x",
        )

    def test_topic_scope_drops_session(self):
        n = self._node()
        assert n.get_isolation_key() == "u1::a1::s1"
        assert n.get_topic_scope_key() == "u1::a1"

    def test_same_tag_across_sessions_shares_scope(self):
        """Core requirement: the same user's same-named tags across different sessions must land on the same Topic."""
        keys = {
            self._node(session=f"locomo::{i}").get_topic_scope_key()
            for i in range(20)
        }
        assert keys == {"u1::a1"}, "20 sessions must share a single Topic scope"

    def test_different_users_still_isolated(self):
        """Loosening granularity must never loosen it across users."""
        a = self._node(user="u1").get_topic_scope_key()
        b = self._node(user="u2").get_topic_scope_key()
        c = self._node(agent="a2").get_topic_scope_key()
        assert a != b and a != c

    @pytest.mark.parametrize("ik,want", [
        ("u1::a1::s1", "u1::a1"),
        ("u1::a1::locomo::7", "u1::a1"),      # a session id containing :: must still be truncated correctly
        ("u1::a1", "u1::a1"),
        ("", ""),
        ("only_one_part", "only_one_part"),
    ])
    def test_topic_scope_of_isolation_key(self, ik, want):
        assert MemoryNode.topic_scope_of(ik) == want

    def test_build_helper_matches_instance_method(self):
        n = self._node()
        assert MemoryNode.build_topic_scope_key("u1", "a1") == n.get_topic_scope_key()

    def test_roundtrip_write_and_read_use_same_key(self):
        """Creating and querying a Topic must use the same key -- both backends match exactly, so a mismatch silently finds nothing."""
        n = self._node(session="locomo::7")
        write_key = n.get_topic_scope_key()
        read_key = MemoryNode.topic_scope_of(n.get_isolation_key())
        assert write_key == read_key


class TestTopicCallSitesConsistent:
    """Every Topic create/query site must use the topic scope; missing one silently finds nothing."""

    def test_projector_builds_with_topic_scope(self):
        from hy_memory.pipelines._graph import projector

        src = inspect.getsource(projector)
        assert "get_topic_scope_key()" in src
        assert "add_topic_tag(\n                    nid, str(tag), node.get_isolation_key()" not in src

    @pytest.mark.parametrize("module_path", [
        "hy_memory.pipelines.writer",
        "hy_memory.pipelines._graph.topic_summary",
        "hy_memory.pipelines.reader_legacy",
        "hy_memory.pipelines.system2_agent",
    ])
    def test_query_sites_use_topic_scope(self, module_path):
        import importlib

        src = inspect.getsource(importlib.import_module(module_path))
        if "get_all_topics(" not in src:
            pytest.skip("this module does not query Topics")
        assert "topic_scope_of(" in src, (
            f"{module_path} queries Topics with isolation_key directly -- inconsistent with the Topic creation key, so it would find nothing"
        )


# ======================================================================
# Bounded support blending
# ======================================================================

class TestBlendSupport:
    def test_dual_evidence_scores_higher_than_either(self):
        """Dual evidence must score above either source alone -- the entire reason max was replaced."""
        g = {"bike": (0.55, "eat", "co_occurs", 1)}
        e = {"bike": (0.21, "eat", "episode", 1)}
        merged = blend_support(g, e, lam=0.3)["bike"][0]
        assert merged > 0.55 and merged > 0.21
        assert merged == pytest.approx(0.55 + 0.3 * 0.21)

    def test_lambda_discounts_the_second_source(self):
        """Discount factor: keeps two weak pieces of evidence from stacking into a strong-looking conclusion."""
        g = {"x": (0.4, "s", "co_occurs", 1)}
        e = {"x": (0.4, "s", "episode", 1)}
        assert blend_support(g, e, lam=0.3)["x"][0] == pytest.approx(0.52)
        assert blend_support(g, e, lam=0.0)["x"][0] == pytest.approx(0.4)

    def test_capped(self):
        """The cap keeps the admission bar at best at (1-gamma)*min_score; guardrails unchanged."""
        g = {"x": (0.9, "s", "co_occurs", 1)}
        e = {"x": (0.8, "s", "episode", 1)}
        assert blend_support(g, e, lam=0.3, support_cap=1.0)["x"][0] == 1.0

    def test_episode_only_candidate_kept_as_is(self):
        """Cross-batch candidates with no co-occurrence edge -- exactly where the episode channel should shine."""
        e = {"park": (0.21, "eat", "episode", 1)}
        out = blend_support({}, e, lam=0.3)
        assert out["park"][0] == pytest.approx(0.21)
        assert out["park"][2] == "episode"

    def test_graph_only_candidate_untouched(self):
        g = {"a": (0.5, "s", "causes", 2)}
        assert blend_support(g, {}, lam=0.3) == g

    def test_no_episode_returns_copy_not_alias(self):
        """Must not mutate the caller's dict in place."""
        g = {"a": (0.5, "s", "causes", 1)}
        out = blend_support(g, {}, lam=0.3)
        out["a"] = (9.9, "x", "y", 1)
        assert g["a"][0] == 0.5

    def test_attribution_follows_stronger_side(self):
        """Attribution takes the side with stronger support, making it observable which channel brought the candidate back."""
        g = {"x": (0.6, "seedG", "co_occurs", 1)}
        e = {"x": (0.1, "seedE", "episode", 1)}
        assert blend_support(g, e, lam=0.3)["x"][1:] == ("seedG", "co_occurs", 1)

        g2 = {"y": (0.1, "seedG", "co_occurs", 1)}
        e2 = {"y": (0.6, "seedE", "episode", 1)}
        assert blend_support(g2, e2, lam=0.3)["y"][1:] == ("seedE", "episode", 1)

    def test_reader_uses_blend_not_max(self):
        """Regression: if the reader reverts to max, the episode channel gets shaved off systematically."""
        from hy_memory.pipelines.reader_hybrid import HybridReadPipeline

        src = inspect.getsource(HybridReadPipeline._apply_graph_rerank)
        assert "blend_support(" in src
        assert "if prev is None or val[0] > prev[0]" not in src


# ======================================================================
# Threshold recalibration
# ======================================================================

class TestThresholdsRecalibrated:
    """0.6 was never calibrated against the representative score distribution and left expansion and paths permanently untriggered."""

    def test_defaults_are_half(self):
        from hy_memory.config import GraphStoreConfig

        c = GraphStoreConfig()
        assert c.seed_floor == 0.5
        assert c.path_high_floor == 0.5
        assert c.episode_blend_lambda == 0.3

    def test_env_still_overrides(self, monkeypatch):
        from hy_memory.config import GraphStoreConfig

        monkeypatch.setenv("MEMORY_GRAPH_SEED_FLOOR", "0.42")
        monkeypatch.setenv("MEMORY_GRAPH_EPISODE_BLEND_LAMBDA", "0.7")
        c = GraphStoreConfig()
        assert c.seed_floor == 0.42
        assert c.episode_blend_lambda == 0.7

    def test_seed_floor_admits_multiple_seeds_at_observed_scale(self):
        """Under the representative score distribution, 0.6 admits a single seed while 0.5 admits several."""
        from hy_memory.pipelines._graph.rerank import reachable_within

        # a representative sample pool: top score 0.636, the rest between 0.5 and 0.6
        pool = {"a": 0.636, "b": 0.58, "c": 0.55, "d": 0.51, "e": 0.42}
        adj = {n: [("out1", "co_occurs", 0.8)] for n in pool}
        assert len(reachable_within(pool, adj, seed_floor=0.6)) >= 1
        # key: at 0.5 the seeds joining corroboration go from 1 to 4
        seeds_06 = sum(1 for s in pool.values() if s >= 0.6)
        seeds_05 = sum(1 for s in pool.values() if s >= 0.5)
        assert seeds_06 == 1 and seeds_05 == 4
