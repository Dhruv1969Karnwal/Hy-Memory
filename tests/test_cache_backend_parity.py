"""Every cache backend must implement the same set of **stateful** methods.

Why this guard exists: `CacheBase` leaves no-op defaults for the stateful methods (`return None` /
`return False`), so a new backend runs without implementing everything -- at the cost of **missing implementations degrading silently, with no error at all**.

Measured cost: dev10 implemented episode splitting only in `cache_sqlite`, while production runs
`cache_mysql`, so `get_episode_state` always returned None and every write started a new episode.
Across 8798 online splitting decisions **all came back `new` and `idle_seconds` was always null**: the feature never took effect, yet not a single error was logged.

Audit methods (memory_operations / pipeline_logs / metrics) are excluded -- losing them only hurts observability.
This covers only stateful methods that **change functional behavior**.
"""

import inspect

import pytest

from hy_memory.data.cache_base import CacheBase
from hy_memory.data.cache_mysql import MysqlCache
from hy_memory.data.cache_sqlite import SqliteCache


# Stateful methods that change functional behavior: a missing implementation silently disables the feature
STATEFUL_METHODS = [
    "get_episode_state",
    "upsert_episode_state",
]

BACKENDS = [
    pytest.param(SqliteCache, id="sqlite"),
    pytest.param(MysqlCache, id="mysql"),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", STATEFUL_METHODS)
def test_backend_overrides_stateful_method(backend, method):
    """Each backend must **implement it itself** instead of inheriting the base class no-op."""
    assert hasattr(backend, method), f"{backend.__name__} is missing {method}"
    own = getattr(backend, method)
    base = getattr(CacheBase, method)
    assert own is not base, (
        f"{backend.__name__}.{method} inherits the CacheBase no-op default directly -- "
        f"on this backend the feature silently does nothing (no error, degraded behavior)"
    )


@pytest.mark.parametrize("method", STATEFUL_METHODS)
def test_signatures_match_base(method):
    """Signatures must match the base class, otherwise callers passing base-class arguments hit a TypeError (usually swallowed by an except)."""
    base_sig = inspect.signature(getattr(CacheBase, method))
    for backend in (SqliteCache, MysqlCache):
        sig = inspect.signature(getattr(backend, method))
        assert list(sig.parameters) == list(base_sig.parameters), (
            f"{backend.__name__}.{method} signature does not match CacheBase:\n"
            f"  base: {base_sig}\n  impl: {sig}"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_episode_state_table_declared(backend):
    """The DDL must declare episode_state -- implementing the method without creating the table also fails silently."""
    import hy_memory.data.cache_mysql as m
    import hy_memory.data.cache_sqlite as s

    mod = s if backend is SqliteCache else m
    src = inspect.getsource(mod)
    assert "CREATE TABLE IF NOT EXISTS episode_state" in src, (
        f"{mod.__name__} does not declare the episode_state table"
    )


class TestRollingSummaryFullyRemoved:
    """Rolling summary was removed as a whole; leftover references would break the app on the startup path."""

    @pytest.mark.parametrize("method", [
        "get_summary_buffer", "upsert_summary_buffer", "clear_summary_buffer",
    ])
    def test_summary_buffer_methods_gone(self, method):
        for cls in (CacheBase, SqliteCache, MysqlCache):
            assert not hasattr(cls, method), f"{cls.__name__} still has {method}"

    def test_summary_buffer_table_gone(self):
        import hy_memory.data.cache_mysql as m
        import hy_memory.data.cache_sqlite as s

        for mod in (s, m):
            assert "summary_buffer" not in inspect.getsource(mod), (
                f"{mod.__name__} still has summary_buffer DDL or queries"
            )

    def test_modules_gone(self):
        for name in ("hy_memory.agent.rolling_summary", "hy_memory.agent.summarizer"):
            with pytest.raises(ImportError):
                __import__(name)

    def test_enable_summary_switch_gone(self):
        from hy_memory.config import LLMConfig
        from hy_memory.pipelines.base import WriteRequest

        assert not hasattr(LLMConfig(), "enable_summary")
        assert not hasattr(LLMConfig(), "summary_buffer_window")
        assert not hasattr(WriteRequest(), "enable_summary")

    def test_public_api_param_gone(self):
        from hy_memory.client import HyMemoryClient

        for fn in ("add", "async_add"):
            if not hasattr(HyMemoryClient, fn):
                continue
            params = inspect.signature(getattr(HyMemoryClient, fn)).parameters
            assert "enable_summary" not in params, f"client.{fn} still has enable_summary"

    def test_mem_agent_summary_path_gone(self):
        from hy_memory.agent.mem_agent import MemAgent, ProcessMode

        assert not hasattr(ProcessMode, "SUMMARY_ONLY")
        params = inspect.signature(MemAgent.process_add).parameters
        assert "enable_summary" not in params
        src = inspect.getsource(MemAgent)
        assert "self.summarizer" not in src
