# -*- coding: utf-8 -*-
"""
Trigger conditions for degrading L1_RAW to SHADOW (tail of writer._run_agent).

Background: when the extractor concludes "this dialogue holds no extractable user facts" (pure technical
Q&A and the like) it returns memory=[]; the previous `if stored_ids and memory_id` meant empty stored_ids -> no
degradation, so L1_RAW stayed ACTIVE for good and was recalled by list / search / keyword_search.

This test pins down five semantics:
  1. empty extraction (success=True, memory=[]) -> **must degrade** (the point of this fix)
  2. failed extraction (success=False, e.g. EMPTY_RESPONSE) -> **must not degrade**, stays ACTIVE for retry
  3. normal extraction with content -> still degrades (regression protection, original behaviour unchanged)
  4. an explicitly requested non-raw layer (metadata passes extra["layer"]) -> **must not degrade**, otherwise
     it amounts to a logical delete of the data the caller just wrote
  5. upsert raising during degradation -> does not change the success verdict of this write (fault tolerance kept)
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hy_memory.models.memory import MemoryNode, MemoryLayer, MemoryStatus
from hy_memory.pipelines.base import WriteRequest, WriteResponse


# ---------------------------------------------------------------- fakes


@dataclass
class _FakeAgentResult:
    """Carries only the fields _run_agent actually reads, avoiding AgentResult construction details."""

    success: bool = True
    extracted_info: Dict[str, Any] = field(default_factory=dict)
    suggested_layer: Optional[str] = None
    summary: Optional[str] = None
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    should_merge: bool = False
    merge_target_id: Optional[str] = None
    tokens_used: int = 0
    extract_elapsed_ms: float = 1.0
    summary_elapsed_ms: float = 0.0
    error: Optional[str] = None
    error_code: Optional[str] = None


class _FakeMemAgent:
    def __init__(self, result: _FakeAgentResult):
        self._result = result

    async def process_add(self, **kwargs):
        return self._result


class _FakeVectorStore:
    """In-memory VDB. Records (node_id, status) per upsert so assertions can tell whether degradation happened."""

    def __init__(self, fail_upsert: bool = False):
        self.nodes: Dict[str, MemoryNode] = {}
        self.upsert_calls: List[tuple] = []
        self._fail_upsert = fail_upsert

    async def upsert(self, node: MemoryNode) -> str:
        if self._fail_upsert:
            raise RuntimeError("vdb down")
        nid = node.node_id or uuid.uuid4().hex
        node.node_id = nid
        self.nodes[nid] = node
        self.upsert_calls.append((nid, node.status))
        return nid

    async def get_by_id(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    async def search(self, *a, **kw):
        return []

    async def list_by_user(self, *a, **kw):
        return []


class _FakeEmbed:
    async def embed_batch(self, texts):
        return [[0.0] * 8 for _ in texts]

    async def embed_queued(self, text):
        return [0.0] * 8


# ---------------------------------------------------------------- harness


def _build_writer(agent_result: _FakeAgentResult, store: _FakeVectorStore):
    """Build a writer used solely to run _run_agent.

    cache=None makes every pipeline log return immediately (`_write_extract_log` checks it on the first line),
    so there is no dependency on sqlite or on observability being installed.
    """
    from hy_memory.config import MemoryConfig
    from hy_memory.pipelines.writer import MemoryWriter

    cfg = MemoryConfig()
    w = MemoryWriter(cfg, embed_service=_FakeEmbed(), vector_store=store, cache=None)
    w._mem_agent = _FakeMemAgent(agent_result)
    w._vector_store = store
    # When content is extracted, the _direct_store branch is enough (no real reconcile LLM run needed)
    w._reconcile_and_store = _stub_store          # type: ignore[assignment]
    w._direct_store = _stub_store                 # type: ignore[assignment]
    return w


async def _stub_store(**kwargs):
    """(stored_ids, recon_error, ops_counts, recon_tokens)"""
    metas = kwargs.get("new_memories_meta") or []
    ids = [f"fact-{i}" for i in range(len(metas))]
    return ids, None, {}, {}


def _mk_node(layer: MemoryLayer = MemoryLayer.L1_RAW) -> MemoryNode:
    return MemoryNode(
        node_id="raw-1",
        user_id="u1",
        agent_id="default_agent",
        session_id="s1",
        content="Can you access the git repository directly",
        layer=layer,
        status=MemoryStatus.ACTIVE,
        is_latest=True,
    )


class _Span:
    """tracer span placeholder (_run_agent only calls these two methods)."""

    def set_output(self, *a, **kw):
        pass

    def set_error(self, *a, **kw):
        pass


def _run_agent(agent_result: _FakeAgentResult,
               node: Optional[MemoryNode] = None,
               store: Optional[_FakeVectorStore] = None,
               extra: Optional[Dict[str, Any]] = None,
               memory_id: Optional[str] = None,
               upsert_store: Optional[_FakeVectorStore] = None,
               request_id: str = "req-test"):
    """Run _run_agent once and return (mem_node, response, store).

    upsert_store: when passed, it is used as the vector_store argument of _run_agent
                  (to simulate a failing degradation write); store stays the writer's internal store.
    memory_id:    defaults to node.node_id; an empty string simulates a failed L1 write.
    """
    node = node or _mk_node()
    store = store or _FakeVectorStore()
    store.nodes[node.node_id] = node

    w = _build_writer(agent_result, store)
    req = WriteRequest(
        content=node.content,
        user_id="u1",
        agent_id="default_agent",
        session_id="s1",
        extra=dict(extra or {}),
        request_id=request_id,
    )
    resp = WriteResponse(success=True)

    asyncio.run(
        w._run_agent(
            request=req,
            response=resp,
            vector_store=upsert_store or store,
            mem_node=node,
            memory_id=node.node_id if memory_id is None else memory_id,
            tracer_span=_Span(),
        )
    )
    return node, resp, store


# ---------------------------------------------------------------- tests


class TestShadowOnEmptyExtract:
    def test_empty_extract_still_shadows_l1_raw(self):
        """An empty extraction (a normal verdict) must still degrade -- the core semantics of this fix."""
        node, resp, store = _run_agent(
            _FakeAgentResult(success=True, extracted_info={"memory": [], "intentions": []})
        )

        assert resp.extra["agent_status"] == "success"
        assert resp.extra["agent_nodes"] == 0, "an empty extraction must not produce higher-layer nodes"
        assert node.status == MemoryStatus.SHADOW, "L1_RAW must still degrade on an empty extraction"
        assert (node.node_id, MemoryStatus.SHADOW) in store.upsert_calls

    def test_failed_extract_keeps_active(self):
        """A failed extraction stays ACTIVE: the data may still be salvageable, so it is kept for retry."""
        node, resp, store = _run_agent(
            _FakeAgentResult(
                success=False,
                extracted_info={},
                error="empty response",
                error_code="EMPTY_RESPONSE",
            )
        )

        assert resp.extra["agent_status"] == "failed"
        assert node.status == MemoryStatus.ACTIVE, "a failed extraction must not degrade"
        assert all(s != MemoryStatus.SHADOW for _, s in store.upsert_calls)

    def test_successful_extract_still_shadows(self):
        """Regression protection: the degradation behaviour of a normal extraction with content is unchanged."""
        node, resp, store = _run_agent(
            _FakeAgentResult(
                success=True,
                extracted_info={"memory": [{"content": "The user is learning guitar", "layer": "L2_FACT"}]},
            )
        )

        assert resp.extra["agent_status"] == "success"
        assert resp.extra["agent_nodes"] >= 1
        assert node.status == MemoryStatus.SHADOW

    def test_explicit_non_raw_layer_is_not_shadowed(self):
        """When the caller explicitly names a non-raw layer it must not degrade -- that would logically delete the data it just wrote.

        The layer travels from metadata into extra (client.async_add puts metadata into extra verbatim),
        so an L2_FACT node simulates that scenario directly.

        Uses "extracted content" rather than "empty extraction": only the former reaches the degradation branch,
        which is what actually proves the layer guard works (with an empty extraction the old condition would not degrade either, so nothing is observable).
        """
        node, resp, store = _run_agent(
            _FakeAgentResult(
                success=True,
                extracted_info={"memory": [{"content": "The user is learning guitar", "layer": "L2_FACT"}]},
            ),
            node=_mk_node(layer=MemoryLayer.L2_FACT),
            extra={"layer": MemoryLayer.L2_FACT.value},
        )

        assert resp.extra["agent_status"] == "success"
        assert resp.extra["agent_nodes"] >= 1, "this case only means something if it reaches the degradation branch"
        assert node.status == MemoryStatus.ACTIVE, "a non-L1_RAW node must not be degraded"

    def test_shadow_upsert_failure_does_not_fail_write(self):
        """A failing degradation write only warns; it does not change the success verdict of this write."""
        node, resp, _ = _run_agent(
            _FakeAgentResult(success=True, extracted_info={"memory": []}),
            upsert_store=_FakeVectorStore(fail_upsert=True),
            request_id="req-shadow-fail",
        )

        # The failed degradation is swallowed by except, so the agent still reports success
        assert resp.extra["agent_status"] == "success"
        assert node.status == MemoryStatus.SHADOW, (
            "the status was already changed in memory (only the upsert failed) -- here we only assert the write verdict is unaffected"
        )


class TestShadowGuards:
    def test_no_memory_id_skips_shadow(self):
        """When the L1 write fails memory_id is an empty string, so degradation must not be attempted."""
        node, resp, store = _run_agent(
            _FakeAgentResult(success=True, extracted_info={"memory": []}),
            memory_id="",
            request_id="req-no-mid",
        )

        assert resp.extra["agent_status"] == "success"
        assert node.status == MemoryStatus.ACTIVE
        assert all(s != MemoryStatus.SHADOW for _, s in store.upsert_calls)
