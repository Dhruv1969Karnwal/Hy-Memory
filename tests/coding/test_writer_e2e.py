# -*- coding: utf-8 -*-
"""
End-to-end write-path tests.

Chains judge / preproc / extractor / reconciler / store together to
verify CodingWriter coordinates them correctly.
"""

import json

import pytest

from hy_memory.coding.extractor import CodingMemoryExtractor
from hy_memory.coding.reconciler import CodingMemoryReconciler
from hy_memory.coding.writer import CodingWriter, is_coding_segment
from hy_memory.pipelines.base import ChatMessage, ToolCall

from tests.coding._helpers import MockLLMProvider, FailingLLMProvider


def _classify_resp(is_coding):
    return json.dumps({"is_coding": is_coding})


def _extract_resp(experiences, fact=None):
    if isinstance(fact, list):
        if len(fact) <= 1:
            fact = fact[0] if fact else None
        else:
            # Compatibility regression only: old models may still return a multi-element facts array.
            return json.dumps({"experiences": experiences, "facts": fact})
    return json.dumps({"experiences": experiences, "fact": fact})


def _reconcile_resp(ops):
    return json.dumps({"ops": ops})


def _coding_messages():
    return [
        ChatMessage(role="user", content="Help me package and publish to the private package index"),
        ChatMessage(role="assistant", content="", tool_calls=[
            ToolCall(id="1", name="Read", arguments={"path": "scripts/publish_package.sh"}),
        ]),
        ChatMessage(role="tool", content="200 lines", tool_call_id="1", tool_name="Read"),
        ChatMessage(role="assistant", content="", tool_calls=[
            ToolCall(id="2", name="Bash", arguments={"command": "twine upload"}),
        ]),
        ChatMessage(role="tool", content="OK uploaded", tool_call_id="2", tool_name="Bash"),
    ]


@pytest.mark.asyncio
class TestIsCodingSegment:
    async def test_no_tool_short_circuit(self):
        msgs = [ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="hello")]
        # even if the LLM returns true, has_any_tool_message False must short-circuit to False
        llm = MockLLMProvider('{"is_coding": true}')
        result = await is_coding_segment(msgs, llm)
        assert result is False
        # the LLM must not be called
        assert len(llm.calls) == 0

    async def test_with_tool_classify_coding(self):
        llm = MockLLMProvider(_classify_resp(True))
        result = await is_coding_segment(_coding_messages(), llm)
        assert result is True

    async def test_with_tool_classify_chat(self):
        llm = MockLLMProvider(_classify_resp(False))
        result = await is_coding_segment(_coding_messages(), llm)
        assert result is False


@pytest.mark.asyncio
class TestCodingWriterE2E:
    async def test_trace_history_shared_across_working_and_coding_variants(
        self, coding_config, mock_embed,
    ):
        from hy_memory.coding.store import CodingMemoryStore

        working = CodingMemoryStore(
            coding_config, embed_service=mock_embed, variant="working",
        )
        coding = CodingMemoryStore(
            coding_config, embed_service=mock_embed, variant="coding",
        )
        await working.initialize()
        await coding.initialize()
        try:
            await working.append_trace(
                [ChatMessage(role="user", content="Sort out the financial reports first")],
                request_id="shared-trace", user_id="u1", agent_id="a1",
            )
            recent = await coding.get_recent_trace_messages(
                user_id="u1", agent_id="a1", max_user_turns=5,
            )
            assert [m.content for m in recent if m.role == "user"] == [
                "Sort out the financial reports first"
            ]
            assert working._db_path != coding._db_path
            assert working._trace_db_path == coding._trace_db_path
        finally:
            await coding.close()
            await working.close()

    async def test_trace_history_keeps_last_five_user_turns(self, coding_store):
        msgs = []
        for i in range(7):
            msgs.extend([
                ChatMessage(role="user", content=f"history-{i}"),
                ChatMessage(role="assistant", content="A" * 8000),
                ChatMessage(role="tool", content="T" * 5000, tool_name="Read"),
            ])
        await coding_store.append_trace(
            msgs, request_id="trace-1", user_id="u1", agent_id="a1",
        )
        recent = await coding_store.get_recent_trace_messages(
            user_id="u1", agent_id="a1", max_user_turns=5,
        )
        assert [m.content for m in recent if m.role == "user"] == [
            "history-2", "history-3", "history-4", "history-5", "history-6"
        ]
        assert all("chars omitted" in m.content for m in recent if m.role == "assistant")
        assert all("bytes omitted" in m.content for m in recent if m.role == "tool")

    async def test_trace_history_retention_is_bounded(self, coding_store):
        for i in range(105):
            await coding_store.append_trace(
                [ChatMessage(role="user", content=f"turn-{i}")],
                request_id=f"trace-{i}", user_id="u1", agent_id="a1",
            )
        with coding_store._trace_lock:
            count = coding_store._trace_conn.execute(
                "SELECT COUNT(*) AS n FROM coding_trace WHERE user_id=? AND agent_id=?",
                ("u1", "a1"),
            ).fetchone()["n"]
        assert count == 100

    async def test_full_path_add(self, coding_store, mock_embed):
        """Full e2e: tool messages present -> judge=True -> extract -> reconcile=ADD -> persisted"""
        # One LLM mock sequence answers extract + reconcile (note: we call
        # writer.write directly here, skipping the judge, so the sequence only covers extract + reconcile)
        ext_resp = _extract_resp([{
            "task": "Package and publish to the private package index",
            "search_keys": ["credential location", "upload command"],
            "solution": "Run scripts/publish_package.sh X.Y.Z; credentials are provided via environment variables",
            "boundary_envs": "twine>=6\\nurl: packages.example.com",
            "confidence": 0.9,
        }])
        rec_resp = _reconcile_resp([{"draft_idx": 0, "action": "ADD"}])
        llm = MockLLMProvider([ext_resp, rec_resp])

        ext = CodingMemoryExtractor(llm)
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        writer = CodingWriter(
            store=coding_store, extractor=ext, reconciler=rec,
            llm_provider=llm, embed_service=mock_embed,
        )
        resp = await writer.write(
            _coding_messages(),
            user_id="u1", agent_id="a1",
            workspace_id="github.com/x/y", branch="main",
            session_id="s1",
        )
        assert resp["success"] is True
        assert resp["scene"] == "productivity"
        assert len(resp["ops"]) == 1
        assert resp["ops"][0]["action"] == "ADD"
        assert len(resp["experience_ids"]) == 1

        # verify persistence
        experience_id = resp["experience_ids"][0]
        got = await coding_store.get_by_id(experience_id, user_id="u1")
        assert got is not None
        assert got.task == "Package and publish to the private package index"
        assert got.search_keys == ["credential location", "upload command"]
        assert got.boundary_envs.startswith("twine>=6")
        assert got.workspace_id == "github.com/x/y"
        assert got.files == ["scripts/publish_package.sh"]

    async def test_value_bar_returns_empty(self, coding_store, mock_embed):
        """Extractor returns 0 memories (value bar not met)"""
        llm = MockLLMProvider(_extract_resp([]))
        ext = CodingMemoryExtractor(llm)
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        writer = CodingWriter(
            store=coding_store, extractor=ext, reconciler=rec,
            llm_provider=llm, embed_service=mock_embed,
        )
        resp = await writer.write(
            _coding_messages(), user_id="u1", workspace_id="ws/a",
        )
        assert resp["success"] is True
        assert resp["ops"] == []
        assert resp["experience_ids"] == []

    async def test_no_workspace_still_writes(self, coding_store, mock_embed):
        """A missing workspace_id no longer blocks drafts; both drafts must persist"""
        ext_resp = _extract_resp([
            {"task": "T1", "solution": "S", "confidence": 0.9},
            {"task": "T2", "solution": "S", "confidence": 0.9},
        ])
        rec_resp = _reconcile_resp([
            {"draft_idx": 0, "action": "ADD"},
            {"draft_idx": 1, "action": "ADD"},
        ])
        llm = MockLLMProvider([ext_resp, rec_resp])
        ext = CodingMemoryExtractor(llm)
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        writer = CodingWriter(
            store=coding_store, extractor=ext, reconciler=rec,
            llm_provider=llm, embed_service=mock_embed,
        )
        resp = await writer.write(
            _coding_messages(), user_id="u1", workspace_id=None,
        )
        assert len(resp["experience_ids"]) == 2
        assert sorted(await coding_store.list_user_tasks("u1")) == ["T1", "T2"]

    async def test_extractor_failure(self, coding_store, mock_embed):
        """Extractor LLM fails -> empty drafts -> empty ops"""
        ext = CodingMemoryExtractor(FailingLLMProvider())
        rec = CodingMemoryReconciler(coding_store, mock_embed, MockLLMProvider("{}"))
        writer = CodingWriter(
            store=coding_store, extractor=ext, reconciler=rec,
            llm_provider=FailingLLMProvider(), embed_service=mock_embed,
        )
        resp = await writer.write(_coding_messages(), user_id="u1", workspace_id="ws")
        assert resp["success"] is True
        assert resp["ops"] == []

    async def test_existing_tasks_not_shown_to_extractor(self, coding_store, mock_embed):
        """Tasks already in the store must not enter the extraction prompt -- extraction only reads this segment; dedup is reconcile's job"""
        from hy_memory.coding.types import CodingMemory
        await coding_store.insert(CodingMemory(
            experience_id="e_old", user_id="u1", task="old task", solution="S",
        ))

        llm = MockLLMProvider([_extract_resp([]), _reconcile_resp([])])
        ext = CodingMemoryExtractor(llm)
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        writer = CodingWriter(
            store=coding_store, extractor=ext, reconciler=rec,
            llm_provider=llm, embed_service=mock_embed,
        )
        await writer.write(_coding_messages(), user_id="u1", workspace_id="ws/a")
        assert "old task" not in llm.calls[0]["prompt"]


@pytest.mark.asyncio
class TestDualLayerWrite:
    """Dual-layer write + full crosslinking within one batch (section 4)"""

    def _writer(self, store, embed, llm, *, verbatim_fallback_enabled=False):
        return CodingWriter(
            store=store,
            extractor=CodingMemoryExtractor(
                llm,
                verbatim_fallback_enabled=verbatim_fallback_enabled,
            ),
            reconciler=CodingMemoryReconciler(store, embed, llm),
            llm_provider=llm, embed_service=embed,
        )

    async def test_both_layers_written_and_crosslinked(self, coding_store, mock_embed):
        llm = MockLLMProvider([
            _extract_resp(
                [{"task": "Package and publish to the private package index", "solution": "S", "confidence": 0.9}],
                [{"statement": "The publish script is at scripts/publish_package.sh", "files": ["scripts/publish_package.sh"]}],
            ),
            _reconcile_resp([{"draft_idx": 0, "action": "ADD"}]),
        ])
        writer = self._writer(coding_store, mock_embed, llm)

        resp = await writer.write(
            _coding_messages(), user_id="u1", workspace_id="ws/a", session_id="s1",
        )

        assert len(resp["experience_ids"]) == 1
        assert len(resp["fact_ids"]) == 1
        assert resp["fact_ops_summary"] == {"insert": 1, "supersede": 0, "total": 1}

        eid, fid = resp["experience_ids"][0], resp["fact_ids"][0]
        exp = await coding_store.get_by_id(eid, user_id="u1")
        fact = await coding_store.get_fact(fid, user_id="u1")
        assert exp.source_fact_ids == [fid]
        assert fact.derived_experience_ids == [eid]
        assert fact.statement == "The publish script is at scripts/publish_package.sh"

    async def test_all_experiences_linked_to_unique_fact(self, coding_store, mock_embed):
        """Every experience from one trace is bidirectionally linked to the single session fact."""
        llm = MockLLMProvider([
            _extract_resp(
                [{"task": "T1", "solution": "S"}, {"task": "T2", "solution": "S"}],
                [{"statement": "Fact A"}, {"statement": "Fact B"}],
            ),
            _reconcile_resp([
                {"draft_idx": 0, "action": "ADD"},
                {"draft_idx": 1, "action": "ADD"},
            ]),
        ])
        writer = self._writer(coding_store, mock_embed, llm)

        resp = await writer.write(_coding_messages(), user_id="u1")
        eids, fids = resp["experience_ids"], resp["fact_ids"]
        assert len(eids) == 2 and len(fids) == 1

        for eid in eids:
            exp = await coding_store.get_by_id(eid, user_id="u1")
            assert exp.source_fact_ids == fids
        fact = await coding_store.get_fact(fids[0], user_id="u1")
        assert sorted(fact.derived_experience_ids) == sorted(eids)

    async def test_skip_not_linked(self, coding_store, mock_embed):
        """A SKIPped experience was never really persisted, so it must not be linked to this batch of facts."""
        llm = MockLLMProvider([
            _extract_resp(
                [{"task": "T1", "solution": "S"}],
                [{"statement": "Fact A"}],
            ),
            _reconcile_resp([{"draft_idx": 0, "action": "SKIP", "reason": "dup"}]),
        ])
        writer = self._writer(coding_store, mock_embed, llm)

        resp = await writer.write(_coding_messages(), user_id="u1")
        assert resp["experience_ids"] == []
        assert len(resp["fact_ids"]) == 1
        fact = await coding_store.get_fact(resp["fact_ids"][0], user_id="u1")
        assert fact.derived_experience_ids == []

    async def test_update_keeps_old_fact_link(self, coding_store, mock_embed):
        """UPDATE overwrites the body without keeping a trace, but fact links take the union; old links survive."""
        from hy_memory.coding.types import CodingMemory
        await coding_store.insert(CodingMemory(
            experience_id="e1", user_id="u1", task="old task", solution="old solution",
            source_fact_ids=["f_before"],
        ))
        llm = MockLLMProvider([
            _extract_resp(
                [{"task": "old task", "solution": "new solution"}],
                [{"statement": "New fact from this round"}],
            ),
            _reconcile_resp([
                {"draft_idx": 0, "action": "UPDATE", "target_experience_id": "e1"},
            ]),
        ])
        writer = self._writer(coding_store, mock_embed, llm)

        resp = await writer.write(_coding_messages(), user_id="u1")
        exp = await coding_store.get_by_id("e1", user_id="u1")
        assert exp.solution == "new solution"
        assert exp.source_fact_ids == ["f_before", resp["fact_ids"][0]]

    async def test_facts_only_no_experience(self, coding_store, mock_embed):
        """Experience layer empty, fact layer non-empty: facts still persist, experience-layer reconcile never runs."""
        llm = MockLLMProvider(_extract_resp([], [{"statement": "Facts only"}]))
        writer = self._writer(coding_store, mock_embed, llm)

        resp = await writer.write(_coding_messages(), user_id="u1")
        assert resp["experience_ids"] == []
        assert len(resp["fact_ids"]) == 1
        # extract once; no candidates in the fact-layer store so no LLM call; experience-layer reconcile never ran
        assert len(llm.calls) == 1

    async def test_empty_experiences_do_not_persist_verbatim_fact_by_default(
        self, coding_store, mock_embed,
    ):
        llm = MockLLMProvider(_extract_resp([], []))
        writer = self._writer(coding_store, mock_embed, llm)

        resp = await writer.write(_coding_messages(), user_id="u1")
        assert resp["ops"] == []
        assert resp["fact_ids"] == []
        assert resp["fact_ops"] == []

    async def test_empty_experiences_can_persist_verbatim_fact_when_enabled(
        self, coding_store, mock_embed,
    ):
        """With the switch explicitly on, an empty extraction on both layers still persists one verbatim fact.

        End-to-end confirmation that with the switch on, the fallback gets through reconcile and really persists.
        """
        llm = MockLLMProvider(_extract_resp([], []))
        writer = self._writer(
            coding_store,
            mock_embed,
            llm,
            verbatim_fallback_enabled=True,
        )

        resp = await writer.write(_coding_messages(), user_id="u1")
        assert resp["ops"] == []            # the experience layer gets no fallback
        assert len(resp["fact_ids"]) == 1
        assert [op["action"] for op in resp["fact_ops"]] == ["INSERT"]

        fact = await coding_store.get_fact(resp["fact_ids"][0], user_id="u1")
        assert fact.statement == "Help me package and publish to the private package index"
        assert "twine upload" in fact.details      # the raw trace text went into details
