# -*- coding: utf-8 -*-
"""Reconciler tests -- ADD / UPDATE / DELETE / SKIP / fallback"""

import json

import pytest

from hy_memory.coding.reconciler import CodingMemoryReconciler
from hy_memory.coding.types import CodingMemory, CodingMemoryDraft

from tests.coding._helpers import MockLLMProvider, FailingLLMProvider


def _llm_resp(ops):
    return json.dumps({"ops": ops})


@pytest.mark.asyncio
class TestReconcilerActions:
    async def test_all_add_on_empty_store(self, coding_store, mock_embed):
        llm = MockLLMProvider(_llm_resp([
            {"draft_idx": 0, "action": "ADD"},
            {"draft_idx": 1, "action": "ADD"},
        ]))
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        d1 = CodingMemoryDraft(task="T1", solution="S1", user_id="u1")
        d2 = CodingMemoryDraft(task="T2", solution="S2", user_id="u1")
        ops = await rec.reconcile([d1, d2])
        assert len(ops) == 2
        assert all(op.action == "ADD" for op in ops)
        assert all(op.target_experience_id for op in ops)
        tasks = await coding_store.list_user_tasks("u1", 10)
        assert sorted(tasks) == ["T1", "T2"]

    async def test_update(self, coding_store, mock_embed):
        # Seed m1
        await coding_store.insert(CodingMemory(
            experience_id="m1", user_id="u1", task="T1",
            search_keys=["old"], solution="old",
        ))
        llm = MockLLMProvider(_llm_resp([
            {"draft_idx": 0, "action": "UPDATE",
             "target_experience_id": "m1", "reason": "newer solution"},
        ]))
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        d = CodingMemoryDraft(task="T1", search_keys=["new1"], solution="new",
                              user_id="u1")
        ops = await rec.reconcile([d])
        assert len(ops) == 1 and ops[0].action == "UPDATE"
        got = await coding_store.get_by_id("m1", "u1")
        assert got.solution == "new"
        assert got.search_keys == ["new1"]

    async def test_skip(self, coding_store, mock_embed):
        await coding_store.insert(CodingMemory(
            experience_id="m1", user_id="u1", task="X", solution="Y",
        ))
        llm = MockLLMProvider(_llm_resp([
            {"draft_idx": 0, "action": "SKIP", "reason": "duplicate"},
        ]))
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        d = CodingMemoryDraft(task="X", solution="Y",
                              user_id="u1")
        ops = await rec.reconcile([d])
        assert len(ops) == 1 and ops[0].action == "SKIP"
        # The store is unchanged
        tasks = await coding_store.list_user_tasks("u1", 10)
        assert tasks == ["X"]

    async def test_add_with_delete_multi_op(self, coding_store, mock_embed):
        # The old memory is to be DELETEd
        await coding_store.insert(CodingMemory(
            experience_id="m_dep", user_id="u1", task="use old.sh",
            solution="old script",
        ))
        llm = MockLLMProvider(_llm_resp([
            {"draft_idx": 0, "action": "ADD"},
            {"draft_idx": 0, "action": "DELETE", "target_experience_id": "m_dep",
             "reason": "user explicitly deprecated old.sh"},
        ]))
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        d = CodingMemoryDraft(task="use new CLI", solution="new way",
                              user_id="u1")
        ops = await rec.reconcile([d])
        assert sorted(op.action for op in ops) == ["ADD", "DELETE"]
        # The old one is gone
        assert await coding_store.get_by_id("m_dep", "u1") is None
        # The new one is in
        tasks = await coding_store.list_user_tasks("u1", 10)
        assert "use new CLI" in tasks
        assert "use old.sh" not in tasks


@pytest.mark.asyncio
class TestReconcilerFallback:
    async def test_llm_failure_fallback_add_all(self, coding_store, mock_embed):
        rec = CodingMemoryReconciler(coding_store, mock_embed, FailingLLMProvider())
        d1 = CodingMemoryDraft(task="T1", solution="S", user_id="u1")
        d2 = CodingMemoryDraft(task="T2", solution="S", user_id="u1")
        ops = await rec.reconcile([d1, d2])
        assert len(ops) == 2 and all(op.action == "ADD" for op in ops)

    async def test_missing_draft_idx_fallback_add(self, coding_store, mock_embed):
        # The LLM only handled draft 0 and missed draft 1
        llm = MockLLMProvider(_llm_resp([{"draft_idx": 0, "action": "ADD"}]))
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        d1 = CodingMemoryDraft(task="T1", solution="S", user_id="u1")
        d2 = CodingMemoryDraft(task="T2", solution="S", user_id="u1")
        ops = await rec.reconcile([d1, d2])
        # T1 comes from the LLM ADD; T2 from the fallback ADD
        actions = [op.action for op in ops]
        assert actions.count("ADD") == 2

    async def test_invalid_op_filtered(self, coding_store, mock_embed):
        # An UPDATE without target_experience_id is dropped and draft_idx=0 falls back to ADD
        llm = MockLLMProvider(_llm_resp([
            {"draft_idx": 0, "action": "UPDATE"},  # Missing target -> dropped
        ]))
        rec = CodingMemoryReconciler(coding_store, mock_embed, llm)
        d = CodingMemoryDraft(task="T", solution="S", user_id="u1")
        ops = await rec.reconcile([d])
        # fallback ADD
        assert any(op.action == "ADD" for op in ops)

    async def test_empty_drafts(self, coding_store, mock_embed):
        rec = CodingMemoryReconciler(coding_store, mock_embed, MockLLMProvider("{}"))
        ops = await rec.reconcile([])
        assert ops == []

    async def test_garbage_llm_output_fallback(self, coding_store, mock_embed):
        rec = CodingMemoryReconciler(coding_store, mock_embed, MockLLMProvider("not json"))
        d = CodingMemoryDraft(task="T", solution="S", user_id="u1")
        ops = await rec.reconcile([d])
        # Parse failure -> fallback ADD
        assert any(op.action == "ADD" for op in ops)
