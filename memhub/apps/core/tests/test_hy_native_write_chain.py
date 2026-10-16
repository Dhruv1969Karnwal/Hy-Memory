"""HY writes with its own prompts and judgment; the product adds no audit of its own."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from hy_memory.agent.reconciler import ReconcileOp, ReconcileResult

from one_memory.hy_response_guard import HyReconcileGuard
from one_memory.hy_source_context import HySourceContext, HySupportedMemory, HyWriteBindings
from one_memory.hy_write_verification import HyExtractionGuard
from one_memory.imports.storage import EvidenceStore
from tests.test_hy_permission_raw_context import (
    ScriptedOracle,
    assert_outputs,
    bind_oracle,
    native_client,
    raw_client,
)
from tests.test_hy_source_context import source

PRODUCT_RULE_MARKERS = ("# 来源与语气保真", "# 原始发言主体规则", "owner记录命题的陈述权威")


def test_a_raw_turn_reaches_hy_with_its_native_prompts_and_no_product_audit(tmp_path):
    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, _):
            task = supervisor.prepare_conversation(records[0])[0]
            oracle = ScriptedOracle(records)
            bind_oracle(supervisor, oracle)
            result = await supervisor._create_runner().run(task.task_id)
            assert not oracle.errors, oracle.errors
            assert result.status == "completed", (result.last_error_code, oracle.stages)
            assert "extract" in oracle.stages
            assert "review" not in oracle.stages
            assert "citations" not in oracle.stages
            sent = json.dumps([body for _path, body in oracle.requests], ensure_ascii=False)
            for marker in PRODUCT_RULE_MARKERS:
                assert marker not in sent, marker
            # Each memory still points at the words of its own author in this turn.
            client = await native_client(supervisor, task)
            await assert_outputs(supervisor, task, records, client)

    asyncio.run(scenario())


def test_an_imported_memory_keeps_the_owner_hy_gives_it():
    class Native:
        received = None

        async def reconcile(self, **kwargs):
            self.received = kwargs["new_memories_with_meta"]
            return ReconcileResult(
                ops=[ReconcileOp(op="ADD", content="用户偏好简洁表达。", owner="user")]
            )

    native = Native()
    guard = HyReconcileGuard(native)
    guard.imported_contents = frozenset({"用户偏好简洁表达。"})
    result = asyncio.run(
        guard.reconcile(
            new_memories=["用户偏好简洁表达。"],
            new_memories_with_meta=[{"content": "用户偏好简洁表达。", "layer": "L2_FACT"}],
        )
    )
    assert "owner" not in native.received[0]
    assert result.ops[0].owner == "user"
    assert guard.failures == []


def bound(tmp_path, *, known=False):
    store = EvidenceStore(tmp_path / "evidence.db")
    old = source(store, "old", "user", "Atlas用本地存储。")
    current = source(store, "now", "user", "Atlas也要能离线同步。")
    memories = (
        HySupportedMemory(
            hy_memory_id="prior",
            governed_memory_id="prior",
            revision=1,
            content="用户要求Atlas使用本地存储。",
            owner="user",
            evidence_ids=(old.evidence_id,),
        ),
    )
    context = HySourceContext(
        current=(current,), history=(old,), known_memories=memories if known else ()
    )
    return HyWriteBindings(context), old, current


def test_the_turn_hy_extracts_must_be_the_turn_that_was_sent(tmp_path):
    bindings, _old, current = bound(tmp_path)
    calls = []

    class Native:
        async def extract(self, content, context, **kwargs):
            calls.append(content)
            return SimpleNamespace(success=True, info={"memory": []})

    guard = HyExtractionGuard(Native())
    guard.bindings = bindings
    sent = json.dumps([{"role": "user", "content": current.verbatim}], ensure_ascii=False)
    assert asyncio.run(guard.extract(sent, {})).success and calls == [sent]
    with pytest.raises(ValueError, match="input_mismatch"):
        asyncio.run(guard.extract(json.dumps([{"role": "user", "content": "别的内容"}]), {}))
    assert calls == [sent]
    assert guard.failures == ["hy_memory_write_verification_input_mismatch"]


class PriorStore:
    async def get_by_id(self, key):
        return SimpleNamespace(
            user_id="local", agent_id="atlas", owner="user", content="用户要求Atlas使用本地存储。"
        )


def merge_into_prior(content):
    class Native:
        async def reconcile(self, **kwargs):
            return ReconcileResult(
                ops=[ReconcileOp(op="UPDATE", memory_id="prior", owner="user", content=content)]
            )

    return Native()


ARGS = {"user_id": "local", "agent_id": "atlas", "vector_store": PriorStore()}


def test_hy_may_only_change_a_memory_this_write_was_allowed_to_see(tmp_path):
    allowed, _old, _current = bound(tmp_path / "allowed", known=True)
    guard = HyReconcileGuard(merge_into_prior("用户要求Atlas本地存储并支持离线同步。"))
    guard.bindings = allowed
    assert asyncio.run(guard.reconcile(**ARGS)).success

    unseen, _old, _current = bound(tmp_path / "unseen")
    guard = HyReconcileGuard(merge_into_prior("用户要求Atlas本地存储并支持离线同步。"))
    guard.bindings = unseen
    with pytest.raises(ValueError, match="target_unbound"):
        asyncio.run(guard.reconcile(**ARGS))


def test_a_memory_hy_merges_cites_this_turn_and_what_it_already_cited(tmp_path):
    bindings, old, current = bound(tmp_path, known=True)
    merged = "用户要求Atlas本地存储并支持离线同步。"
    guard = HyReconcileGuard(merge_into_prior(merged))
    guard.bindings = bindings
    assert asyncio.run(guard.reconcile(**ARGS)).success
    assert bindings.support_for(merged, "user") == (current.evidence_id, old.evidence_id)


HY_USER_ONLY_LINE = "只提取 NEW user messages 中首次出现的新信息。"


def _section(prompt, heading):
    _, found, rest = prompt.partition(heading)
    assert found, heading
    return rest.partition("\n---\n")[0]


def _served_extract_prompts(tmp_path):
    from one_memory.hy_memory_bridge import build_hy_memory_config
    from tests.test_memory_language import _profile

    config = build_hy_memory_config(_profile(), api_key=None, data_dir=tmp_path)
    return config.llm.extract_system_prompt_zh, config.llm.extract_system_prompt_en


def test_hys_native_extraction_only_reads_what_the_user_said():
    from hy_memory.agent.prompts_zh import EXTRACT_SYSTEM_PROMPT_ZH

    assert HY_USER_ONLY_LINE in EXTRACT_SYSTEM_PROMPT_ZH
    # HY already knows how to keep an assistant's words apart from the person's.
    assert '`"agent"` —— AI 助手' in _section(EXTRACT_SYSTEM_PROMPT_ZH, "# OWNER（归属）\n")


def test_hy_extracts_both_speakers_and_keeps_its_own_authorship_rules(tmp_path):
    from hy_memory.agent.prompts_zh import EXTRACT_SYSTEM_PROMPT_ZH as native

    for served in _served_extract_prompts(tmp_path):
        assert HY_USER_ONLY_LINE not in served
        principle = _section(served, "# 提取原则\n")
        assert "用户" in principle and "AI 助手" in principle
        assert "OWNER" in principle
        for heading in ("# OWNER（归属）\n", "# 主体规则\n"):
            assert _section(served, heading) == _section(native, heading), heading
        for marker in PRODUCT_RULE_MARKERS:
            assert marker not in served, marker


def test_a_reworded_extraction_line_still_asks_for_both_speakers(tmp_path, monkeypatch, caplog):
    from hy_memory.agent.prompts_zh import EXTRACT_SYSTEM_PROMPT_ZH as native

    from one_memory import hy_memory_bridge

    class Reworded:
        EXTRACT_SYSTEM_PROMPT_ZH = native.replace(HY_USER_ONLY_LINE, "HY 换了提取原则的措辞。")

    real_import = hy_memory_bridge.import_module
    monkeypatch.setattr(
        hy_memory_bridge,
        "import_module",
        lambda name: Reworded if name == "hy_memory.agent.prompts_zh" else real_import(name),
    )
    served, _ = _served_extract_prompts(tmp_path)
    assert "AI 助手提供的" in served
    assert "hy_extraction_scope_rule_not_found" in caplog.text
