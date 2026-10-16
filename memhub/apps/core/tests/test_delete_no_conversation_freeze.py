"""删掉一条记忆，不得让同一对话之后的新 turn 停止沉淀（RC10）。

沙箱真 HY 实验里实测过：人删掉一条记忆后，同一对话之后的新消息也不再成为记忆。
处理一段对话时，每个 chunk 都要过一遍上下文预检；已处理过的旧 turn 的证据
正托着被删的那条记忆，于是预检在旧 chunk 上抛 evidence_changed，整段对话中止，
新 turn 连同它一起被丢下。live 路径还把这个错误吞成一行日志，实时同步就此静默停住。

删除删的是那条记忆实体，不是"永远不许再说这件事"：
新 turn 是新的陈述，应当照常沉淀；同一段旧原话（按证据 id）不得把已删记忆抽回。

三条入口各守一条：即时 prepare、live 到达、backfill 回放。
全部走真实的临时 Core、真实的连续处理授权与守卫，只把 HY 换成可控的替身。
"""

import asyncio

import pytest

from one_memory.imports.models import WrittenBy
from one_memory.imports.runner import live_sync_arrival
from one_memory.memory_governance import MemoryStatus
from one_memory.memory_tasks import MemoryTaskStatus
from tests.test_continuous_processing_integration import IntegrationCore


@pytest.fixture
def core(tmp_path, monkeypatch):
    return IntegrationCore(tmp_path, monkeypatch)


def _memory_for(core, record):
    return core.app.state.memory_governance_store.get(
        "integration-hy-" + record.evidence_id, user_id="local"
    )


async def _first_turn_then_delete(core):
    """第一个 turn 沉淀成记忆，然后人把它删掉。返回那一轮的原话与连续授权。"""
    core.grant_source()
    await core.save_and_probe()
    receipt = core.kernel_enable()
    first = core.put_complete("first-turn")
    results = await core.run_prepared(first)
    assert [t.status for t in results] == [MemoryTaskStatus.COMPLETED]
    memory = _memory_for(core, first)
    # 正向前提：第一轮确实沉淀出了一条可删的记忆。
    assert memory is not None and memory.status is MemoryStatus.CONFIRMED
    forgotten = []

    async def forget_in_hy(memory_id):
        # HY 那头的遗忘另有守护；这里只要治理侧的删除照常落地。
        forgotten.append(memory_id)
        return {"deleted": True}

    core.app.state.memory_processing_supervisor.delete_hy_memory = forget_in_hy
    deleted = await core.client.post(
        f"/api/v1/memories/{memory.memory_id}/actions/delete",
        json={"request_id": "delete-first", "expected_revision": memory.revision},
    )
    assert deleted.status_code == 200, deleted.text
    assert _memory_for(core, first).status is MemoryStatus.DELETED
    assert forgotten == [memory.memory_id]
    return first, receipt


def test_prepare_after_a_deletion_still_settles_the_new_turn(core):
    async def scenario():
        async with core.running():
            first, _receipt = await _first_turn_then_delete(core)
            second = core.put_complete("second-turn")
            supervisor = core.app.state.memory_processing_supervisor

            prepared = await asyncio.to_thread(supervisor.prepare_conversation, second)
            # 已完成的旧 turn 在合并前剔除、不再预检重跑；只有新 turn 是待做的。
            by_input = {t.input_ref: t for t in prepared}
            assert set(by_input) == {second.evidence_id}
            pending = [t for t in prepared if t.status is not MemoryTaskStatus.COMPLETED]
            assert [t.input_ref for t in pending] == [second.evidence_id]
            for task in pending:
                await supervisor._create_runner().run(task.task_id)

            memory = _memory_for(core, second)
            assert memory is not None and memory.status is MemoryStatus.CONFIRMED
            assert _memory_for(core, first).status is MemoryStatus.DELETED
            # 新 turn 照常处理，但被删的记忆和它托着的原话都不进这一轮的上下文。
            sent = next(r for r in core.bridge.requests if second.evidence_id in r.evidence_ids)
            context = sent.source_context
            assert context is not None
            assert first.evidence_id not in {
                r.evidence_id for r in (*context.history, *context.supporting)
            }
            deleted_id = "integration-hy-" + first.evidence_id
            assert deleted_id not in {m.governed_memory_id for m in context.known_memories}

    asyncio.run(scenario())


async def _live_settled(supervisor, tasks):
    """等防抖、准备与派发都走完，而不是等固定秒数：慢机器上也不会早判。"""
    loop = asyncio.get_running_loop()
    end = loop.time() + 30
    while loop.time() < end:
        idle = not supervisor._debounces and not supervisor._preparations
        busy = any(
            t.status in {MemoryTaskStatus.PENDING, MemoryTaskStatus.RUNNING} for t in tasks.list()
        )
        if idle and not busy:
            return
        await asyncio.sleep(0.05)
    pytest.fail("live 到达的准备与派发没有在界限内走完")


def test_live_arrival_after_a_deletion_still_settles_the_new_turn(core):
    async def scenario():
        async with core.running():
            first, _receipt = await _first_turn_then_delete(core)
            second = core.put("second-turn")
            reply = core.put(
                "second-turn-reply",
                conversation="same-conversation",
                role="assistant",
                written_by="agent",
            )
            supervisor = core.app.state.memory_processing_supervisor
            assert _memory_for(core, second) is None

            token = live_sync_arrival.set(True)
            try:
                supervisor.enqueue_evidence(second)
                supervisor.enqueue_evidence(reply)
            finally:
                live_sync_arrival.reset(token)
            await _live_settled(supervisor, core.app.state.memory_task_store)

            memory = _memory_for(core, second)
            assert memory is not None and memory.status is MemoryStatus.CONFIRMED
            assert _memory_for(core, first).status is MemoryStatus.DELETED

    asyncio.run(scenario())


@pytest.mark.parametrize("deleted", [True, False], ids=["deleted", "control-kept"])
def test_reprocessing_the_same_old_evidence_does_not_bring_a_deleted_memory_back(core, deleted):
    """按证据 id 挡住：同一段旧原话再交给 HY 一次，也不得重新长出被删的那条记忆。

    对照组不删：同一条重跑路径确实会把同一段原话再抽成一条新记忆——
    证明挡住它的是删除，而不是这条路径本来就到不了写入。
    """

    async def scenario():
        async with core.running():
            if deleted:
                first, _receipt = await _first_turn_then_delete(core)
            else:
                core.grant_source()
                await core.save_and_probe()
                core.kernel_enable()
                first = core.put_complete("first-turn")
                await core.run_prepared(first)
            original = core.bridge.ingest

            async def reextract(request):
                # 真 HY 重抽同一段原话时给的是新 id，内容相同。
                result = await original(request)
                return result.model_copy(
                    update={
                        "memories": tuple(
                            m.model_copy(update={"memory_id": "reextracted-" + m.memory_id})
                            for m in result.memories
                        )
                    }
                )

            core.bridge.ingest = reextract
            tasks = core.app.state.memory_task_store
            done = next(t for t in tasks.list() if t.input_ref == first.evidence_id)
            tasks.requeue_completed(done.task_id, reason="reprocess_same_evidence")

            finished = await core.app.state.memory_processing_supervisor._create_runner().run(
                done.task_id
            )

            back = core.app.state.memory_governance_store.get(
                "reextracted-integration-hy-" + first.evidence_id, user_id="local"
            )
            if deleted:
                assert back is None or back.status is not MemoryStatus.CONFIRMED
                assert finished.status is not MemoryTaskStatus.COMPLETED
            else:
                assert finished.status is MemoryTaskStatus.COMPLETED, finished.last_error_code
                assert back is not None and back.status is MemoryStatus.CONFIRMED

    asyncio.run(scenario())


def test_backfill_after_a_deletion_still_settles_the_new_turn(core):
    """backfill 修复前就不冻结：已完成的旧 chunk 在入队时跳过，出错的 chunk 也只丢它自己。

    只拿掉其中一道它仍是绿的；两道都拿掉时才红（实测）。
    它守的是 backfill 这一路的结果，不单独守"已完成跳过"——那一道由 prepare 的守卫看住。
    """

    async def scenario():
        async with core.running():
            first, receipt = await _first_turn_then_delete(core)
            second = core.put_complete("second-turn")
            core.put(
                "second-turn-answer",
                conversation=second.conversation_id,
                role="assistant",
                written_by=WrittenBy.AGENT,
            )
            supervisor = core.app.state.memory_processing_supervisor
            assert _memory_for(core, second) is None

            await supervisor.backfill_off_thread(expected_permission_revision=receipt.revision)
            await core.wait_completed(minimum_tasks=2)

            memory = _memory_for(core, second)
            assert memory is not None and memory.status is MemoryStatus.CONFIRMED
            assert _memory_for(core, first).status is MemoryStatus.DELETED

    asyncio.run(scenario())
