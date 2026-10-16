"""Bounded factory races with a temporary native HY client and no model calls."""

import asyncio
import json
import threading

import httpx
import pytest
from hy_memory.models.memory import MemoryLayer, MemoryNode

from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.hy_permission_coding_store import HyPermissionCodingStore
from one_memory.hy_permission_context import _Wiring, permission_bridge_factory
from one_memory.hy_permission_store import HyPermissionVectorStore
from one_memory.hy_permission_transport import HyPermissionTransportError
from one_memory.memory_governance import MemoryDraft, MemoryKind
from tests.person_door import as_person
from tests.test_hy_permission_context import whole_client


def test_cancelled_start_cannot_install_wiring_after_factory_releases_client(tmp_path):
    async def scenario():
        with whole_client(tmp_path) as (supervisor, record, _, _):
            task = supervisor.prepare_conversation(record)[0]
            lease, request, _ = supervisor._create_runner()._prepare_permission_lease(task)
            bridge = supervisor._bridge(lease.profile, None, tmp_path)
            client = await bridge._get_client()
            native_vector, native_graph = client._vector_store, client._graph_store
            original_coding = client._config.coding.enable
            run = client._loop_thread.run
            submitted, release, completed = (threading.Event() for _ in range(3))
            worker_errors = []
            bound_contexts = []
            sends = []

            def mock_send(request):
                sends.append(request)
                pytest.fail("This startup-only race must not make any model request")

            def delayed_run(coroutine):
                if coroutine.cr_code is not _Wiring.enter.__code__:
                    return run(coroutine)
                # Model a worker being descheduled just before submitting to HY.
                submitted.set()
                try:
                    if not release.wait(5):
                        coroutine.close()
                        raise TimeoutError("test did not release startup worker")
                    return run(coroutine)
                except BaseException as error:
                    worker_errors.append(error)
                    raise
                finally:
                    completed.set()

            client._loop_thread.run = delayed_run
            factory = permission_bridge_factory(
                supervisor._bridge, http_transport=httpx.MockTransport(mock_send)
            )

            async def ingest():
                async with factory(lease, None, tmp_path) as bound:
                    bound_contexts.append(bound)
                    await bound.ingest_extracted(request, (record.verbatim,))

            pending = asyncio.create_task(ingest())
            try:
                assert await asyncio.to_thread(submitted.wait, 5)
                pending.cancel()
                # A fixed factory may keep cancellation pending until it has
                # joined startup. Let both that behavior and early exit proceed.
                await asyncio.wait({pending}, timeout=0.1)
                released_before_start = not bridge._access_lock.locked()
                release.set()
                assert await asyncio.to_thread(completed.wait, 5)
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(pending, 5)
                assert not worker_errors
                assert not sends
                assert bridge._access_owner is None
                assert not bridge._access_lock.locked()
                assert not isinstance(client._vector_store, HyPermissionVectorStore), (
                    "Cancelled startup installed a live permission wrapper after the "
                    "factory exited and released exclusive ownership; "
                    f"released_before_start={released_before_start}"
                )
                assert client._vector_store is native_vector
                assert client._graph_store is native_graph
                assert client._config.coding.enable == original_coding
            finally:
                release.set()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                assert await asyncio.to_thread(completed.wait, 5)
                client._loop_thread.run = run
                # The regression leaves finished=True with live wiring; force
                # cleanup directly so the failing test leaks no SDK resources.
                for bound in bound_contexts:
                    if bound.wiring is not None:
                        await asyncio.to_thread(run, bound.wiring.exit())
                await supervisor.shutdown()

    asyncio.run(scenario())


def test_completed_start_restores_client_and_revokes_saved_vector_reference(tmp_path):
    async def scenario():
        with whole_client(tmp_path) as (supervisor, record, _, _):
            task = supervisor.prepare_conversation(record)[0]
            lease, request, _ = supervisor._create_runner()._prepare_permission_lease(task)
            bridge = supervisor._bridge(lease.profile, None, tmp_path)
            client = await bridge._get_client()
            vector, graph = client._vector_store, client._graph_store
            factory = permission_bridge_factory(
                supervisor._bridge,
                http_transport=httpx.MockTransport(lambda _: pytest.fail("unexpected send")),
            )
            try:
                async with factory(lease, None, tmp_path) as bound:
                    await bound._start(request)
                    wrapped = client._vector_store
                    assert isinstance(wrapped, HyPermissionVectorStore)
                    assert client._graph_store is None
                    coding = client._coding_store
                    assert isinstance(coding, HyPermissionCodingStore)
                    assert client._coding_writer.reconciler.store is coding
                    assert client._cache is None
                assert client._vector_store is vector
                assert client._graph_store is graph
                assert client._coding_writer.reconciler.store is client._coding_store
                assert not isinstance(client._coding_store, HyPermissionCodingStore)
                assert client._cache is not None
                with pytest.raises(HyMemoryBridgeError, match="hy_store_closed"):
                    wrapped.recheck()
                with pytest.raises(HyMemoryBridgeError, match="hy_coding_store_closed"):
                    coding.recheck()
            finally:
                await supervisor.shutdown()

    asyncio.run(scenario())


def test_graph_disabled_keeps_graph_absent_even_for_cross_evidence_operations(tmp_path):
    async def scenario():
        with whole_client(tmp_path) as (supervisor, record, _, _):
            task = supervisor.prepare_conversation(record)[0]
            lease, request, _ = supervisor._create_runner()._prepare_permission_lease(task)
            bridge = supervisor._bridge(lease.profile, None, tmp_path)
            client = await bridge._get_client()
            native_graph = client._graph_store
            assert native_graph is None
            factory = permission_bridge_factory(
                supervisor._bridge,
                http_transport=httpx.MockTransport(lambda _: pytest.fail("unexpected send")),
            )
            try:
                async with factory(lease, None, tmp_path) as bound:
                    await bound._start(request)
                    assert client._graph_store is None
                    assert bound.wiring is not None and bound.wiring.graph is None
                assert client._graph_store is native_graph

                async with factory(lease, None, tmp_path) as digest_bound:
                    await digest_bound._start_operation(
                        user_id=request.user_id,
                        agent_id=request.agent_id,
                        request_id="digest-test",
                        session_id="digest:test",
                        cross_evidence_scope=True,
                    )
                    assert client._graph_store is None
                    assert digest_bound.wiring is not None
                    assert digest_bound.wiring.graph is None
                assert client._graph_store is native_graph
            finally:
                await supervisor.shutdown()

    asyncio.run(scenario())


def test_persisted_native_proof_does_not_authorize_deleted_governed_memory(tmp_path):
    async def scenario():
        with whole_client(tmp_path) as (supervisor, record, permissions, permission):
            runner = supervisor._create_runner()
            task = supervisor.prepare_conversation(record)[0]
            lease, request, _ = runner._prepare_permission_lease(task)
            factory = permission_bridge_factory(
                supervisor._bridge,
                http_transport=httpx.MockTransport(lambda _: pytest.fail("unexpected send")),
            )
            bridge = supervisor._bridge(lease.profile, None, tmp_path)
            client = await bridge._get_client()
            run = client._loop_thread.run
            native = MemoryNode(
                node_id="synthetic-governed-memory",
                user_id=request.user_id,
                agent_id=request.agent_id,
                session_id=request.session_id,
                content=record.verbatim,
                owner="agent",
                layer=MemoryLayer.L2_FACT,
                embedding=[1.0] + [0.0] * 383,
            )
            try:
                # Seed only through the guarded native store; profile readback
                # persists the factory's own proof ledger, as in a completed run.
                async with factory(lease, None, tmp_path) as first:
                    await first._start(request)
                    store = first.wiring.vector
                    assert await asyncio.to_thread(run, store.upsert(native)) == native.node_id
                    positive = await asyncio.to_thread(run, store.get_by_id(native.node_id))
                    assert positive.content == record.verbatim
                    await first.profile_memories(user_id=request.user_id, agent_id=request.agent_id)
                governed = as_person(supervisor.governance).create(
                    MemoryDraft(
                        user_id=request.user_id,
                        kind=MemoryKind.CHAT,
                        content=record.verbatim,
                        scopes=("synthetic-scope",),
                        evidence_ids=(record.evidence_id,),
                    ),
                    request_id="synthetic-publication",
                    memory_id=native.node_id,
                )
                rejected = as_person(supervisor.governance).delete(
                    governed.memory_id,
                    user_id=request.user_id,
                    request_id="synthetic-rejection",
                    expected_revision=governed.revision,
                )
                assert rejected.status == "deleted"
                # Explicit renewed consent permits the unchanged source text,
                # but must not reinstate a deleted derived native memory.
                snapshot = permissions.snapshot(
                    user_id=request.user_id,
                    source_ids=(record.source_id,),
                    profile=lease.profile,
                    evidence=(record,),
                )
                permissions.grant(
                    snapshot, user_confirmed=True, expected_revision=permission.revision
                )
                next_task = supervisor.prepare_conversation(record)[0]
                assert next_task.task_id != task.task_id
                next_lease, next_request, _ = runner._prepare_permission_lease(next_task)
                assert next_lease.source_context is None
                assert next_lease.recheck()
                async with factory(next_lease, None, tmp_path) as second:
                    await second._start(next_request)
                    reused = await asyncio.to_thread(
                        run, second.wiring.vector.get_by_id(native.node_id)
                    )
                    assert reused is None, (
                        "Evidence-only proof reload exposed a deleted governed memory "
                        "as native context in a new permitted task"
                    )
            finally:
                await supervisor.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize("rejection_point", ["after_start", "after_proof_load"])
def test_deletion_during_proof_binding_blocks_next_physical_send(tmp_path, rejection_point):
    async def scenario():
        with whole_client(tmp_path) as (supervisor, record, _, _):
            task = supervisor.prepare_conversation(record)[0]
            lease, request, _ = supervisor._create_runner()._prepare_permission_lease(task)
            bridge = supervisor._bridge(lease.profile, None, tmp_path)
            client = await bridge._get_client()
            run = client._loop_thread.run
            sends = []

            def handler(outbound):
                body = json.loads(outbound.content)
                sends.append(body)
                return httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "model": "test-embedding",
                        "data": [
                            {
                                "object": "embedding",
                                "index": i,
                                "embedding": [1.0] + [0.0] * 383,
                            }
                            for i, _ in enumerate(body["input"])
                        ],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    },
                )

            factory = permission_bridge_factory(
                supervisor._bridge, http_transport=httpx.MockTransport(handler)
            )
            native = MemoryNode(
                node_id="synthetic-rejection-race",
                user_id=request.user_id,
                agent_id=request.agent_id,
                session_id=request.session_id,
                content=record.verbatim,
                owner="agent",
                layer=MemoryLayer.L2_FACT,
                embedding=[1.0] + [0.0] * 383,
            )
            try:
                async with factory(lease, None, tmp_path) as first:
                    await first._start(request)
                    await asyncio.to_thread(run, first.wiring.vector.upsert(native))
                    permitted = await asyncio.to_thread(
                        run, first.wiring.vector.get_by_id(native.node_id)
                    )
                    assert permitted is not None
                    await asyncio.to_thread(
                        run, client._embed_service.embed(permitted.content, use_cache=False)
                    )
                    assert len(sends) == 1 and sends[0]["input"] == [record.verbatim]
                    await first.profile_memories(user_id=request.user_id, agent_id=request.agent_id)

                governed = as_person(supervisor.governance).create(
                    MemoryDraft(
                        user_id=request.user_id,
                        kind=MemoryKind.CHAT,
                        content=record.verbatim,
                        scopes=("synthetic-scope",),
                        evidence_ids=(record.evidence_id,),
                    ),
                    request_id="synthetic-race-publication",
                    memory_id=native.node_id,
                )

                def reject():
                    result = as_person(supervisor.governance).delete(
                        governed.memory_id,
                        user_id=request.user_id,
                        request_id="synthetic-race-rejection",
                        expected_revision=governed.revision,
                    )
                    assert result.status == "deleted"

                try:
                    async with factory(lease, None, tmp_path) as second:
                        if rejection_point == "after_proof_load":
                            load = second.proofs.load

                            def reject_after_load(current_lease):
                                approved = load(current_lease)
                                assert native.node_id in approved
                                # A governance write can commit between the two
                                # separately awaited load/guard database reads.
                                reject()
                                return approved

                            second.proofs.load = reject_after_load
                        await second._start(request)
                        if rejection_point == "after_start":
                            reject()
                        candidate = await asyncio.to_thread(
                            run, second.wiring.vector.get_by_id(native.node_id)
                        )
                        if candidate is not None:
                            await asyncio.to_thread(
                                run,
                                client._embed_service.embed(candidate.content, use_cache=False),
                            )
                except (HyMemoryBridgeError, HyPermissionTransportError) as error:
                    assert str(error).startswith("model_content_")
                assert (
                    supervisor.governance.get(native.node_id, user_id="local").status == "deleted"
                )
                assert len(sends) == 1, (
                    f"Deleted proof context reached the physical transport at {rejection_point}; "
                    f"expected one approved control send, observed {len(sends)} sends"
                )
            finally:
                await supervisor.shutdown()

    asyncio.run(scenario())
