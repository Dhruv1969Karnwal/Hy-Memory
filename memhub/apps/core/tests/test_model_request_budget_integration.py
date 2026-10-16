"""Real HY leaf retries plus durable budget accounting; no network or provider."""

import asyncio
import sqlite3
import threading
from contextlib import suppress

import httpx
import pytest
from hy_memory.agent.llm_provider import LLMRetryError

from one_memory.config import Settings
from one_memory.database_access import close_main_databases
from one_memory.hy_permission_transport import HyPermissionTransport, HyPermissionTransportError
from one_memory.model_request_budget import (
    BudgetedModelTransport,
    ModelRequestBudget,
    ModelRequestBudgetError,
)
from tests.test_hy_permission_transport import Wire, lease, native
from tests.test_model_processing_permissions import Harness


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Real network is forbidden in budget integration tests")

    monkeypatch.setattr("socket.socket.connect", denied)
    monkeypatch.setattr("socket.getaddrinfo", denied)


def budget(database, limit=2, body_limit=4096):
    return ModelRequestBudget(
        database, request_limit=limit, body_byte_limit=body_limit, output_token_limit=64
    )


def reservations(database):
    with sqlite3.connect(database) as db:
        return db.execute(
            "SELECT sequence,body_bytes,output_tokens,reserved_at "
            "FROM model_request_budget_events ORDER BY sequence"
        ).fetchall()


@pytest.mark.parametrize("failure", ["http-status", "connection"])
def test_native_hy_retries_consume_durable_budget_and_latch_denial(tmp_path, failure):
    database = tmp_path / "budget.sqlite3"
    limits = budget(database)

    def fail(request, _number):
        if failure == "connection":
            raise httpx.ConnectError("synthetic connection failure", request=request)
        return httpx.Response(503)

    wire = Wire(fail)

    async def scenario():
        permission = lease()
        provider, _ = native(permission)
        provider._llm_config.max_retries = 6
        with pytest.raises(HyPermissionTransportError, match="budget_exceeded"):
            async with HyPermissionTransport(
                permission, http_transport=BudgetedModelTransport(limits, inner=wire)
            ) as transport:
                transport.bind_provider(provider)
                with pytest.raises(HyPermissionTransportError, match="budget_exceeded"):
                    await provider.complete("synthetic sensitive content", max_tokens=16)
                assert len(wire.requests) == 2
                assert transport.denied_code == "model_content_transport_budget_exceeded"
                assert len(reservations(database)) == 2
                for call in (
                    lambda: provider.complete("try again", max_tokens=1),
                    lambda: provider.complete_messages(
                        [{"role": "user", "content": "again"}], max_tokens=1
                    ),
                    transport.seal,
                ):
                    with pytest.raises(HyPermissionTransportError, match="budget_exceeded"):
                        await call()
                assert len(wire.requests) == 2

        # A fresh budget object and a fresh lease still see the consumed rows.
        fresh_permission = lease()
        fresh_provider, _ = native(fresh_permission)
        fresh_wire = Wire()
        with pytest.raises(HyPermissionTransportError, match="budget_exceeded"):
            async with HyPermissionTransport(
                fresh_permission,
                http_transport=BudgetedModelTransport(budget(database), inner=fresh_wire),
            ) as transport:
                transport.bind_provider(fresh_provider)
                await fresh_provider.complete("new lease", max_tokens=16)
        assert fresh_wire.requests == []

    asyncio.run(scenario())
    assert len(reservations(database)) == 2
    assert wire.closed == 1
    persisted = database.read_bytes()
    assert b"synthetic sensitive content" not in persisted
    assert b"synthetic-secret-never-log" not in persisted


def test_parallel_native_leases_share_one_budget_before_sending(tmp_path):
    database = tmp_path / "budget.sqlite3"
    budgets = [budget(database, limit=3) for _ in range(8)]
    wires = [Wire() for _ in budgets]

    async def attempt(limits, wire):
        permission = lease()
        provider, _ = native(permission)
        try:
            async with HyPermissionTransport(
                permission, http_transport=BudgetedModelTransport(limits, inner=wire)
            ) as transport:
                transport.bind_provider(provider)
                await provider.complete("parallel synthetic request", max_tokens=16)
            return True
        except HyPermissionTransportError as error:
            assert error.code == "model_content_transport_budget_exceeded"
            return False

    async def scenario():
        return await asyncio.gather(
            *(attempt(limits, wire) for limits, wire in zip(budgets, wires, strict=True))
        )

    assert sum(asyncio.run(scenario())) == 3
    assert sum(len(wire.requests) for wire in wires) == len(reservations(database)) == 3


def test_cancelled_physical_attempt_is_not_refunded_on_reopen(tmp_path):
    database = tmp_path / "budget.sqlite3"

    async def scenario():
        entered = asyncio.Event()

        async def wait_forever(request):
            entered.set()
            await asyncio.Event().wait()

        transport = BudgetedModelTransport(
            budget(database, limit=1), inner=httpx.MockTransport(wait_forever)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            request = asyncio.create_task(
                client.post("https://synthetic.test", json={"max_tokens": 16})
            )
            await asyncio.wait_for(entered.wait(), 2)
            assert len(reservations(database)) == 1
            request.cancel()
            with suppress(asyncio.CancelledError):
                await request
        with pytest.raises(ModelRequestBudgetError, match="exhausted"):
            budget(database, limit=1).reserve(b'{"max_tokens":16}')

    asyncio.run(scenario())
    assert len(reservations(database)) == 1


@pytest.mark.parametrize("violation", ["output", "body", "remote-embedding"])
def test_payload_budget_denial_is_terminal_before_any_physical_send(
    tmp_path, violation, monkeypatch
):
    database = tmp_path / "budget.sqlite3"
    wire = Wire()
    monkeypatch.setattr("hy_memory.core.embed_service._truncate_for_embedding", lambda text: text)

    async def scenario():
        permission = lease()
        provider, embedder = native(permission)
        limits = budget(database, body_limit=100 if violation == "body" else 4096)
        with pytest.raises(HyPermissionTransportError, match="budget_exceeded"):
            async with HyPermissionTransport(
                permission, http_transport=BudgetedModelTransport(limits, inner=wire)
            ) as transport:
                transport.bind_provider(provider)
                transport.bind_embed_service(embedder)
                with pytest.raises(HyPermissionTransportError, match="budget_exceeded"):
                    if violation == "remote-embedding":
                        await embedder.embed("synthetic embedding", use_cache=False)
                    else:
                        await provider.complete(
                            "synthetic body" * 20,
                            max_tokens=65 if violation == "output" else 16,
                        )
                with pytest.raises(HyPermissionTransportError, match="budget_exceeded"):
                    await provider.complete("smaller", max_tokens=1)
        assert wire.requests == [] and reservations(database) == []

    asyncio.run(scenario())


def test_budget_store_failure_cannot_send_without_a_reservation(tmp_path, monkeypatch):
    database = tmp_path / "budget.sqlite3"
    limits = budget(database)
    wire = Wire()

    def fail_to_open(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic inaccessible budget store")

    async def scenario():
        permission = lease()
        provider, _ = native(permission)
        with pytest.raises(LLMRetryError, match="hy_memory_model_request_failed"):
            async with HyPermissionTransport(
                permission, http_transport=BudgetedModelTransport(limits, inner=wire)
            ) as transport:
                transport.bind_provider(provider)
                with monkeypatch.context() as scoped:
                    # Pooled connections would outlive the patch; start cold.
                    close_main_databases()
                    scoped.setattr(sqlite3, "connect", fail_to_open)
                    await provider.complete("must not send", max_tokens=16)
        assert wire.requests == []

    asyncio.run(scenario())
    assert reservations(database) == []


def test_lease_rejection_precedes_budget_reservation(tmp_path):
    database = tmp_path / "budget.sqlite3"
    limits = budget(database)
    wire = Wire()

    async def scenario():
        permission = lease()
        provider, _ = native(permission)
        with pytest.raises(HyPermissionTransportError, match="lease_denied"):
            async with HyPermissionTransport(
                permission, http_transport=BudgetedModelTransport(limits, inner=wire)
            ) as transport:
                transport.bind_provider(provider)
                permission.revoked = True
                await provider.complete("revoked request", max_tokens=16)
        assert wire.requests == []

    asyncio.run(scenario())
    assert reservations(database) == []


@pytest.mark.parametrize("limit", [None, 2])
def test_actual_core_startup_creates_budget_only_when_opted_in(tmp_path, monkeypatch, limit):
    from one_memory.api import create_app
    from one_memory.secrets import InMemorySecretStore

    monkeypatch.setattr("one_memory.model_presets._preset_data", lambda: None)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        source_home=tmp_path / "home",
        automatic_memory_processing=False,
        imports_enabled=False,
        live_sync_enabled=False,
        live_sync_watch_enabled=False,
        model_request_limit=limit,
    )
    app = create_app(settings, source_catalog={}, secret_store=InMemorySecretStore())

    async def scenario():
        async with app.router.lifespan_context(app):
            assert (app.state.model_request_budget is None) is (limit is None)
            with sqlite3.connect(settings.data_dir / "one-memory.sqlite3") as db:
                table = db.execute(
                    "SELECT name FROM sqlite_master WHERE name='model_request_budget_events'"
                ).fetchone()
                assert (table is None) is (limit is None)
                assert db.execute("SELECT COUNT(*) FROM model_profiles").fetchone()[0] == 0
                assert db.execute("SELECT COUNT(*) FROM memory_tasks").fetchone()[0] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("stop_mode", ["cancel", "cancel-shutdown"])
def test_offthread_dispatch_drains_real_positive_check_without_late_queue(
    tmp_path, monkeypatch, stop_mode
):
    h = Harness(tmp_path)
    h.grant()
    supervisor = h.supervisor()
    prepared = supervisor.prepare_conversation(h.record)
    assert len(prepared) == 1
    real_guard = supervisor._guard_task
    entered, release, exited = (threading.Event() for _ in range(3))
    checks = []

    async def scenario():
        loop_thread = threading.get_ident()

        def checked(task):
            assert threading.get_ident() != loop_thread
            allowed = real_guard(task)
            checks.append(allowed)
            assert allowed
            entered.set()
            assert release.wait(4)
            exited.set()
            return allowed

        monkeypatch.setattr(supervisor, "_guard_task", checked)
        pending = asyncio.create_task(supervisor.resume_off_thread())
        stopping = None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            stopping = (
                asyncio.create_task(supervisor.shutdown())
                if stop_mode == "cancel-shutdown"
                else pending
            )
            await asyncio.sleep(0.02)
            stopping.cancel()
            await asyncio.sleep(0.02)
            stopping.cancel()
            await asyncio.sleep(0.02)
            assert not pending.done() and not stopping.done()
            assert len(supervisor._dispatch_checks) == 1
        finally:
            release.set()
            await asyncio.gather(*({pending, stopping} - {None}), return_exceptions=True)
            await supervisor.shutdown()
        assert checks == [True] and exited.is_set()
        assert supervisor._dispatch_checks == set()
        assert len(supervisor._queue) == 0 and supervisor._workers == {}
        assert h.factory_calls == [] and h.bridge.calls == []
        assert h.tasks.list()[0].attempts == 0

    asyncio.run(scenario())
