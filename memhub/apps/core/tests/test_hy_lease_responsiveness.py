"""Blocked lease checks, local failing doubles only; no network or credentials."""

import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from one_memory.hy_memory_bridge import HyMemoryBridge, HyMemoryBridgeError
from one_memory.hy_permission_context import _LeasedBridge, permission_bridge_factory
from one_memory.hy_permission_transport import HyPermissionTransportError


class ControlledLease:
    def __init__(self, *, blocked_call=1, error=None):
        self.receipt = SimpleNamespace(user_id="local")
        self.profile = object()
        self.task_id = "synthetic-task"
        self.records = (SimpleNamespace(evidence_id="synthetic-evidence"),)
        self.blocked_call = blocked_call
        self.error = error
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self.revoked = False

    def recheck(self):
        self.calls.append(threading.get_ident())
        if len(self.calls) == self.blocked_call:
            self.entered.set()
            try:
                if not self.release.wait(2):
                    raise HyMemoryBridgeError("test_check_was_not_released")
                if self.revoked:
                    raise HyMemoryBridgeError("model_content_permission_revoked")
                if self.error is not None:
                    raise self.error
            finally:
                self.exited.set()
        return self.receipt


class ControlledLoop:
    def __init__(self):
        self.calls = []

    def run(self, coroutine):
        self.calls.append(threading.get_ident())
        return asyncio.run(coroutine)


class ControlledWiring:
    def __init__(self, *, error=None):
        self.client = SimpleNamespace(_loop_thread=ControlledLoop())
        self.error = error
        self.finished = 0
        self.closed = 0

    async def finish(self):
        self.finished += 1
        if self.error is not None:
            raise self.error
        return {}

    async def exit(self):
        self.closed += 1


REQUEST = SimpleNamespace(
    user_id="local",
    evidence_ids=("synthetic-evidence",),
    agent_id="agent",
    session_id="session",
    request_id="synthetic-request",
)


@asynccontextmanager
async def blocked_operation(lease, operation):
    main_thread = threading.get_ident()
    pending = asyncio.create_task(operation())
    try:
        # A timer on this loop must run while the synchronous check is held.
        async with asyncio.timeout(0.5):
            while not lease.entered.is_set():
                await asyncio.sleep(0.005)
            ticked = asyncio.Event()
            asyncio.get_running_loop().call_later(0.02, ticked.set)
            await ticked.wait()
        assert not lease.exited.is_set(), "lease check blocked the main event loop"
        assert main_thread not in lease.calls, "lease check executed on the main thread"
        yield pending
    finally:
        lease.release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        assert lease.exited.is_set()


@pytest.mark.parametrize("operation", ["ingest", "ingest_extracted"])
def test_cancelled_start_joins_check_without_later_sdk_entry(operation):
    async def scenario():
        lease = ControlledLease()
        bridge = SimpleNamespace(
            _get_client=AsyncMock(), ingest=AsyncMock(), ingest_extracted=AsyncMock()
        )
        bound = _LeasedBridge(bridge, lease, object(), None)

        async def start():
            try:
                if operation == "ingest":
                    await bound.ingest(REQUEST)
                else:
                    await bound.ingest_extracted(REQUEST, ("Synthetic original",))
            finally:
                await bound.close()

        async with blocked_operation(lease, start) as pending:
            pending.cancel()
            await asyncio.sleep(0.02)
            pending.cancel()
            await asyncio.sleep(0.02)
            assert not pending.done(), "cancel abandoned the active lease check"
            assert not bridge._get_client.called
            lease.release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert not bridge._get_client.called
        assert not bridge.ingest.called and not bridge.ingest_extracted.called
        assert bound.wiring is None

    asyncio.run(scenario())


def test_revoked_start_preserves_error_and_never_opens_sdk():
    async def scenario():
        lease = ControlledLease()
        bridge = SimpleNamespace(_get_client=AsyncMock())
        bound = _LeasedBridge(bridge, lease, object(), None)
        async with blocked_operation(lease, lambda: bound._start(REQUEST)) as pending:
            lease.revoked = True
            lease.release.set()
            with pytest.raises(HyMemoryBridgeError, match="^model_content_permission_revoked$"):
                await pending
        await bound.close()
        assert not bridge._get_client.called and bound.wiring is None

    asyncio.run(scenario())


def test_released_start_reaches_client_once_and_preserves_client_error():
    async def scenario():
        lease = ControlledLease()
        expected = HyMemoryBridgeError("hy_memory_init_failed")
        bridge = SimpleNamespace(_get_client=AsyncMock(side_effect=expected))
        bound = _LeasedBridge(bridge, lease, object(), None)
        async with blocked_operation(lease, lambda: bound._start(REQUEST)) as pending:
            assert not bridge._get_client.called
            lease.release.set()
            with pytest.raises(HyMemoryBridgeError) as raised:
                await pending
            assert raised.value is expected
        bridge._get_client.assert_awaited_once_with()
        await bound.close()
        assert len(lease.calls) == 1 and bound.wiring is None

    asyncio.run(scenario())


@pytest.mark.parametrize("transport_failure", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_postcheck_and_transport_error_recheck_are_joined(transport_failure, cancel):
    async def scenario():
        expected = HyMemoryBridgeError("model_content_evidence_changed")
        lease = ControlledLease(blocked_call=2 if transport_failure else 1, error=expected)
        bound = _LeasedBridge(object(), lease, object(), None)
        wiring = ControlledWiring(
            error=HyPermissionTransportError("controlled_transport_failure")
            if transport_failure
            else None
        )
        bound.wiring = wiring

        async def checked():
            try:
                await bound._check()
            finally:
                await bound.close()

        async with blocked_operation(lease, checked) as pending:
            if cancel:
                pending.cancel()
                await asyncio.sleep(0.02)
                assert not pending.done() and wiring.closed == 0
            lease.release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                with pytest.raises(HyMemoryBridgeError) as raised:
                    await pending
                assert raised.value is expected
        assert wiring.finished == int(transport_failure)
        assert wiring.closed == 1 and bound.finished

    asyncio.run(scenario())


@pytest.mark.parametrize("revoked", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_factory_error_check_holds_exclusive_access_until_joined(tmp_path, revoked, cancel):
    async def scenario():
        lease = ControlledLease()
        bridge = HyMemoryBridge(
            profile=lease.profile,
            api_key=None,
            data_dir=tmp_path,
            client=SimpleNamespace(close=lambda: None),
        )
        factory = permission_bridge_factory(lambda *_: bridge)
        expected = HyPermissionTransportError("controlled_transport_failure")

        async def failed_context():
            async with factory(lease, None, tmp_path):
                raise expected

        async with blocked_operation(lease, failed_context) as pending:
            assert bridge._access_lock.locked()
            lease.revoked = revoked
            if cancel:
                pending.cancel()
                await asyncio.sleep(0.02)
                assert not pending.done() and bridge._access_lock.locked()
            lease.release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                code = "model_content_permission_revoked" if revoked else expected.code
                with pytest.raises(HyMemoryBridgeError, match=f"^{code}$"):
                    await pending
        assert bridge._access_owner is None and not bridge._access_lock.locked()
        await bridge.close()
        assert bridge._client is None

    asyncio.run(scenario())
