"""Task-exclusive HY wiring; provenance metadata stays beside the existing Core DB."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any, cast

import httpx

from one_memory.database import take_the_write_lock_first
from one_memory.database_access import main_database
from one_memory.hy_memory_bridge import (
    HyMemoryBridge,
    HyMemoryBridgeError,
    HyMemoryIngestRequest,
    HyMemoryIngestResult,
    HyMemoryOutput,
    LocalHyEmbedService,
    _joined_thread,
)
from one_memory.hy_permission_coding_store import HyPermissionCodingStore
from one_memory.hy_permission_store import (
    HyNodePermission,
    HyPermissionGraphStore,
    HyPermissionVectorStore,
)
from one_memory.hy_permission_transport import HyPermissionTransport, HyPermissionTransportError
from one_memory.hy_response_guard import HyResponseGuard
from one_memory.hy_step_timing import HyStepTimer
from one_memory.memory_server.migrations import hy_node_permissions_ddl
from one_memory.memory_tasks import evidence_ref
from one_memory.model_processing_permission_guard import ModelProcessingPermissionLease


class _NativeProofs:
    def __init__(self, database: Path) -> None:
        self.database = database
        self._loaded_states: dict[str, tuple[int, str]] | None = None
        with main_database(database).write() as db, db:
            # Schema owned by `memory_server.migrations` (migration 26).
            db.execute(hy_node_permissions_ddl("sqlite"))

    def load(self, lease: ModelProcessingPermissionLease) -> dict[str, HyNodePermission]:
        allowed = {record.evidence_id: evidence_ref(record) for record in lease.records}
        with main_database(self.database).read() as db:
            rows = db.execute(
                "SELECT p.node_id,p.fingerprint,p.evidence_json,g.revision,g.status "
                "FROM hy_node_permissions p LEFT JOIN governed_memories g ON g.memory_id=p.node_id "
                # Superseded nodes stay attested: every later head links back to them.
                "WHERE g.status IS NULL OR g.status IN ('confirmed','superseded')"
            ).fetchall()
        proofs = {}
        states = {}
        for identifier, fingerprint, evidence_json, revision, status in rows:
            refs = json.loads(evidence_json)
            if (
                not isinstance(refs, list)
                or not refs
                or not all(isinstance(ref, dict) for ref in refs)
            ):
                continue
            if all(allowed.get(ref.get("evidence_id")) == ref for ref in refs):
                proofs[identifier] = HyNodePermission(
                    fingerprint, tuple(ref["evidence_id"] for ref in refs)
                )
                if revision is not None:
                    states[identifier] = (revision, status)
        self._loaded_states = states
        return proofs

    def guard(self, values: dict[str, HyNodePermission]) -> Callable[[], None]:
        identifiers = tuple(values)

        def state() -> dict[str, tuple[int, str]]:
            found: dict[str, tuple[int, str]] = {}
            with main_database(self.database).read() as db:
                for start in range(0, len(identifiers), 900):
                    batch = identifiers[start : start + 900]
                    placeholders = ",".join("?" for _ in batch)
                    for identifier, revision, status in db.execute(
                        "SELECT memory_id,revision,status FROM governed_memories "
                        f"WHERE memory_id IN ({placeholders})",
                        batch,
                    ):
                        found[identifier] = (revision, status)
            return found

        if self._loaded_states is None:
            raise HyMemoryBridgeError("model_content_context_unverified")
        before = dict(self._loaded_states)

        def check() -> None:
            if state() != before:
                raise HyMemoryBridgeError("model_content_context_changed")

        check()
        return check

    def save(
        self, lease: ModelProcessingPermissionLease, values: dict[str, HyNodePermission]
    ) -> None:
        refs = {record.evidence_id: evidence_ref(record) for record in lease.records}
        with main_database(self.database).write(autocommit=True) as db:
            take_the_write_lock_first(db)
            with db:
                lease.recheck()
                db.executemany(
                    "INSERT INTO hy_node_permissions VALUES (?,?,?) ON CONFLICT(node_id) "
                    "DO UPDATE SET fingerprint=excluded.fingerprint,"
                    "evidence_json=excluded.evidence_json",
                    [
                        (
                            identifier,
                            proof.fingerprint,
                            json.dumps([refs[key] for key in proof.evidence_ids]),
                        )
                        for identifier, proof in values.items()
                    ],
                )

    def refresh(
        self, lease: ModelProcessingPermissionLease, node_id: str
    ) -> HyNodePermission | None:
        del lease
        with main_database(self.database).read() as db:
            row = db.execute(
                "SELECT fingerprint,evidence_json FROM hy_node_permissions WHERE node_id=?",
                (node_id,),
            ).fetchone()
        if row is None:
            return None
        fingerprint = str(row["fingerprint"])
        refs = json.loads(row["evidence_json"])
        if (
            len(fingerprint) != 64
            or any(value not in "0123456789abcdef" for value in fingerprint)
            or not isinstance(refs, list)
            or not refs
            or not all(isinstance(ref, dict) for ref in refs)
        ):
            return None
        evidence_ids = tuple(ref.get("evidence_id") for ref in refs)
        if any(not isinstance(value, str) or not value for value in evidence_ids):
            return None
        return HyNodePermission(fingerprint, evidence_ids)


class _BoundLease:
    def __init__(
        self,
        lease: ModelProcessingPermissionLease,
        proof_check: Callable[[], None],
        step_timer: HyStepTimer | None = None,
    ) -> None:
        self.lease = lease
        self.step_timer = step_timer
        self.user_id = lease.user_id
        self.profile = lease.profile
        self.task_id = lease.task_id
        self.records = lease.records
        self.source_context = lease.source_context
        self.proof_check = proof_check
        self.vector: HyPermissionVectorStore | None = None
        self.coding: HyPermissionCodingStore | None = None

    def recheck(self) -> None:
        timer = self.step_timer
        with timer.step("permission_check") if timer is not None else nullcontext():
            self.lease.recheck()
            self.proof_check()
            if self.vector is not None:
                self.vector.recheck()
            if self.coding is not None:
                self.coding.recheck()


class _Wiring:
    def __init__(
        self,
        client: Any,
        lease: ModelProcessingPermissionLease,
        proofs_store: _NativeProofs,
        proofs: dict[str, HyNodePermission],
        agent_id: str,
        session_id: str,
        transport: httpx.AsyncBaseTransport | None,
        proof_check: Callable[[], None],
        step_timer: HyStepTimer,
    ) -> None:
        self.client, self.lease, self.proofs_store, self.proofs = (
            client,
            lease,
            proofs_store,
            proofs,
        )
        self.step_timer = step_timer
        self.agent_id, self.session_id, self.http_transport = agent_id, session_id, transport
        self.cross_evidence_scope = False
        self.proof_check = proof_check
        self.saved: list[tuple[Any, str, Any]] = []
        self.transport: HyPermissionTransport
        self.vector: HyPermissionVectorStore
        self.graph: HyPermissionGraphStore | None = None
        self.coding: HyPermissionCodingStore
        self.active = False

    def replace(self, target: Any, name: str, value: Any) -> None:
        self.saved.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    async def enter(self) -> None:
        client, registry = self.client, self.client._registry
        config = client._config
        if config.llm.enable_summary or config.llm.enable_query_rewrite:
            raise HyMemoryBridgeError("model_content_context_unverified")
        bound = _BoundLease(self.lease, self.proof_check, self.step_timer)
        self.transport = await HyPermissionTransport(
            bound, http_transport=self.http_transport, step_timer=self.step_timer
        ).__aenter__()
        self.active = True
        try:
            for guard in getattr(client, "_one_memory_write_guards", ()):
                if hasattr(type(guard), "step_timer"):
                    self.replace(guard, "step_timer", self.step_timer)
            system2 = await registry.get_system2_writer()
            writers = (await registry.get_writer(), system2._writer)
            native_vector = client._vector_store
            store_lease = _BoundLease(self.lease, self.proof_check, self.step_timer)
            self.vector = HyPermissionVectorStore(
                native_vector,
                cast(ModelProcessingPermissionLease, store_lease),
                agent_id=self.agent_id,
                session_id=self.session_id,
                cross_evidence_scope=self.cross_evidence_scope,
                node_permissions=self.proofs,
                refresh_permission=lambda node_id: asyncio.to_thread(
                    self.proofs_store.refresh, self.lease, node_id
                ),
                step_timer=self.step_timer,
            )
            bound.vector = self.vector
            embed = client._embed_service
            if isinstance(embed, LocalHyEmbedService):
                self.transport.bind_local_embed_service(embed)
            else:
                self.transport.bind_embed_service(embed)
            for writer in writers:
                provider = writer.mem_agent.llm_provider
                self.transport.bind_provider(
                    provider.provider if isinstance(provider, HyResponseGuard) else provider
                )
                reconciler = writer.reconciler
                provider = reconciler._get_llm()
                self.transport.bind_provider(
                    provider.provider if isinstance(provider, HyResponseGuard) else provider
                )
                self.replace(writer, "_external_vector_store", self.vector)
                self.replace(writer, "_vector_store", self.vector)
                self.replace(writer, "_vector_store_initialized", True)
                self.replace(writer, "_cache", None)
            self.replace(client, "_vector_store", self.vector)
            native_graph = client._graph_store
            if self.cross_evidence_scope and native_graph is not None:
                self.graph = HyPermissionGraphStore(native_graph, self.proof_check)
                self.replace(client, "_graph_store", self.graph)
            else:
                self.replace(client, "_graph_store", None)
            self.replace(registry, "_shared_vector_store", self.vector)
            self.replace(
                registry,
                "_shared_graph_store",
                self.graph if self.graph is not None else None,
            )
            self.replace(registry, "_shared_cache", None)
            self.replace(system2, "_vector_store", self.vector)
            self.replace(
                system2,
                "_graph_store",
                self.graph if self.graph is not None else None,
            )
            self.replace(system2, "_cache", None)
            # Result fallback may call search; rebuild only its non-generative reader.
            self.replace(registry, "_reader", None)
            self.replace(registry, "_readers", {})
            await self._enter_coding(bound, embed)
            self.transport.assert_allowed()
        except BaseException:
            await self.exit()
            raise

    async def _enter_coding(self, bound: _BoundLease, embed: Any) -> None:
        """HY judges and writes the coding route itself; every seam it uses is bound here."""
        client = self.client
        # Created now, so the judge and every coding stage share one bound provider.
        writer = await client._get_coding_writer()
        stages = (writer, writer.extractor, writer.reconciler, writer.fact_reconciler)
        native_store = client._coding_store
        if any(
            stage.store is not native_store
            for stage in (writer, writer.reconciler, writer.fact_reconciler)
        ) or any(
            item.embed_service is not embed
            for item in (native_store, writer, writer.reconciler, writer.fact_reconciler)
        ):
            raise HyMemoryBridgeError("model_content_context_unverified")
        providers = {
            id(p): p for p in (client._coding_llm_provider, *(s.llm_provider for s in stages))
        }
        for provider in providers.values():
            self.transport.bind_provider(provider)
        self.coding = HyPermissionCodingStore(
            native_store,
            cast(
                ModelProcessingPermissionLease,
                _BoundLease(self.lease, self.proof_check, self.step_timer),
            ),
            agent_id=self.agent_id,
            session_id=self.session_id,
            node_permissions=self.proofs,
        )
        bound.coding = self.coding
        for stage in (writer, writer.reconciler, writer.fact_reconciler):
            self.replace(stage, "store", self.coding)
        self.replace(client, "_coding_store", self.coding)
        for stage in (writer, writer.extractor, writer.reconciler):
            self.replace(stage, "cache", None)
        # The judge logs its prompt through the client's own cache.
        self.replace(client, "_cache", None)

    async def finish(self) -> dict[str, HyNodePermission]:
        self.transport.assert_allowed()
        self.vector.recheck()
        self.coding.recheck()
        return {**self.vector.written_nodes, **self.coding.written_nodes}

    async def retired(self) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        self.transport.assert_allowed()
        return (
            tuple(sorted(self.vector.retired_node_ids | self.coding.retired_ids)),
            {**self.vector.superseded_nodes, **self.coding.superseded_ids},
        )

    async def exit(self) -> None:
        if not self.active:
            return
        self.active = False
        try:
            await self.transport.__aexit__(None, None, None)
        finally:
            for target, name, value in reversed(self.saved):
                setattr(target, name, value)
            self.saved.clear()
            if hasattr(self, "vector"):
                await self.vector.close()
            if self.graph is not None:
                await self.graph.close()
            if hasattr(self, "coding"):
                await self.coding.close()


class _LeasedBridge:
    def __init__(
        self,
        bridge: HyMemoryBridge,
        lease: ModelProcessingPermissionLease,
        proofs: _NativeProofs,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        self.bridge, self.lease, self.proofs, self.transport = bridge, lease, proofs, transport
        self.wiring: _Wiring | None = None
        self.finished = False
        self.step_timer = HyStepTimer()
        self.request_id: str | None = None

    def _step(self, name: str) -> Any:
        return self.step_timer.step(name)

    async def _recheck(self) -> None:
        def recheck() -> None:
            with self._step("permission_check"):
                self.lease.recheck()

        await _joined_thread(recheck)

    async def _start(self, request: HyMemoryIngestRequest) -> None:
        if self.wiring is not None or self.finished:
            raise HyMemoryBridgeError("model_content_context_reused")
        if request.user_id != self.lease.user_id or set(request.evidence_ids) - {
            item.evidence_id for item in self.lease.records
        }:
            raise HyMemoryBridgeError("model_content_context_unverified")
        await self._start_operation(
            user_id=request.user_id,
            agent_id=request.agent_id,
            request_id=request.request_id,
            session_id=request.session_id,
        )

    async def _start_operation(
        self,
        *,
        user_id: str,
        agent_id: str,
        request_id: str,
        session_id: str,
        cross_evidence_scope: bool = False,
    ) -> None:
        if self.wiring is not None or self.finished:
            raise HyMemoryBridgeError("model_content_context_reused")
        if user_id != self.lease.user_id:
            raise HyMemoryBridgeError("model_content_context_unverified")
        self.request_id = request_id
        await self._recheck()
        client = cast(Any, await self.bridge._get_client())
        with self._step("permission_store_read"):
            known = await asyncio.to_thread(self.proofs.load, self.lease)
            proof_check = await asyncio.to_thread(self.proofs.guard, known)
        self.wiring = _Wiring(
            client,
            self.lease,
            self.proofs,
            known,
            agent_id,
            session_id,
            self.transport,
            proof_check,
            self.step_timer,
        )
        self.wiring.cross_evidence_scope = cross_evidence_scope
        await _joined_thread(client._loop_thread.run, self.wiring.enter())

    async def list_agent_ids(self, *, user_id: str, limit: int = 1_000) -> tuple[str, ...]:
        if user_id != self.lease.user_id or self.wiring is not None or self.finished:
            raise HyMemoryBridgeError("model_content_context_unverified")
        await self._recheck()
        result = await self.bridge.list_agent_ids(user_id=user_id, limit=limit)
        await self._recheck()
        return result

    async def digest(self, *, user_id: str, agent_id: str) -> dict[str, object]:
        await self._start_operation(
            user_id=user_id,
            agent_id=agent_id,
            request_id=f"digest:{self.lease.task_id}:{agent_id}",
            session_id=f"digest:{self.lease.task_id}",
            cross_evidence_scope=True,
        )
        try:
            result = dict(await self.bridge.digest(user_id=user_id, agent_id=agent_id))
        except HyMemoryBridgeError:
            await self._check()
            raise
        proofs = await self._check()
        with self._step("permission_store_write"):
            await asyncio.to_thread(self.proofs.save, self.lease, proofs)
        await self.close()
        await self._recheck()
        return result

    async def graph_snapshot(self, *, user_id: str, agent_id: str, limit: int = 300) -> Any:
        await self._start_operation(
            user_id=user_id,
            agent_id=agent_id,
            request_id=f"digest-graph:{self.lease.task_id}:{agent_id}",
            session_id=f"digest:{self.lease.task_id}",
            cross_evidence_scope=True,
        )
        try:
            result = await self.bridge.graph_snapshot(
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
            )
        except HyMemoryBridgeError:
            await self._check()
            raise
        await self._check()
        await self.close()
        await self._recheck()
        return result

    async def ingest(self, request: HyMemoryIngestRequest) -> HyMemoryIngestResult:
        await self._start(request)
        try:
            result = await self.bridge.ingest(request)
        except HyMemoryBridgeError:
            await self._check()
            raise
        return await self._with_retired(result)

    async def ingest_extracted(
        self, request: HyMemoryIngestRequest, contents: tuple[str, ...]
    ) -> HyMemoryIngestResult:
        await self._start(request)
        try:
            result = await self.bridge.ingest_extracted(request, contents)
        except HyMemoryBridgeError:
            await self._check()
            raise
        return await self._with_retired(result)

    async def _with_retired(self, result: HyMemoryIngestResult) -> HyMemoryIngestResult:
        await self._check()
        if self.wiring is None:
            raise HyMemoryBridgeError("model_content_context_unverified")
        try:
            retired, superseded = await _joined_thread(
                self.wiring.client._loop_thread.run, self.wiring.retired()
            )
        except HyPermissionTransportError as error:
            await self._recheck()
            raise HyMemoryBridgeError(error.code) from None
        return result.model_copy(
            update={"retired_memory_ids": retired, "superseded_memory_ids": superseded}
        )

    async def _check(self) -> dict[str, HyNodePermission]:
        if self.wiring is None or self.finished:
            raise HyMemoryBridgeError("model_content_context_unverified")
        await self._recheck()
        try:
            return await _joined_thread(self.wiring.client._loop_thread.run, self.wiring.finish())
        except HyPermissionTransportError as error:
            await self._recheck()
            raise HyMemoryBridgeError(error.code) from None

    async def profile_memories(self, *, user_id: str, agent_id: str) -> tuple[HyMemoryOutput, ...]:
        if self.wiring is None or user_id != self.lease.user_id or agent_id != self.wiring.agent_id:
            raise HyMemoryBridgeError("model_content_context_unverified")
        result = await self.bridge.profile_memories(user_id=user_id, agent_id=agent_id)
        proofs = await self._check()
        with self._step("permission_store_write"):
            await asyncio.to_thread(self.proofs.save, self.lease, proofs)
        # SDK context closes before governance can update a previously known memory.
        await self.close()
        return result

    async def close(self) -> None:
        if self.wiring is not None and not self.finished:
            self.finished = True
            try:
                await _joined_thread(self.wiring.client._loop_thread.run, self.wiring.exit())
            finally:
                self.step_timer.emit(request_id=self.request_id, task_id=self.lease.task_id)


def permission_bridge_factory(
    resolve_bridge: Callable[..., Any],
    *,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> Callable[..., Any]:
    @asynccontextmanager
    async def factory(
        lease: ModelProcessingPermissionLease, secret: str | None, data_dir: Path
    ) -> AsyncIterator[Any]:
        bridge = resolve_bridge(lease.profile, secret, data_dir)
        if not isinstance(bridge, HyMemoryBridge):
            raise HyMemoryBridgeError("model_content_context_unverified")
        proofs = _NativeProofs(data_dir / "one-memory.sqlite3")
        async with bridge.exclusive_access():
            await bridge.prepare()
            bound = _LeasedBridge(bridge, lease, proofs, http_transport)
            try:
                yield bound
            except BaseException as error:
                with suppress(Exception):
                    await bound.close()
                if isinstance(error, HyPermissionTransportError):
                    await _joined_thread(lease.recheck)
                    raise HyMemoryBridgeError(error.code) from None
                raise
            else:
                try:
                    await bound.close()
                except HyPermissionTransportError as error:
                    raise HyMemoryBridgeError(error.code) from None

    return factory
