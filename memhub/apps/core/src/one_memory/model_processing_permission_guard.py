"""Live global permission checks for remote model-content egress.

``config.json`` is the only durable permission state.  Tasks do not carry a
permission receipt, revision, source scope, or model pin.  Every remote send
and every publication boundary comes back through the lease and reads the
current final-state boolean.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from one_memory.hy_source_context import HySourceContext
from one_memory.imports.models import EvidenceRecord
from one_memory.memory_governance import MemoryGovernanceStore, MemoryStatus
from one_memory.memory_server.config_state import ConfigStore
from one_memory.memory_tasks import MemoryTask, MemoryTaskStatus
from one_memory.model_profiles import ModelProfileSnapshot, ModelProfileStore


def permission_error(code: str) -> RuntimeError:
    from one_memory.hy_memory_bridge import HyMemoryBridgeError

    return HyMemoryBridgeError(
        code if code.startswith("model_content_") else f"model_content_{code}"
    )


def error_is_policy_stop(code: str) -> bool:
    """A disabled final-state permission is an intentional stop."""

    return code in {
        "model_content_source_disabled",
        "hy_memory_evidence_changed",
    }


@dataclass(frozen=True)
class ModelProcessingPermissionLease:
    task_id: str
    user_id: str
    profile: ModelProfileSnapshot
    records: tuple[EvidenceRecord, ...]
    source_context: HySourceContext | None
    _guard: ModelProcessingPermissionGuard = field(repr=False, compare=False)

    def recheck(self) -> None:
        """Read current config/profile immediately before an egress boundary."""

        self.recheck_permission()
        self._guard.check_context(self.records, self.source_context)

    def recheck_permission(self) -> None:
        self._guard.check(self.profile, self.records)


class ModelProcessingPermissionGuard:
    def __init__(
        self,
        config: ConfigStore,
        governance: MemoryGovernanceStore,
        profiles: ModelProfileStore,
    ) -> None:
        self.config = config
        self.governance = governance
        self.profiles = profiles

    def require_allowed(self) -> None:
        if not self.config.model_content_allowed():
            raise permission_error("permission_required")

    def check(
        self,
        profile: ModelProfileSnapshot,
        records: tuple[EvidenceRecord, ...] = (),
    ) -> None:
        """Validate only live global consent and the currently active model."""

        self.require_allowed()
        if self.profiles.active_identity() != (profile.profile_id, profile.revision):
            raise permission_error("profile_changed")
        if records and any(record.user_id != records[0].user_id for record in records):
            raise permission_error("context_scope_invalid")

    def lease(
        self,
        task: MemoryTask,
        records: tuple[EvidenceRecord, ...],
        context: HySourceContext | None,
    ) -> ModelProcessingPermissionLease:
        combined = self.context_records(records, context)
        lease = ModelProcessingPermissionLease(
            task_id=task.task_id,
            user_id=task.user_id,
            profile=task.profile_snapshot,
            records=combined,
            source_context=context,
            _guard=self,
        )
        lease.recheck()
        return lease

    @staticmethod
    def context_records(
        records: tuple[EvidenceRecord, ...],
        context: HySourceContext | None,
    ) -> tuple[EvidenceRecord, ...]:
        by_id = {record.evidence_id: record for record in records}
        if len(by_id) != len(records):
            raise permission_error("context_scope_invalid")
        if context is not None:
            for record in (*context.current, *context.history, *context.supporting):
                previous = by_id.get(record.evidence_id)
                if previous is not None and previous != record:
                    raise permission_error("evidence_changed")
                by_id[record.evidence_id] = record
        return tuple(by_id.values())

    def check_context(
        self,
        records: tuple[EvidenceRecord, ...],
        context: HySourceContext | None,
    ) -> None:
        if context is None:
            return
        ids = {record.evidence_id for record in records}
        if not context.current:
            raise permission_error("context_scope_invalid")
        context_records = (*context.current, *context.history, *context.supporting)
        if len({record.evidence_id for record in context_records}) != len(context_records):
            raise permission_error("context_scope_invalid")
        for known in context.known_memories:
            if not known.evidence_ids or not set(known.evidence_ids) <= ids:
                raise permission_error("context_support_missing")
            memory = self.governance.get(
                known.governed_memory_id,
                user_id=context.current[0].user_id,
            )
            if (
                memory is None
                or memory.revision != known.revision
                or memory.content != known.content
                or set(memory.evidence_ids) != set(known.evidence_ids)
                or memory.status is not MemoryStatus.CONFIRMED
            ):
                raise permission_error("context_changed")

    def reject_unstarted(self, task_id: str, code: str) -> None:
        stopped = error_is_policy_stop(code)
        with self._task_write() as tasks:
            tasks.execute(
                "UPDATE memory_tasks SET status=?, last_error_code=?, updated_at=? "
                "WHERE task_id=? AND status IN (?,?,?,?)",
                (
                    MemoryTaskStatus.CANCELLED if stopped else MemoryTaskStatus.FAILED,
                    str(permission_error(code)),
                    datetime.now(UTC).isoformat(),
                    task_id,
                    MemoryTaskStatus.PENDING,
                    MemoryTaskStatus.WAITING_FOR_PERMISSION,
                    MemoryTaskStatus.WAITING_FOR_MODEL,
                    MemoryTaskStatus.DEFERRED_OUTSIDE_WINDOW,
                ),
            )

    @contextmanager
    def _task_write(self) -> Iterator[object]:
        from one_memory.database import take_the_write_lock_first
        from one_memory.database_access import main_database

        with main_database(self.governance.database_path).write(
            autocommit=True,
            foreign_keys=True,
        ) as connection:
            take_the_write_lock_first(connection)
            with connection:
                yield connection
