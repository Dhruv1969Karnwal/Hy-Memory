import asyncio
import sqlite3

from one_memory.memory_server.migration_runner import SqliteMigrationStore, migrate
from one_memory.memory_server.migrations import MIGRATIONS


def test_fresh_schema_omits_removed_permission_state_machines(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    asyncio.run(migrate(SqliteMigrationStore(path), MIGRATIONS))

    removed = {
        "continuous_policy_history",
        "processing_requests",
        "model_content_scope_seals",
        "model_content_scope_members",
        "memory_task_model_permissions",
        "memory_task_continuous_permissions",
        "model_content_permissions",
        "model_content_evidence_additions",
        "model_content_previews",
        "model_content_authority_state",
    }
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert removed.isdisjoint(tables)
    assert "memory_tasks" in tables
