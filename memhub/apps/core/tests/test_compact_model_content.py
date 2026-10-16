"""Server-held previews preserve exact grants without round-tripping all refs."""

import json
import sqlite3

import pytest

from one_memory.model_content_permissions import (
    ModelContentPermissionError,
    ModelContentPermissionStore,
)
from tests.test_model_content_bulk import many_records
from tests.test_model_content_permissions import USER, evidence_value
from tests.test_model_content_permissions import env as env


def test_compact_preview_grants_exact_scope_and_survives_restart(env):
    records = many_records(env, 1000)
    snapshot = env.snapshot(records=tuple(records), future=True)
    result = env.permissions.retain_preview(snapshot)
    assert result["evidence"] == [] and result["evidence_count"] == 1000
    assert len(json.dumps(result)) < 5000
    assert env.permissions.receipt(user_id=USER) is None
    restarted = ModelContentPermissionStore(env.database)
    receipt = restarted.grant_preview(
        result["preview_id"], user_id=USER, expected_revision=0, user_confirmed=True
    )
    assert receipt.snapshot == snapshot
    assert env.check(receipt) == receipt
    with pytest.raises(ModelContentPermissionError, match="permission_changed"):
        restarted.grant_preview(
            result["preview_id"], user_id=USER, expected_revision=0, user_confirmed=True
        )


@pytest.mark.parametrize(
    "change", ["edit", "revoke-source", "grant-race", "expiry", "wrong-owner", "confirmation"]
)
def test_compact_preview_does_not_bypass_live_checks(env, change):
    result = env.permissions.retain_preview(env.snapshot(future=True))
    if change == "edit":
        with sqlite3.connect(env.database) as db:
            db.execute("UPDATE evidence SET verbatim='changed raw original'")
    elif change == "revoke-source":
        env.sources.revoke(user_id=USER, source_id=env.record.source_id)
    elif change == "grant-race":
        env.grant()
    elif change == "expiry":
        with sqlite3.connect(env.database) as db:
            db.execute("UPDATE model_content_previews SET expires_at='2000-01-01T00:00:00+00:00'")
    with pytest.raises(ModelContentPermissionError):
        env.permissions.grant_preview(
            result["preview_id"],
            user_id="other" if change == "wrong-owner" else USER,
            expected_revision=0,
            user_confirmed=change != "confirmation",
        )


def test_new_originals_keep_explicit_future_scope_but_not_exact_scope(env):
    result = env.permissions.retain_preview(env.snapshot(future=True))
    later = env.evidence.put(evidence_value("new-arrival")).record
    receipt = env.permissions.grant_preview(
        result["preview_id"], user_id=USER, expected_revision=0, user_confirmed=True
    )
    assert len(receipt.snapshot.evidence) == 1
    assert env.check(receipt, records=(later,)) == receipt


def test_compact_receipt_is_metadata_only_and_version_bound(env):
    result = env.permissions.retain_preview(env.snapshot())
    receipt = env.permissions.grant_preview(
        result["preview_id"], user_id=USER, expected_revision=0, user_confirmed=True
    )
    compact = env.permissions.compact_receipt(receipt)
    assert compact["revision"] == receipt.revision
    assert compact["snapshot"]["snapshot_hash"] == result["snapshot_hash"]
    assert compact["snapshot"]["source_counts"] == {env.record.source_id: 1}
    assert env.record.verbatim not in json.dumps(compact)
    assert compact["snapshot"]["evidence"] == []


@pytest.mark.parametrize("future", [False, True])
def test_preview_cannot_rebase_over_a_grant_or_revoke_between_reads(env, future):
    first = env.grant(future=future)
    preview = env.snapshot(future=future)
    env.permissions.revoke(user_id=USER, expected_revision=first.revision, user_confirmed=True)
    with pytest.raises(ModelContentPermissionError, match="permission_changed"):
        env.permissions.retain_preview(preview)
    assert env.permissions.receipt(user_id=USER).revoked_at is not None


def test_shared_migration_declares_the_retained_preview_table() -> None:
    """Retained previews are product state; migration 28 must recreate them.

    This is a red-state guard for the A1-04 schema extraction. The table exists
    in the historical f6d624a schema and is still read/written by this module,
    so migration 28's generator must declare it for both SQLite and MySQL.
    """
    from one_memory.memory_server import migrations

    sqlite_schema = migrations.model_content_permissions_ddl("sqlite")
    mysql_schema = migrations.model_content_permissions_ddl("mysql")
    assert "model_content_previews" in sqlite_schema
    assert "model_content_previews" in mysql_schema
