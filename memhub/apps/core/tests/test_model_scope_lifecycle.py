"""Bound derived/temporary data without changing consent or original history."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest

import one_memory.model_content_permissions as content
from one_memory.model_content_permissions import ModelContentPermissionError
from tests.test_model_content_permissions import USER, evidence_value
from tests.test_model_content_permissions import env as env


def rows(env, table):
    with sqlite3.connect(env.database) as db:
        return db.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()


def tracked_grant(env, revision):
    return env.permissions.grant(
        env.snapshot(), expected_revision=revision, user_confirmed=True, _track_scope=True
    )


def test_regrant_retires_only_old_derived_index_and_keeps_history(env):
    receipts = [tracked_grant(env, revision) for revision in range(3)]
    assert [(r[0], r[1]) for r in rows(env, "model_content_scope_seals")] == [(USER, 3)]
    assert [(r[0], r[1]) for r in rows(env, "model_content_scope_members")] == [(USER, 3)]
    for receipt in receipts:
        assert env.permissions.receipt(user_id=USER, revision=receipt.revision) == receipt
    with pytest.raises(ModelContentPermissionError, match="permission_changed"):
        env.check(receipts[0])
    with patch.object(
        env.permissions, "_evidence_batch", wraps=env.permissions._evidence_batch
    ) as batch:
        assert env.check(receipts[-1]) == receipts[-1]
        assert sum(len(call.args[1]) for call in batch.call_args_list) == 1
    assert len(rows(env, "evidence")) == 1


def test_revoke_retires_derived_index_without_deleting_receipts_or_other_users(env):
    receipt = tracked_grant(env, 0)
    with sqlite3.connect(env.database) as db:
        db.execute(
            "INSERT INTO model_content_scope_seals "
            "SELECT 'other',revision,snapshot_hash,schema_version,valid "
            "FROM model_content_scope_seals"
        )
        db.execute(
            "INSERT INTO model_content_scope_members "
            "SELECT 'other',revision,evidence_id FROM model_content_scope_members"
        )
    before = {table: rows(env, table) for table in ("evidence", "source_grants")}
    revoked = env.permissions.revoke(user_id=USER, expected_revision=1, user_confirmed=True)
    assert [r[0] for r in rows(env, "model_content_scope_seals")] == ["other"]
    assert [r[0] for r in rows(env, "model_content_scope_members")] == ["other"]
    assert env.permissions.receipt(user_id=USER, revision=1) == receipt
    assert env.permissions.receipt(user_id=USER) == revoked
    assert before == {table: rows(env, table) for table in before}
    with pytest.raises(ModelContentPermissionError, match="permission_revoked"):
        env.check(receipt)


def test_failed_replacement_rolls_back_index_cleanup_and_receipt(env, monkeypatch):
    receipt = tracked_grant(env, 0)
    tables = (
        "model_content_permissions",
        "model_content_scope_seals",
        "model_content_scope_members",
    )
    before = {table: rows(env, table) for table in tables}

    def fail_record(*_args, **_kwargs):
        raise sqlite3.OperationalError("controlled write failure")

    monkeypatch.setattr(content.model_content_seals, "record", fail_record)
    with pytest.raises(ModelContentPermissionError, match="permission_store_unavailable"):
        tracked_grant(env, 1)
    assert before == {table: rows(env, table) for table in tables}
    assert env.check(receipt) == receipt


def test_unsealed_replacement_does_not_inherit_prior_seal(env):
    tracked_grant(env, 0)
    receipt = env.permissions.grant(env.snapshot(), expected_revision=1, user_confirmed=True)
    with sqlite3.connect(env.database) as db:
        db.execute("DELETE FROM model_content_scope_seals")
        db.execute("DELETE FROM model_content_scope_members")
    assert rows(env, "model_content_scope_seals") == []
    assert rows(env, "model_content_scope_members") == []
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE evidence SET verbatim='changed without revision'")
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)


def test_preview_byte_budget_evicts_earliest_without_changing_survivor(env, monkeypatch):
    snapshot = env.snapshot(future=True)
    size = len(snapshot.model_dump_json().encode())
    monkeypatch.setattr(content, "_RETAINED_PREVIEW_BYTES", size * 2, raising=False)
    previews = [env.permissions.retain_preview(snapshot) for _ in range(3)]
    saved = rows(env, "model_content_previews")
    assert len(saved) == 2
    assert {r[0] for r in saved} == {p["preview_id"] for p in previews[1:]}
    assert sum(len(r[4].encode()) for r in saved) <= size * 2
    with pytest.raises(ModelContentPermissionError, match="preview_expired"):
        env.permissions.grant_preview(
            previews[0]["preview_id"], user_id=USER, expected_revision=0, user_confirmed=True
        )
    receipt = env.permissions.grant_preview(
        previews[-1]["preview_id"], user_id=USER, expected_revision=0, user_confirmed=True
    )
    assert receipt.snapshot == snapshot and env.check(receipt) == receipt


def test_single_oversized_preview_does_not_delete_valid_previous_preview(env, monkeypatch):
    snapshot = env.snapshot()
    first = env.permissions.retain_preview(snapshot)
    before = rows(env, "model_content_previews")
    monkeypatch.setattr(content, "_RETAINED_PREVIEW_BYTES", 1, raising=False)
    with pytest.raises(ModelContentPermissionError, match="content_scope_too_large"):
        env.permissions.retain_preview(snapshot)
    assert rows(env, "model_content_previews") == before
    assert first["preview_id"] == before[0][0]
    assert env.permissions.receipt(user_id=USER) is None


def test_preview_count_limit_and_expiry_keep_no_authority(env):
    snapshot = env.snapshot()
    first = env.permissions.retain_preview(snapshot)
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE model_content_previews SET expires_at='2000-01-01T00:00:00+00:00'")
    previews = [env.permissions.retain_preview(snapshot) for _ in range(17)]
    assert len(rows(env, "model_content_previews")) == 16
    for preview in (first, previews[0]):
        with pytest.raises(ModelContentPermissionError, match="preview_expired"):
            env.permissions.grant_preview(
                preview["preview_id"], user_id=USER, expected_revision=0, user_confirmed=True
            )
    assert env.permissions.receipt(user_id=USER) is None
    assert len(rows(env, "evidence")) == 1


def test_preview_eviction_rolls_back_when_new_insert_fails(env, monkeypatch):
    snapshot = env.snapshot()
    monkeypatch.setattr(
        content, "_RETAINED_PREVIEW_BYTES", len(snapshot.model_dump_json().encode())
    )
    first = env.permissions.retain_preview(snapshot)
    with sqlite3.connect(env.database) as db:
        db.execute(
            "CREATE TRIGGER fail_preview_insert BEFORE INSERT ON model_content_previews "
            "BEGIN SELECT RAISE(ABORT,'controlled preview failure'); END"
        )
    before = rows(env, "model_content_previews")
    with pytest.raises(ModelContentPermissionError, match="permission_store_unavailable"):
        env.permissions.retain_preview(snapshot)
    assert rows(env, "model_content_previews") == before
    receipt = env.permissions.grant_preview(
        first["preview_id"], user_id=USER, expected_revision=0, user_confirmed=True
    )
    assert env.check(receipt) == receipt


def test_concurrent_previews_respect_shared_byte_budget(env, monkeypatch):
    snapshot = env.snapshot(future=True)
    size = len(snapshot.model_dump_json().encode())
    monkeypatch.setattr(content, "_RETAINED_PREVIEW_BYTES", 2 * size)
    ready = Barrier(8)

    def retain(_index):
        ready.wait(timeout=5)
        return env.permissions.retain_preview(snapshot)

    with ThreadPoolExecutor(max_workers=8) as pool:
        returned = list(pool.map(retain, range(8)))
    saved = rows(env, "model_content_previews")
    assert len(saved) == 2
    assert sum(len(row[4].encode()) for row in saved) == 2 * size
    assert {row[0] for row in saved} <= {preview["preview_id"] for preview in returned}
    assert all(row[4] == snapshot.model_dump_json() for row in saved)
    assert env.permissions.receipt(user_id=USER) is None


def test_regrant_cleanup_keeps_current_unrequested_member_invalidation(env):
    other = env.evidence.put(evidence_value("unrequested")).record
    for revision in range(3):
        receipt = env.permissions.grant(
            env.snapshot(records=(env.record, other)),
            expected_revision=revision,
            user_confirmed=True,
            _track_scope=True,
        )
    assert len(rows(env, "model_content_scope_seals")) == 1
    assert len(rows(env, "model_content_scope_members")) == 2
    assert env.check(receipt) == receipt
    with sqlite3.connect(env.database) as db:
        db.execute(
            "UPDATE evidence SET verbatim='changed but same revision/hash' WHERE evidence_id=?",
            (other.evidence_id,),
        )
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)
