"""Indexed scope invalidation retains complete-snapshot semantics without rereading it."""

import sqlite3
from unittest.mock import patch

import pytest

from one_memory.model_content_permissions import ModelContentPermissionError
from tests.test_model_content_bulk import many_records
from tests.test_model_content_permissions import env as env
from tests.test_model_content_permissions import evidence_value


def seal(env, count=6):
    records = many_records(env, count)
    receipt = env.permissions.grant(
        env.snapshot(records=tuple(records), future=True),
        expected_revision=0,
        user_confirmed=True,
        _track_scope=True,
    )
    return records, receipt


def test_sealed_scope_only_hydrates_outbound_records(env):
    records, receipt = seal(env, 1000)
    with patch.object(
        env.permissions, "_evidence_batch", wraps=env.permissions._evidence_batch
    ) as batch:
        assert env.check(receipt) == receipt
        assert sum(len(call.args[1]) for call in batch.call_args_list) == 1
    added = env.evidence.put(evidence_value("after-grant")).record
    with patch.object(
        env.permissions, "_evidence_batch", wraps=env.permissions._evidence_batch
    ) as batch:
        assert env.check(receipt, records=(added,)) == receipt
        assert sum(len(call.args[1]) for call in batch.call_args_list) == 1
    assert len(records) == 1000


@pytest.mark.parametrize(
    "mutation", ["body", "role", "owner", "time", "delete", "origin", "origin_delete", "origin_add"]
)
def test_unrequested_member_changes_invalidate_even_without_revision(env, mutation):
    records, receipt = seal(env)
    target = records[-1]
    assert env.check(receipt) == receipt
    with sqlite3.connect(env.database) as db:
        if mutation == "origin_add":
            db.execute(
                "INSERT INTO evidence_origins"
                "(evidence_id,origin,raw_meta_json,source_version,observed_at) "
                "VALUES (?,'manual_import','{}',NULL,?)",
                (target.evidence_id, target.imported_at.isoformat()),
            )
        else:
            sql = {
                "body": "UPDATE evidence SET verbatim='edited without hash' WHERE evidence_id=?",
                "role": "UPDATE evidence SET role='system' WHERE evidence_id=?",
                "owner": "UPDATE evidence SET user_id='other' WHERE evidence_id=?",
                "time": "UPDATE evidence SET created_at='2026-09-01T00:00:00Z' WHERE evidence_id=?",
                "delete": "DELETE FROM evidence WHERE evidence_id=?",
                "origin": "UPDATE evidence_origins SET raw_meta_json='{\"new\":true}' "
                "WHERE evidence_id=?",
                "origin_delete": "DELETE FROM evidence_origins WHERE evidence_id=?",
            }[mutation]
            db.execute(sql, (target.evidence_id,))
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)


@pytest.mark.parametrize("tracking", ["missing", "altered"])
def test_missing_or_modified_trigger_uses_full_validation(env, tracking):
    records, receipt = seal(env)
    with sqlite3.connect(env.database) as db:
        db.execute("DROP TRIGGER model_content_seal_evidence_update")
        if tracking == "altered":
            db.execute(
                "CREATE TRIGGER model_content_seal_evidence_update "
                "AFTER UPDATE ON evidence BEGIN SELECT 1; END"
            )
        db.execute(
            "UPDATE evidence SET verbatim='untracked edit' WHERE evidence_id=?",
            (records[-1].evidence_id,),
        )
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)


def test_legacy_unsealed_receipt_still_checks_full_scope(env):
    records = many_records(env, 6)
    receipt = env.permissions.grant(
        env.snapshot(records=tuple(records)), expected_revision=0, user_confirmed=True
    )
    with sqlite3.connect(env.database) as db:
        db.execute("DELETE FROM model_content_scope_seals")
        db.execute("DELETE FROM model_content_scope_members")
    with patch.object(
        env.permissions, "_evidence_batch", wraps=env.permissions._evidence_batch
    ) as batch:
        assert env.check(receipt) == receipt
        assert sum(len(call.args[1]) for call in batch.call_args_list) == 6


def test_seal_never_substitutes_for_current_source_or_profile(env):
    _, receipt = seal(env)
    env.root.rename(env.root.with_name("old-root"))
    env.root.mkdir()
    with pytest.raises(ModelContentPermissionError, match="source_"):
        env.check(receipt)


def test_corrupt_receipt_with_same_revision_cannot_reuse_seal(env):
    _, receipt = seal(env)
    with sqlite3.connect(env.database) as db:
        db.execute(
            "UPDATE model_content_permissions SET "
            "snapshot_json=json_set(snapshot_json,'$.evidence[0].content_hash',?)",
            ("e" * 64,),
        )
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)
