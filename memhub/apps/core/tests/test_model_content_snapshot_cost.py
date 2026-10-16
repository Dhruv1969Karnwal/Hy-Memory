"""Reuse derived fingerprints only after reading every raw row in this transaction."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

import one_memory.model_content_permissions as content
from one_memory.database_access import close_main_databases
from one_memory.model_content_permissions import ModelContentPermissionError
from tests.test_model_content_bulk import many_records
from tests.test_model_content_permissions import USER
from tests.test_model_content_permissions import env as env


def test_warm_scope_reuses_derivation_but_reads_complete_evidence_and_origins(env, monkeypatch):
    records = many_records(env, 40)
    receipt = env.permissions.grant(
        env.snapshot(records=tuple(records)), expected_revision=0, user_confirmed=True
    )
    connects, statements = [], []
    original_connect = sqlite3.connect

    def connect(*args, **kwargs):
        db = original_connect(*args, **kwargs)
        connects.append(db)
        db.set_trace_callback(statements.append)
        return db

    close_main_databases()  # pooled connections would skip the patch

    monkeypatch.setattr(sqlite3, "connect", connect)
    with patch.object(
        content, "model_content_evidence_snapshot", wraps=content.model_content_evidence_snapshot
    ) as derive:
        assert env.check(receipt) == receipt
        # A fresh connection per check, so each transaction is observed.
        close_main_databases()
        assert env.check(receipt) == receipt
        assert derive.call_count == 2, "Only the actual supplied input needs fresh derivation"
    assert len(connects) == 2
    assert sum("FROM evidence " in sql for sql in statements) == 2
    assert sum("FROM evidence_origins " in sql for sql in statements) == 2
    assert all("verbatim" in sql for sql in statements if "FROM evidence " in sql)
    assert all("raw_meta_json" in sql for sql in statements if "FROM evidence_origins " in sql)


@pytest.mark.parametrize("mutation", ["body", "author", "time", "origin", "delete", "user"])
def test_warm_cache_detects_unrequested_raw_mutations_without_hash_revision_changes(env, mutation):
    records = many_records(env, 4)
    receipt = env.permissions.grant(
        env.snapshot(records=tuple(records)), expected_revision=0, user_confirmed=True
    )
    assert env.check(receipt) == receipt
    sql = {
        "body": "UPDATE evidence SET verbatim='changed raw text' WHERE evidence_id=?",
        "author": "UPDATE evidence SET written_by='unknown' WHERE evidence_id=?",
        "time": "UPDATE evidence SET created_at='2026-09-01T00:00:00Z' WHERE evidence_id=?",
        "origin": (
            "UPDATE evidence_origins SET raw_meta_json='{\"changed\":true}' WHERE evidence_id=?"
        ),
        "delete": "DELETE FROM evidence WHERE evidence_id=?",
        "user": "UPDATE evidence SET user_id='other' WHERE evidence_id=?",
    }[mutation]
    with sqlite3.connect(env.database) as db:
        db.execute(sql, (records[-1].evidence_id,))
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)


def test_derived_cache_is_bounded_and_store_local(env, monkeypatch):
    monkeypatch.setattr(content, "_EVIDENCE_SNAPSHOT_CACHE_LIMIT", 1, raising=False)
    records = many_records(env, 4)
    receipt = env.permissions.grant(
        env.snapshot(records=tuple(records)), expected_revision=0, user_confirmed=True
    )
    assert env.check(receipt) == receipt
    assert len(env.permissions._evidence_snapshot_cache) <= 1
    fresh = content.ModelContentPermissionStore(env.database)
    assert len(fresh._evidence_snapshot_cache) == 0
    assert (
        fresh.check(
            user_id=USER,
            expected_revision=receipt.revision,
            expected_revocation_revision=receipt.revocation_revision,
            profile=env.profile,
            evidence=(env.record,),
        )
        == receipt
    )
    assert len(fresh._evidence_snapshot_cache) <= 1


def test_derived_cache_payload_limit_can_disable_retention(env, monkeypatch):
    monkeypatch.setattr(content, "_EVIDENCE_SNAPSHOT_CACHE_BYTES", 1, raising=False)
    receipt = env.grant()
    assert env.check(receipt) == receipt
    assert len(env.permissions._evidence_snapshot_cache) == 0


def test_larger_realistic_receipt_fits_bounded_parse_cache(env):
    from tests.test_model_content_parse_cost import (
        repeated_receipt_measurement,
        synthetic_large_receipt,
    )

    synthetic_large_receipt(env, count=45_419)
    result = repeated_receipt_measurement(env.permissions, repeats=2)
    assert result["snapshot_parses"] == 1


def test_large_grant_builds_a_valid_scope_seal_in_bounded_batches(env, monkeypatch):
    records = many_records(env, 1001)
    snapshot = env.snapshot(records=tuple(records))
    receipt = env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)

    with sqlite3.connect(env.database) as db:
        assert db.execute(
            "SELECT count(*) FROM model_content_scope_members WHERE user_id=? AND revision=?",
            (receipt.user_id, receipt.revision),
        ).fetchone()[0] == len(records)
        assert (
            db.execute(
                "SELECT valid FROM model_content_scope_seals WHERE user_id=? AND revision=?",
                (receipt.user_id, receipt.revision),
            ).fetchone()[0]
            == 1
        )

    def must_not_rescan(*_args, **_kwargs):
        raise AssertionError("a sealed large scope must not rescan every Evidence row")

    monkeypatch.setattr(env.permissions, "_validate_snapshot", must_not_rescan)
    assert env.check(receipt, records=(records[0],)) == receipt


def test_cached_metadata_cannot_hide_corrupt_origin_json(env):
    receipt = env.grant()
    assert env.check(receipt) == receipt
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE evidence_origins SET raw_meta_json='not-json'")
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)


@pytest.mark.parametrize("mutation", ["insert", "delete", "version", "kind"])
def test_cached_scope_checks_entire_origin_row_membership(env, mutation):
    records = many_records(env, 4)
    receipt = env.permissions.grant(
        env.snapshot(records=tuple(records)), expected_revision=0, user_confirmed=True
    )
    assert env.check(receipt) == receipt
    other = records[-1]
    with sqlite3.connect(env.database) as db:
        if mutation == "insert":
            db.execute(
                "INSERT INTO evidence_origins "
                "(evidence_id,origin,raw_meta_json,source_version,observed_at) "
                "VALUES (?,'manual_import','{}',NULL,?)",
                (other.evidence_id, other.imported_at.isoformat()),
            )
        else:
            sql = {
                "delete": "DELETE FROM evidence_origins WHERE evidence_id=? AND origin='file'",
                "version": (
                    "UPDATE evidence_origins SET source_version='changed' "
                    "WHERE evidence_id=? AND origin='file'"
                ),
                "kind": (
                    "UPDATE evidence_origins SET origin='unknown' "
                    "WHERE evidence_id=? AND origin='file'"
                ),
            }[mutation]
            db.execute(sql, (other.evidence_id,))
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)


def test_old_transaction_cannot_poison_new_transaction_derivation(env, monkeypatch):
    receipt = env.grant()
    old_batch = env.permissions._evidence_batch
    changed = False

    def write_while_old_snapshot_open(connection, identifiers):
        nonlocal changed
        if identifiers and not changed:
            changed = True
            with sqlite3.connect(env.database) as db:
                db.execute("UPDATE evidence SET verbatim='changed without revision'")
            # A new check is another caller, so it runs on a thread of its own.
            with ThreadPoolExecutor(1) as other_caller:
                later = other_caller.submit(env.check, receipt)
                with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
                    later.result()
        yield from old_batch(connection, identifiers)

    with monkeypatch.context() as scoped:
        scoped.setattr(env.permissions, "_evidence_batch", write_while_old_snapshot_open)
        assert env.check(receipt) == receipt
    assert changed
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)


def test_cache_key_is_not_delimiter_ambiguous(env):
    receipt = env.grant()
    assert env.check(receipt) == receipt
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE evidence SET workspace='a,b',project='c'")
    updated = env.evidence.get(env.record.evidence_id)
    next_receipt = env.permissions.grant(
        env.snapshot(records=(updated,)), expected_revision=1, user_confirmed=True
    )
    assert env.check(next_receipt, records=(updated,)) == next_receipt
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE evidence SET workspace='a',project='b,c'")
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(next_receipt, records=(updated,))
