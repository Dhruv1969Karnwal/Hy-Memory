"""Memoize parsing only; every permission decision still reads current SQLite state."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from sys import getsizeof
from threading import Barrier
from time import perf_counter
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import one_memory.model_content_permissions as content
from one_memory.database_access import close_main_databases
from one_memory.model_content_permissions import (
    ModelContentPermissionError,
    ModelContentPermissionSnapshot,
    ModelContentPermissionStore,
)
from tests.test_model_content_permissions import SOURCE, USER, evidence_value
from tests.test_model_content_permissions import env as env


def repeated_receipt_measurement(store, *, repeats=6):
    original = ModelContentPermissionSnapshot.model_validate_json
    with patch.object(ModelContentPermissionSnapshot, "model_validate_json", wraps=original) as spy:
        durations = []
        receipts = []
        for _ in range(repeats):
            started = perf_counter()
            receipts.append(store.receipt(user_id=USER))
            durations.append(perf_counter() - started)
    assert receipts[0] is not None
    assert all(receipt == receipts[0] for receipt in receipts)
    return {
        "references": len(receipts[0].snapshot.evidence),
        "reads": repeats,
        "snapshot_parses": spy.call_count,
        "seconds_per_read": durations,
    }


def synthetic_large_receipt(env, count=12_223):
    """Only the parsing benchmark uses synthetic refs without corresponding bodies."""
    snapshot = env.snapshot()
    template = snapshot.evidence[0]
    snapshot = snapshot.model_copy(
        update={
            "evidence": tuple(
                template.model_copy(update={"evidence_id": f"synthetic-evidence-{index:06d}"})
                for index in range(count)
            ),
        }
    )
    receipt = content.ModelContentPermissionReceipt(
        user_id=USER,
        revision=1,
        revocation_revision=0,
        granted_at=template.imported_at,
        revoked_at=None,
        snapshot=snapshot,
    )
    with env.permissions._connect(writing=True) as db:
        env.permissions._append(db, receipt)
    return receipt


def test_large_unchanged_receipt_is_parsed_once_not_once_per_poll(env, record_property):
    synthetic_large_receipt(env)
    measured = repeated_receipt_measurement(env.permissions)
    record_property("parse_benchmark", json.dumps(measured))
    print(json.dumps(measured), flush=True)
    assert measured["references"] == 12_223
    assert measured["snapshot_parses"] == 1, measured


def test_concurrent_cold_reads_share_only_identical_parse_work(env):
    receipt = env.grant()
    barrier = Barrier(4)
    original = ModelContentPermissionSnapshot.model_validate_json

    def read(_):
        barrier.wait(timeout=5)
        return env.permissions.receipt(user_id=USER)

    with (
        patch.object(ModelContentPermissionSnapshot, "model_validate_json", wraps=original) as spy,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        results = list(pool.map(read, range(4)))
    assert results == [receipt] * 4
    assert spy.call_count == 1


@pytest.mark.parametrize("mutation", ["corrupt_json", "owner", "hash", "generation"])
def test_same_revision_snapshot_tampering_cannot_reuse_a_cached_parse(env, mutation):
    receipt = env.grant()
    assert env.check(receipt) == receipt
    with sqlite3.connect(env.database) as db:
        raw = db.execute("SELECT snapshot_json FROM model_content_permissions").fetchone()[0]
        body = json.loads(raw)
        if mutation == "owner":
            body["user_id"] = "other-user"
        elif mutation == "hash":
            body["evidence"][0]["content_hash"] = "0" * 64
        elif mutation == "generation":
            body["generation"]["model"] = "tampered-model"
        changed = "{}" if mutation == "corrupt_json" else json.dumps(body)
        db.execute("UPDATE model_content_permissions SET snapshot_json=?", (changed,))
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt)
    # Returning to exactly the stored bytes is allowed to reuse parsing, not authority.
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE model_content_permissions SET snapshot_json=?", (raw,))
    assert env.check(receipt) == receipt


@pytest.mark.parametrize(
    "field", ["user_id", "revision", "revocation_revision", "granted_at", "revoked_at"]
)
def test_cached_receipt_is_keyed_by_all_database_metadata(env, field):
    receipt = env.grant()
    assert env.permissions.receipt(user_id=USER) == receipt
    values = {
        "user_id": "other-user",
        "revision": 2,
        "revocation_revision": 1,
        "granted_at": (receipt.granted_at + timedelta(seconds=1)).isoformat(),
        "revoked_at": receipt.granted_at.isoformat(),
    }
    with sqlite3.connect(env.database) as db:
        db.execute(f"UPDATE model_content_permissions SET {field}=?", (values[field],))
    if field == "user_id":
        assert env.permissions.receipt(user_id=USER) is None
        with pytest.raises(ModelContentPermissionError, match="permission_snapshot_invalid"):
            env.permissions.receipt(user_id="other-user")
        return
    changed = env.permissions.receipt(user_id=USER)
    assert changed is not None and changed != receipt
    if field == "granted_at":
        assert changed.granted_at == receipt.granted_at + timedelta(seconds=1)
    else:
        with pytest.raises(ModelContentPermissionError):
            env.check(receipt)


def test_byte_different_json_is_reparsed_even_when_semantically_equal(env):
    receipt = env.grant()
    original = ModelContentPermissionSnapshot.model_validate_json
    with patch.object(ModelContentPermissionSnapshot, "model_validate_json", wraps=original) as spy:
        assert env.permissions.receipt(user_id=USER) == receipt
        with sqlite3.connect(env.database) as db:
            raw = db.execute("SELECT snapshot_json FROM model_content_permissions").fetchone()[0]
            changed = json.dumps(json.loads(raw), indent=1)
            assert raw != changed
            db.execute("UPDATE model_content_permissions SET snapshot_json=?", (changed,))
        assert env.permissions.receipt(user_id=USER) == receipt
        assert env.permissions.receipt(user_id=USER) == receipt
    assert spy.call_count == 2


@pytest.mark.parametrize(
    "mutation", ["revoke", "delete", "profile", "root", "source", "evidence", "origins"]
)
def test_cache_hit_still_rechecks_live_authority_and_all_evidence(env, mutation):
    other = env.evidence.put(evidence_value("not-the-requested-record")).record
    receipt = env.permissions.grant(
        env.snapshot(records=(env.record, other)), expected_revision=0, user_confirmed=True
    )
    assert env.check(receipt) == receipt
    if mutation == "revoke":
        env.permissions.revoke(user_id=USER, expected_revision=1, user_confirmed=True)
    elif mutation == "root":
        env.root.rename(env.root.with_name("old-projects"))
        env.root.mkdir()
    elif mutation == "source":
        env.sources.revoke(user_id=USER, source_id=SOURCE)
    else:
        with sqlite3.connect(env.database) as db:
            if mutation == "delete":
                db.execute("DELETE FROM model_content_permissions")
            elif mutation == "profile":
                db.execute("UPDATE model_profiles SET model='changed-without-revision'")
            elif mutation == "evidence":
                db.execute(
                    "UPDATE evidence SET verbatim='changed without revision' WHERE evidence_id=?",
                    (other.evidence_id,),
                )
            else:
                db.execute(
                    "UPDATE evidence_origins SET source_version='changed' WHERE evidence_id=?",
                    (other.evidence_id,),
                )
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt)


def test_warm_parse_and_evidence_checks_share_the_current_transaction(env, monkeypatch):
    receipt = env.grant()
    assert env.check(receipt) == receipt
    validate = env.permissions._validate_snapshot_authority
    changed = False

    def edit_after_receipt_read(connection, snapshot):
        nonlocal changed
        if not changed:
            changed = True
            with sqlite3.connect(env.database) as db:
                db.execute("UPDATE evidence SET verbatim='changed during validation'")
        return validate(connection, snapshot)

    with monkeypatch.context() as scoped:
        scoped.setattr(env.permissions, "_validate_snapshot_authority", edit_after_receipt_read)
        # The receipt SELECT already opened this transaction's SQLite snapshot.
        assert env.check(receipt) == receipt
    assert changed
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)


def test_repeated_check_reuses_parse_but_reads_and_validates_each_transaction(env, monkeypatch):
    receipt = env.grant()
    original_connect = sqlite3.connect
    statements = []
    connections = []

    def connect(*args, **kwargs):
        db = original_connect(*args, **kwargs)
        connections.append(db)
        db.set_trace_callback(statements.append)
        return db

    close_main_databases()  # pooled connections would skip the patch

    monkeypatch.setattr(sqlite3, "connect", connect)
    original = ModelContentPermissionSnapshot.model_validate_json
    with (
        patch.object(ModelContentPermissionSnapshot, "model_validate_json", wraps=original) as spy,
        patch.object(
            env.permissions,
            "_validate_snapshot_authority",
            wraps=env.permissions._validate_snapshot_authority,
        ) as validate,
    ):
        assert env.check(receipt) == receipt
        # A fresh connection per check, so each transaction is observed.
        close_main_databases()
        assert env.check(receipt) == receipt
        assert spy.call_count == 1
        assert validate.call_count == 2
    assert len(connections) == 2
    assert sum("FROM model_content_permissions WHERE user_id=" in sql for sql in statements) == 2
    assert all(
        call.args[0] is db for call, db in zip(validate.call_args_list, connections, strict=True)
    )


def test_historical_cached_receipt_never_becomes_current_after_revoke(env):
    first = env.grant()
    assert env.check(first) == first
    revoked = env.permissions.revoke(user_id=USER, expected_revision=1, user_confirmed=True)
    assert env.permissions.receipt(user_id=USER, revision=1) == first
    assert env.permissions.receipt(user_id=USER) == revoked
    with pytest.raises(ModelContentPermissionError, match="permission_revoked"):
        env.check(first)


def test_cached_parsed_snapshot_is_deeply_immutable(env):
    env.grant()
    receipt = env.permissions.receipt(user_id=USER)
    for model, field, value in (
        (receipt, "revoked_at", receipt.granted_at),
        (receipt.snapshot, "evidence", ()),
        (receipt.snapshot.evidence[0], "revision", 9),
        (receipt.snapshot.sources[0], "revision", 9),
        (receipt.snapshot.profile, "model", "changed"),
        (receipt.snapshot.generation, "model", "changed"),
    ):
        with pytest.raises(ValidationError, match="frozen_instance"):
            setattr(model, field, value)
    assert env.permissions.receipt(user_id=USER) == receipt


def test_parse_cache_is_entry_bounded_and_store_local(env, monkeypatch):
    monkeypatch.setattr(content, "_RECEIPT_PARSE_CACHE_LIMIT", 2, raising=False)
    first = env.grant()
    second = env.permissions.revoke(user_id=USER, expected_revision=1, user_confirmed=True)
    third = env.permissions.grant(env.snapshot(), expected_revision=2, user_confirmed=True)
    store = ModelContentPermissionStore(env.database)
    original = ModelContentPermissionSnapshot.model_validate_json
    with patch.object(ModelContentPermissionSnapshot, "model_validate_json", wraps=original) as spy:
        for revision, expected in ((1, first), (2, second), (3, third), (1, first), (1, first)):
            assert store.receipt(user_id=USER, revision=revision) == expected
        assert spy.call_count == 4
        fresh = ModelContentPermissionStore(env.database)
        assert fresh.receipt(user_id=USER, revision=1) == first
        assert spy.call_count == 5


def test_oversized_parse_result_is_not_retained(env, monkeypatch):
    monkeypatch.setattr(content, "_RECEIPT_PARSE_CACHE_BYTES", 1, raising=False)
    env.grant()
    measured = repeated_receipt_measurement(env.permissions, repeats=2)
    assert measured["snapshot_parses"] == 2


def test_parse_cache_total_bytes_are_bounded(env, monkeypatch):
    receipt = env.grant()
    with sqlite3.connect(env.database) as db:
        row = db.execute("SELECT * FROM model_content_permissions").fetchone()
    entry_size = getsizeof(row) + sum(getsizeof(value) for value in row)
    monkeypatch.setattr(content, "_RECEIPT_PARSE_CACHE_BYTES", entry_size * 2 + 64)
    monkeypatch.setattr(content, "_RECEIPT_PARSE_CACHE_LIMIT", 8)
    store = ModelContentPermissionStore(env.database)
    original = ModelContentPermissionSnapshot.model_validate_json
    with patch.object(ModelContentPermissionSnapshot, "model_validate_json", wraps=original) as spy:
        for offset in (0, 1, 2, 0, 0):
            stamp = receipt.granted_at + timedelta(seconds=offset)
            with sqlite3.connect(env.database) as db:
                db.execute(
                    "UPDATE model_content_permissions SET granted_at=?", (stamp.isoformat(),)
                )
            assert store.receipt(user_id=USER).granted_at == stamp
        assert spy.call_count == 4
