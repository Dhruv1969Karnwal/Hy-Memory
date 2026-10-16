"""Bulk consent reads preserve exact identities and SQLite snapshot boundaries."""

import asyncio
import sqlite3

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from one_memory.database_access import close_main_databases
from one_memory.imports.models import EvidenceOrigin, OriginMetadata, WrittenBy
from one_memory.model_content_permissions import (
    ModelContentPermissionError,
    ModelContentPermissionSnapshot,
    model_content_evidence_snapshot,
)
from one_memory.model_content_routes import model_content_router
from tests.test_model_content_permissions import (
    SOURCE,
    USER,
    evidence_value,
)
from tests.test_model_content_permissions import (
    env as environment_fixture,
)


@pytest.fixture
def env(tmp_path):
    return environment_fixture.__wrapped__(tmp_path)


def many_records(environment, count):
    records = [environment.record]
    with environment.evidence._connect(writing=True) as connection:
        for index in range(1, count):
            value = evidence_value(str(index), text=f"Synthetic {index}: " + "x" * 128)
            if index % 2:
                value = value.model_copy(
                    update={
                        "role": "assistant",
                        "written_by": WrittenBy.AGENT,
                        "origin_meta": {
                            EvidenceOrigin.FILE: OriginMetadata(raw_meta={"nested": [1, "text"]}),
                            EvidenceOrigin.HOOK: OriginMetadata(source_version="fixture-v1"),
                        },
                    }
                )
            records.append(environment.evidence.put(value, connection=connection).record)
    return records


def app_for(environment):
    app = FastAPI()
    app.state.model_profile_store = environment.profiles
    app.state.evidence_store = environment.evidence
    app.state.model_content_permissions = environment.permissions
    app.include_router(model_content_router([]))
    return app


def preview(environment, **changes):
    async def request():
        async with AsyncClient(
            transport=ASGITransport(app=app_for(environment)), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/v1/model-content-permission/preview",
                json={
                    "profile_id": environment.profile.profile_id,
                    "profile_revision": environment.profile.revision,
                    "source_ids": [SOURCE],
                    "all_imported_from_selected_sources": True,
                    **changes,
                },
            )

    return asyncio.run(request())


def observe_connections(monkeypatch):
    connects, statements = [], []
    original = sqlite3.connect

    def observed(*args, **kwargs):
        connection = original(*args, **kwargs)
        connects.append(connection)
        connection.set_trace_callback(lambda sql: statements.append(sql))
        return connection

    close_main_databases()  # pooled connections would skip the patch

    monkeypatch.setattr(sqlite3, "connect", observed)
    return connects, statements


def test_preview_uses_bounded_bulk_reads_without_per_id_connections(env, monkeypatch):
    records = many_records(env, 1000)
    expected = env.snapshot(records=tuple(records))
    connects, statements = observe_connections(monkeypatch)
    response = preview(env)
    assert response.status_code == 200
    actual = ModelContentPermissionSnapshot.model_validate(response.json())
    assert actual == expected
    assert len(connects) <= 2, "preview still opens one connection per Evidence ID"
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) <= 20, "preview still rereads each record separately"
    assert env.permissions.receipt(user_id=USER) is None


def test_hot_check_is_bulk_and_reuses_same_transaction_verified_refs(env, monkeypatch):
    records = many_records(env, 1000)
    receipt = env.permissions.grant(
        env.snapshot(records=tuple(records)), expected_revision=0, user_confirmed=True
    )
    connects, statements = observe_connections(monkeypatch)
    assert env.check(receipt, records=(records[0],)) == receipt
    assert len(connects) == 1
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) <= 20, (
        "checking one input still performs per-record SQL for the entire scope"
    )


def test_all_source_preview_and_future_boundary_share_one_read_snapshot(env, monkeypatch):
    records = many_records(env, 4)
    expected = env.snapshot(records=tuple(records), future=True)
    source = env.permissions._source
    changed = False

    def concurrent_insert(connection, user_id, source_id):
        nonlocal changed
        result = source(connection, user_id, source_id)
        if not changed:
            changed = True
            env.evidence.put(evidence_value("arrived-during-preview"))
        return result

    monkeypatch.setattr(env.permissions, "_source", concurrent_insert)
    response = preview(env, include_future_additions=True)
    assert response.status_code == 200
    actual = ModelContentPermissionSnapshot.model_validate(response.json())
    assert changed and actual == expected, "preview mixed old IDs with a newer future watermark"
    receipt = env.permissions.grant(actual, expected_revision=0, user_confirmed=True)
    assert receipt.snapshot == expected
    added = env.evidence.put(evidence_value("arrived-during-preview")).record
    assert added.evidence_id not in {ref.evidence_id for ref in actual.evidence}
    assert env.check(receipt, records=(added,)) == receipt


@pytest.mark.parametrize("mutation", ["body", "author", "origins", "delete"])
def test_hot_check_still_rejects_changed_unrequested_scope_record(env, mutation):
    records = many_records(env, 4)
    receipt = env.permissions.grant(
        env.snapshot(records=tuple(records)), expected_revision=0, user_confirmed=True
    )
    other = records[-1]
    with sqlite3.connect(env.database) as db:
        sql = {
            "body": (
                "UPDATE evidence SET verbatim='changed without revision or hash' "
                "WHERE evidence_id=?"
            ),
            "author": "UPDATE evidence SET written_by='unknown' WHERE evidence_id=?",
            "origins": "UPDATE evidence_origins SET source_version='changed' WHERE evidence_id=?",
            "delete": "DELETE FROM evidence WHERE evidence_id=?",
        }[mutation]
        db.execute(sql, (other.evidence_id,))
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt, records=(records[0],))


@pytest.mark.parametrize(
    "selection,code",
    [
        ("missing", "evidence_not_found"),
        ("other_user", "evidence_not_found"),
        ("other_source", "invalid_content_scope"),
        ("duplicate", "invalid_content_scope"),
        ("ambiguous", "ambiguous_content_scope"),
    ],
)
def test_bulk_preview_keeps_exact_selection_failures(env, selection, code):
    other_user = env.evidence.put(evidence_value("other-user", user_id="other")).record
    other_source = env.evidence.put(evidence_value("other-source", source_id="cursor")).record
    identifiers = {
        "missing": ["not-present"],
        "other_user": [other_user.evidence_id],
        "other_source": [other_source.evidence_id],
        "duplicate": [env.record.evidence_id, env.record.evidence_id],
        "ambiguous": [env.record.evidence_id],
    }[selection]
    response = preview(
        env, evidence_ids=identifiers, all_imported_from_selected_sources=selection == "ambiguous"
    )
    assert response.status_code == 409
    assert response.json()["detail"] == f"model_content_{code}"
    assert env.permissions.receipt(user_id=USER) is None


def test_bulk_snapshot_preserves_every_existing_hash_field(env):
    records = many_records(env, 8)
    expected = tuple(
        sorted(
            (model_content_evidence_snapshot(record) for record in records),
            key=lambda ref: ref.evidence_id,
        )
    )
    actual = ModelContentPermissionSnapshot.model_validate(preview(env).json())
    assert actual.evidence == expected
    assert not any(record.verbatim in actual.model_dump_json() for record in records)


def test_identifier_selection_does_not_mix_a_newer_future_watermark(env, monkeypatch):
    records = many_records(env, 4)
    expected = env.snapshot(records=tuple(records), future=True)
    original_connect = sqlite3.connect
    inserted = False

    class ConcurrentInsert(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal inserted
            cursor = super().execute(sql, parameters)
            if sql.startswith("SELECT evidence_id FROM evidence WHERE user_id=") and not inserted:
                inserted = True
                env.evidence.put(evidence_value("after-identifier-selection"))
            return cursor

    def connect(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=ConcurrentInsert)

    close_main_databases()  # pooled connections would skip the patch

    monkeypatch.setattr(sqlite3, "connect", connect)
    response = preview(env, include_future_additions=True)
    assert inserted and response.status_code == 200
    snapshot = ModelContentPermissionSnapshot.model_validate(response.json())
    assert snapshot == expected
    receipt = env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
    assert receipt.snapshot == expected
    added = env.evidence.put(evidence_value("after-identifier-selection")).record
    assert added.evidence_id not in {ref.evidence_id for ref in snapshot.evidence}
    assert env.check(receipt, records=(added,)) == receipt


def test_bulk_batches_keep_one_snapshot_and_next_check_observes_an_edit(env, monkeypatch):
    records = many_records(env, 1001)
    snapshot = env.snapshot(records=tuple(records))
    receipt = env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
    # Large grants now build a bounded scope seal. Invalidate it explicitly to
    # exercise the full-snapshot fallback and its single read transaction.
    with sqlite3.connect(env.database) as db:
        db.execute(
            "UPDATE model_content_scope_seals SET valid=0 WHERE user_id=? AND revision=?",
            (receipt.user_id, receipt.revision),
        )
    ordered = sorted(records, key=lambda record: record.evidence_id)
    later = ordered[-1]
    original_batch = env.permissions._evidence_batch
    calls = []

    def edit_after_first_batch(connection, identifiers):
        yield from original_batch(connection, identifiers)
        calls.append(len(identifiers))
        if len(calls) == 1:
            with sqlite3.connect(env.database) as db:
                db.execute(
                    "UPDATE evidence SET verbatim='edited but same stored hash' "
                    "WHERE evidence_id=?",
                    (later.evidence_id,),
                )

    with monkeypatch.context() as scoped:
        scoped.setattr(env.permissions, "_evidence_batch", edit_after_first_batch)
        assert env.check(receipt, records=(ordered[0],)) == receipt
    assert calls == [500, 500, 1]
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt, records=(ordered[0],))


def test_same_transaction_reuse_does_not_trust_forged_request_metadata(env):
    receipt = env.grant()
    forged = env.record.model_copy(update={"written_by": WrittenBy.AGENT})
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt, records=(forged,))
    with pytest.raises(ModelContentPermissionError, match="invalid_content_scope"):
        env.check(receipt, records=(env.record, env.record))


@pytest.mark.parametrize(
    "column,value",
    [
        ("raw_meta_json", "not-json"),
        ("origin", "unknown-origin"),
        ("source_version", "x" * 129),
    ],
)
def test_bulk_corrupt_origins_fail_closed(env, column, value):
    receipt = env.grant()
    with sqlite3.connect(env.database) as db:
        db.execute(f"UPDATE evidence_origins SET {column}=?", (value,))
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.check(receipt)
