"""Confirmation must tolerate append-only import without broadening consent."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

import one_memory.model_content_permissions as content
from one_memory.model_content_permissions import ModelContentPermissionError
from tests.test_model_content_permissions import SOURCE, USER, Environment, evidence_value
from tests.test_model_content_permissions import env as env


def test_confirmation_validation_does_not_hold_the_only_writer(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = env.snapshot(future=True)
    validating = Event()
    release = Event()
    original = env.permissions._validate_snapshot

    def paused_validation(connection, value):
        result = original(connection, value)
        validating.set()
        assert release.wait(2)
        return result

    monkeypatch.setattr(env.permissions, "_validate_snapshot", paused_validation)
    with ThreadPoolExecutor(2) as workers:
        confirmation = workers.submit(
            env.permissions.grant,
            snapshot,
            user_confirmed=True,
            expected_revision=0,
        )
        assert validating.wait(1)
        imported = workers.submit(env.evidence.put, evidence_value("during-long-validation"))
        try:
            assert imported.result(timeout=0.5).record.message_id == "during-long-validation"
        finally:
            release.set()
        assert confirmation.result(timeout=2).snapshot == snapshot


def test_confirmation_rejects_evidence_changed_after_read_validation(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = env.snapshot()
    validated = Event()
    release = Event()
    original = env.permissions._validate_snapshot

    def paused_validation(connection, value):
        result = original(connection, value)
        validated.set()
        assert release.wait(2)
        return result

    monkeypatch.setattr(env.permissions, "_validate_snapshot", paused_validation)
    with ThreadPoolExecutor(1) as worker:
        confirmation = worker.submit(
            env.permissions.grant,
            snapshot,
            user_confirmed=True,
            expected_revision=0,
        )
        assert validated.wait(1)
        env.evidence.put(evidence_value("selected", text="changed after validation"))
        release.set()
        with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
            confirmation.result(timeout=2)
    assert env.permissions.receipt(user_id=USER) is None


@pytest.mark.parametrize("via_ids", [False, True])
@pytest.mark.parametrize("source_id", [SOURCE, "cursor"])
def test_issued_future_preview_survives_new_originals(
    env: Environment, source_id: str, via_ids: bool
) -> None:
    snapshot = (
        env.permissions.snapshot_from_ids(
            user_id=USER,
            source_ids=(SOURCE,),
            profile=env.profile,
            all_imported_from_selected_sources=True,
            include_future_additions=True,
        )
        if via_ids
        else env.snapshot(future=True)
    )
    added = env.evidence.put(evidence_value("during-confirmation", source_id=source_id)).record
    receipt = env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
    assert receipt.snapshot == snapshot
    assert env.check(receipt) == receipt
    if source_id == SOURCE:
        assert env.check(receipt, records=(added,)) == receipt
    else:
        with pytest.raises(ModelContentPermissionError, match="evidence_not_permitted"):
            env.check(receipt, records=(added,))


def test_a_preview_does_not_authorize_omitted_preexisting_originals(env: Environment) -> None:
    omitted = env.evidence.put(evidence_value("omitted-old")).record
    snapshot = env.snapshot(future=True)
    added = env.evidence.put(evidence_value("new-during-confirmation")).record
    receipt = env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
    assert env.check(receipt, records=(added,)) == receipt
    with pytest.raises(ModelContentPermissionError, match="evidence_not_permitted"):
        env.check(receipt, records=(omitted,))


def test_forged_watermark_cannot_reuse_an_issued_preview(env: Environment) -> None:
    env.evidence.put(evidence_value("omitted-old"))
    snapshot = env.snapshot(future=True)
    env.evidence.put(evidence_value("new-during-confirmation"))
    forged = snapshot.model_copy(update={"future_after_sequence": 0})
    with pytest.raises(ModelContentPermissionError, match="content_scope_changed"):
        env.permissions.grant(forged, expected_revision=0, user_confirmed=True)
    assert env.permissions.receipt(user_id=USER) is None


def test_a_different_exact_selection_cannot_reuse_preview_issuance(env: Environment) -> None:
    old = env.evidence.put(evidence_value("second-selected")).record
    snapshot = env.snapshot(records=(env.record, old), future=True)
    env.evidence.put(evidence_value("new-during-confirmation"))
    changed = snapshot.model_copy(update={"evidence": (snapshot.evidence[0],)})
    with pytest.raises(ModelContentPermissionError, match="content_scope_changed"):
        env.permissions.grant(changed, expected_revision=0, user_confirmed=True)


def test_changed_original_still_invalidates_issued_preview(env: Environment) -> None:
    snapshot = env.snapshot(future=True)
    env.evidence.put(evidence_value("selected", text="Revised original"))
    env.evidence.put(evidence_value("new-during-confirmation"))
    with pytest.raises(ModelContentPermissionError, match="evidence_changed"):
        env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)


def test_restart_requires_repreview_but_does_not_destroy_existing_consent(env: Environment) -> None:
    snapshot = env.snapshot(future=True)
    env.evidence.put(evidence_value("new-during-confirmation"))
    env.permissions = content.ModelContentPermissionStore(env.database)
    with pytest.raises(ModelContentPermissionError, match="content_scope_changed"):
        env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
    current = env.snapshot(future=True)
    receipt = env.permissions.grant(current, expected_revision=0, user_confirmed=True)
    env.permissions = content.ModelContentPermissionStore(env.database)
    assert env.check(receipt) == receipt


def test_issued_preview_is_bound_to_the_consent_revision(env: Environment) -> None:
    snapshot = env.snapshot(future=True)
    first = env.grant()
    revoked = env.permissions.revoke(
        user_id=USER, expected_revision=first.revision, user_confirmed=True
    )
    env.evidence.put(evidence_value("new-after-revocation"))
    with pytest.raises(ModelContentPermissionError, match="content_scope_changed"):
        env.permissions.grant(snapshot, expected_revision=revoked.revision, user_confirmed=True)


def test_preview_issuance_expires_without_changing_accepted_scope(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(content, "monotonic", lambda: clock[0])
    snapshot = env.snapshot(future=True)
    env.evidence.put(evidence_value("new-during-confirmation"))
    clock[0] += content.PREVIEW_ISSUANCE_TTL_SECONDS + 1
    with pytest.raises(ModelContentPermissionError, match="content_scope_changed"):
        env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)


def test_preview_issuance_storage_is_bounded(
    env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(content, "PREVIEW_ISSUANCE_LIMIT", 2)
    first = env.snapshot(future=True)
    env.evidence.put(evidence_value("second"))
    env.snapshot(future=True)
    env.evidence.put(evidence_value("third"))
    latest = env.snapshot(future=True)
    env.evidence.put(evidence_value("fourth"))
    with pytest.raises(ModelContentPermissionError, match="content_scope_changed"):
        env.permissions.grant(first, expected_revision=0, user_confirmed=True)
    receipt = env.permissions.grant(latest, expected_revision=0, user_confirmed=True)
    assert receipt.snapshot == latest
