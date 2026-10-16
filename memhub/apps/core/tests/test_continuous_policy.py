import sqlite3
from contextlib import contextmanager
from uuid import uuid4

import pytest

from one_memory import continuous_policy as policy
from one_memory import ingest_journal as journal
from one_memory.imports.sources.registry import RootCandidateKind
from one_memory.model_profiles import ModelProfilePatch
from tests.test_model_content_permissions import SOURCE, evidence_value
from tests.test_model_content_permissions import env as env  # noqa: F401


@contextmanager
def transaction(environment):
    db = sqlite3.connect(environment.database)
    db.row_factory = sqlite3.Row
    db.execute("BEGIN IMMEDIATE")
    try:
        journal.initialize(db)
        policy.initialize(db)
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def enable(db, environment, **overrides):
    values = dict(
        profile=environment.profile,
        expected_revision=0,
        expected_legacy_revision=0,
        operation_id=str(uuid4()),
        backlog_mode="none",
        user_confirmed=True,
    )
    values.update(overrides)
    return policy.enable(db, **values)


def test_explicit_policy_never_upgrades_legacy_or_sends_or_creates_tasks(env):
    with transaction(env) as db:
        assert policy.current(db) is None
        receipt = enable(db, env)
        assert policy.current(db) == receipt
        assert db.execute("SELECT COUNT(*) FROM model_content_permissions").fetchone()[0] == 0
        assert receipt.backlog_events == ()
        with pytest.raises(policy.ContinuousPolicyError, match="evidence_outside_policy"):
            policy.check(db, receipt, evidence_ids=(env.record.evidence_id,))


def test_backlog_snapshot_over_one_hundred_thousand_is_compact(monkeypatch, env) -> None:
    count = 100_001

    class Database:
        in_transaction = True

        def execute(self, _sql, _values):
            return iter(((SOURCE, count, 1),))

    monkeypatch.setattr(policy.ingest_journal, "watermark", lambda _db: count)
    monkeypatch.setattr(
        policy.ModelContentPermissionStore,
        "_source",
        lambda _db, _user, _source: object(),
    )

    snapshot = policy.snapshot_backlog(Database())  # type: ignore[arg-type]

    assert snapshot.after_seq == count
    assert snapshot.source_ids == (SOURCE,)
    assert snapshot.evidence_count == count
    assert snapshot.conversation_count == 1
    with transaction(env) as db:
        receipt = enable(db, env, backlog_mode="include", backlog_snapshot=snapshot)
    assert receipt.backlog_through_seq == count
    assert receipt.backlog_source_ids == (SOURCE,)
    assert receipt.backlog_count == count
    assert receipt.backlog_conversation_count == 1
    assert receipt.backlog_events == ()
    assert len(receipt.model_dump_json()) < 5_000


def test_legacy_backlog_receipt_remains_readable(env) -> None:
    with transaction(env) as db:
        receipt = enable(db, env)
        legacy = receipt.model_dump(
            exclude={"backlog_through_seq", "backlog_source_ids", "backlog_count"}
        )
        legacy["backlog_mode"] = "include"
        legacy["backlog_events"] = (1,)
        restored = policy.ContinuousPolicy.model_validate(legacy)
        db.execute(
            "UPDATE continuous_policy_history SET receipt_json=? WHERE user_id=?",
            (restored.model_dump_json(), restored.user_id),
        )
        assert policy.check(db, restored, evidence_ids=(env.record.evidence_id,)) == restored


@pytest.mark.parametrize("confirmation", [False, 1, "true", None])
def test_explicit_confirmation_is_required(env, confirmation):
    with transaction(env) as db:
        with pytest.raises(policy.ContinuousPolicyError):
            enable(db, env, user_confirmed=confirmation)
        assert policy.current(db) is None


def test_future_old_dated_import_and_new_allowed_source_are_eligible(env, tmp_path):
    with transaction(env) as db:
        receipt = enable(db, env)
    other_root = tmp_path / "other-source"
    other_root.mkdir()
    env.sources.grant(
        user_id="local",
        source_id="cursor",
        root=other_root,
        candidate_kind=RootCandidateKind.DEFAULT,
    )
    with transaction(env) as db:
        value = evidence_value("after", source_id="cursor")
        result = env.evidence.put(value, connection=db)
        event = journal.record(db, result.record.evidence_id)
        assert event.seq > receipt.after_seq
        assert policy.check(db, receipt, evidence_ids=(result.record.evidence_id,)) == receipt


def test_include_uses_current_source_authority_not_the_enable_time_snapshot(env, tmp_path):
    old_ungranted = env.evidence.put(evidence_value("old-cursor", source_id="cursor")).record
    with transaction(env) as db:
        receipt = enable(db, env, backlog_mode="include")
        assert receipt.backlog_source_ids == (SOURCE,)

    other_root = tmp_path / "later-authorized-source"
    other_root.mkdir()
    env.sources.grant(
        user_id="local",
        source_id="cursor",
        root=other_root,
        candidate_kind=RootCandidateKind.DEFAULT,
    )
    with transaction(env) as db:
        assert policy.check(db, receipt, evidence_ids=(old_ungranted.evidence_id,)) == receipt


def test_include_keeps_legacy_backlog_fields_readable_without_using_them_as_a_boundary(env):
    with transaction(env) as db:
        receipt = enable(db, env, backlog_mode="include")
        assert receipt.backlog_events == ()
        assert receipt.backlog_through_seq == 1
        assert receipt.backlog_source_ids == (SOURCE,)
        policy.check(db, receipt, evidence_ids=(env.record.evidence_id,))
        db.execute(
            "UPDATE evidence SET workspace='new identity' WHERE evidence_id=?",
            (env.record.evidence_id,),
        )
        journal.record(db, env.record.evidence_id, cause="maintenance")
        assert policy.check(db, receipt, evidence_ids=(env.record.evidence_id,)) == receipt


def test_only_new_mode_keeps_the_enable_time_boundary(env):
    with transaction(env) as db:
        receipt = enable(db, env, backlog_mode="none")
        with pytest.raises(policy.ContinuousPolicyError, match="evidence_outside_policy"):
            policy.check(db, receipt, evidence_ids=(env.record.evidence_id,))


def test_pause_resume_and_revoke_are_cas_and_old_receipts_never_reactivate(env):
    with transaction(env) as db:
        receipt = enable(db, env)
        paused = policy.transition(
            db, action="pause", operation_id=str(uuid4()), user_confirmed=True, expected_revision=1
        )
        with pytest.raises(policy.ContinuousPolicyError):
            policy.check(db, receipt)
        with pytest.raises(policy.ContinuousPolicyError):
            policy.check(db, paused)
        resumed = policy.transition(
            db, action="resume", operation_id=str(uuid4()), user_confirmed=True, expected_revision=2
        )
        assert resumed.epoch == receipt.epoch and resumed.after_seq == receipt.after_seq
        revoked = policy.transition(
            db, action="revoke", operation_id=str(uuid4()), user_confirmed=True, expected_revision=3
        )
        with pytest.raises(policy.ContinuousPolicyError):
            policy.check(db, resumed)
        with pytest.raises(policy.ContinuousPolicyError, match="requires_new_enable"):
            policy.transition(
                db,
                action="resume",
                operation_id=str(uuid4()),
                user_confirmed=True,
                expected_revision=4,
            )
        new = enable(db, env, expected_revision=revoked.revision)
        assert new.epoch != receipt.epoch


def test_enable_idempotence_rejects_changed_payload_and_superseded_operation(env):
    with transaction(env) as db:
        request_id = str(uuid4())
        receipt = enable(db, env, operation_id=request_id)
        assert enable(db, env, operation_id=request_id) == receipt
        assert db.execute("SELECT COUNT(*) FROM continuous_policy_history").fetchone()[0] == 1
        with pytest.raises(policy.ContinuousPolicyError, match="payload_changed"):
            enable(db, env, operation_id=request_id, backlog_mode="include")
        policy.transition(
            db, action="revoke", operation_id=str(uuid4()), user_confirmed=True, expected_revision=1
        )
        with pytest.raises(policy.ContinuousPolicyError, match="superseded"):
            enable(db, env, operation_id=request_id)


def test_source_revoke_profile_change_and_legacy_changes_block_existing_policy(env):
    with transaction(env) as db:
        receipt = enable(db, env, backlog_mode="include")
    env.sources.revoke(user_id="local", source_id=SOURCE)
    with transaction(env) as db, pytest.raises(RuntimeError):
        policy.check(db, receipt, evidence_ids=(env.record.evidence_id,))
    env.profiles.update(
        env.profile.profile_id,
        ModelProfilePatch(expected_revision=1, model="different"),
        env.secrets,
    )
    with transaction(env) as db, pytest.raises(RuntimeError):
        policy.check(db, receipt)


def test_caller_rollback_includes_policy_and_backlog(env):
    with pytest.raises(RuntimeError, match="rollback"), transaction(env) as db:
        enable(db, env)
        raise RuntimeError("rollback")
    with transaction(env) as db:
        assert policy.current(db) is None


@pytest.mark.parametrize("value", [True, -1, "0", 0.5])
def test_revision_values_are_not_coerced(env, value):
    with (
        transaction(env) as db,
        pytest.raises(policy.ContinuousPolicyError, match="invalid_revision"),
    ):
        enable(db, env, expected_revision=value)


def test_legacy_permission_changes_do_not_invalidate_continuous_policy(env):
    legacy = env.grant()
    with transaction(env) as db:
        receipt = enable(db, env, expected_legacy_revision=legacy.revision)
    env.permissions.revoke(user_id="local", expected_revision=legacy.revision, user_confirmed=True)
    with transaction(env) as db:
        assert policy.check(db, receipt) == receipt
        paused = policy.transition(
            db,
            expected_revision=receipt.revision,
            action="pause",
            operation_id=str(uuid4()),
            user_confirmed=True,
        )
        resumed = policy.transition(
            db,
            expected_revision=paused.revision,
            action="resume",
            operation_id=str(uuid4()),
            user_confirmed=True,
        )
        assert resumed.state == "enabled"


def test_paused_policy_cannot_resume_after_model_identity_changes(env):
    with transaction(env) as db:
        enable(db, env)
        policy.transition(
            db, expected_revision=1, action="pause", operation_id=str(uuid4()), user_confirmed=True
        )
    env.profiles.update(
        env.profile.profile_id,
        ModelProfilePatch(expected_revision=1, model="changed-model"),
        env.secrets,
    )
    with transaction(env) as db:
        with pytest.raises(RuntimeError):
            policy.transition(
                db,
                expected_revision=2,
                action="resume",
                operation_id=str(uuid4()),
                user_confirmed=True,
            )
        assert policy.current(db).state == "paused"


def test_read_check_restores_callers_row_factory(env):
    with transaction(env) as db:
        receipt = enable(db, env)
        db.row_factory = None
        assert policy.check(db, receipt) == receipt
        assert db.row_factory is None


def test_prepared_input_cannot_be_replaced_by_a_newer_allowed_revision(env):
    with transaction(env) as db:
        receipt = enable(db, env)
        first = env.evidence.put(evidence_value("new-input"), connection=db).record
        lease = policy.bind_inputs(db, receipt, (first.evidence_id,))
        policy.recheck_inputs(db, lease)
        env.evidence.put(evidence_value("new-input", text="changed material"), connection=db)
        policy.check(db, receipt, evidence_ids=(first.evidence_id,))
        with pytest.raises(policy.ContinuousPolicyError, match="bound_input_changed"):
            policy.recheck_inputs(db, lease)


def test_prepared_source_cannot_survive_regrant_without_revalidation(env):
    with transaction(env) as db:
        receipt = enable(db, env, backlog_mode="include")
        lease = policy.bind_inputs(db, receipt, (env.record.evidence_id,))
    env.sources.revoke(user_id="local", source_id=SOURCE)
    env.sources.grant(
        user_id="local", source_id=SOURCE, root=env.root, candidate_kind=RootCandidateKind.DEFAULT
    )
    with (
        transaction(env) as db,
        pytest.raises(policy.ContinuousPolicyError, match="bound_source_authority_changed"),
    ):
        policy.recheck_inputs(db, lease)


@pytest.mark.parametrize("inputs", [(), ("duplicate", "duplicate")])
def test_lease_requires_explicit_unique_inputs(env, inputs):
    with transaction(env) as db:
        receipt = enable(db, env)
        with pytest.raises(policy.ContinuousPolicyError, match="unique_inputs"):
            policy.bind_inputs(db, receipt, inputs)
