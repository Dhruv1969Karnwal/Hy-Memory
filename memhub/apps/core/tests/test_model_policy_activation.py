import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from one_memory import continuous_policy as policy
from one_memory.model_policy_activation import ModelPolicyActivation
from one_memory.model_profiles import ModelCapabilities, ModelProfileInput, ModelProfilePatch
from one_memory.secrets import SecretStoreUnavailableError
from tests.test_model_content_permissions import env as env  # noqa: F401


def request(env, **overrides):
    values = dict(
        operation_id=str(uuid4()),
        profile_id=env.profile.profile_id,
        profile_revision=env.profile.revision,
        expected_policy_revision=0,
        expected_legacy_revision=0,
        backlog_mode="none",
        user_confirmed=True,
    )
    values.update(overrides)
    return values


def service(env):
    return ModelPolicyActivation(env.database, env.secrets)


def new_profile(env, *, cloud=False, key=None):
    values = dict(
        name="Other",
        provider_kind="openai_compatible" if cloud else "ollama",
        base_url="https://model.example/v1" if cloud else "http://127.0.0.1:11434",
        model="other-model",
        embedding_provider_kind="local_onnx",
        embedding_model="fixture-local-vector",
        embedding_dimensions=512,
    )
    if key is not None:
        values["api_key"] = key
    profile = env.profiles.create(ModelProfileInput(**values), env.secrets)
    env.profiles.record_capabilities(
        profile.profile_id,
        expected_revision=profile.revision,
        capabilities=ModelCapabilities(
            endpoint_reachable=True, model_available=True, embedding_available=True
        ),
        secrets=env.secrets,
    )
    return profile


def test_current_is_read_only_without_policy_tables(env):
    before = env.database.read_bytes()
    assert service(env).current() is None
    assert env.database.read_bytes() == before
    with sqlite3.connect(env.database) as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='continuous_policy_history'"
        ).fetchone()
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='evidence_ingest_events'"
        ).fetchone()


def test_switch_and_policy_commit_together_without_old_activate(env, monkeypatch):
    other = new_profile(env)
    monkeypatch.setattr(env.profiles, "activate", lambda *args: pytest.fail("old activate"))
    result = service(env).enable(**request(env, profile_id=other.profile_id))
    assert result.created is True
    assert result.policy.profile.profile_id == other.profile_id
    assert env.profiles.active_identity() == (other.profile_id, 1)
    assert service(env).current() == result.policy
    assert env.permissions.receipt(user_id="local") is None


def test_policy_append_failure_rolls_back_active_and_initialization(env, monkeypatch):
    other = new_profile(env)
    monkeypatch.setattr(
        policy, "_append", lambda *args: (_ for _ in ()).throw(RuntimeError("forced"))
    )
    with pytest.raises(RuntimeError):
        service(env).enable(**request(env, profile_id=other.profile_id))
    assert env.profiles.active_identity() == (env.profile.profile_id, 1)
    assert service(env).current() is None
    with sqlite3.connect(env.database) as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='evidence_ingest_events'"
        ).fetchone()


def test_duplicate_operation_returns_same_receipt_without_reactivation(env):
    active = service(env)
    args = request(env)
    first = active.enable(**args)
    second = active.enable(**args)
    assert first.policy == second.policy
    assert second.created is False
    with sqlite3.connect(env.database) as db:
        assert db.execute("SELECT COUNT(*) FROM continuous_policy_history").fetchone()[0] == 1
    other = new_profile(env)
    env.profiles.activate(other.profile_id, env.secrets)
    with pytest.raises(RuntimeError):
        active.enable(**args)
    assert env.profiles.active_identity() == (other.profile_id, 1)


def test_duplicate_operation_payload_and_superseded_revoke_rejected(env):
    active = service(env)
    args = request(env)
    active.enable(**args)
    with pytest.raises(RuntimeError, match="payload_changed"):
        active.enable(**{**args, "backlog_mode": "include"})
    active.transition(
        action="revoke", expected_policy_revision=1, operation_id=str(uuid4()), user_confirmed=True
    )
    with pytest.raises(RuntimeError, match="superseded"):
        active.enable(**args)
    assert active.current().state == "revoked"


@pytest.mark.parametrize(
    "change", ["revision", "capability", "version", "missing_key", "unreadable_key"]
)
def test_model_preconditions_fail_closed_and_preserve_old_active(env, change):
    other = new_profile(
        env,
        cloud=change in {"missing_key", "unreadable_key"},
        key="test-only-key" if change == "unreadable_key" else None,
    )
    if change == "revision":
        env.profiles.update(
            other.profile_id, ModelProfilePatch(expected_revision=1, model="changed"), env.secrets
        )
    elif change in {"capability", "version"}:
        with sqlite3.connect(env.database) as db:
            if change == "capability":
                db.execute(
                    "UPDATE model_profiles SET model_available=0 WHERE profile_id=?",
                    (other.profile_id,),
                )
            else:
                db.execute(
                    "UPDATE model_profile_versions SET model='tampered' WHERE profile_id=?",
                    (other.profile_id,),
                )
    elif change == "unreadable_key":
        env.secrets.delete(env.profiles.secret_ref(other.profile_id))
    with pytest.raises(RuntimeError):
        service(env).enable(**request(env, profile_id=other.profile_id))
    assert env.profiles.active_identity() == (env.profile.profile_id, 1)
    assert service(env).current() is None


def test_stale_legacy_revision_rejects_switch_and_old_revoked_grant_is_not_modified(env):
    old = env.grant()
    revoked = env.permissions.revoke(
        user_id="local", expected_revision=old.revision, user_confirmed=True
    )
    other = new_profile(env)
    with pytest.raises(RuntimeError, match="policy_revision_changed"):
        service(env).enable(
            **request(env, profile_id=other.profile_id, expected_legacy_revision=old.revision)
        )
    assert env.profiles.active_identity() == (env.profile.profile_id, 1)
    result = service(env).enable(
        **request(env, profile_id=other.profile_id, expected_legacy_revision=revoked.revision)
    )
    assert result.policy.legacy_revision == revoked.revision
    assert env.permissions.receipt(user_id="local") == revoked


def test_pause_resume_revoke_cas_and_resume_does_not_switch_model(env):
    active = service(env)
    active.enable(**request(env))
    op = str(uuid4())
    paused = active.transition(
        action="pause", expected_policy_revision=1, operation_id=op, user_confirmed=True
    )
    assert paused.policy.state == "paused"
    assert (
        active.transition(
            action="pause", expected_policy_revision=1, operation_id=op, user_confirmed=True
        ).created
        is False
    )
    other = new_profile(env)
    env.profiles.activate(other.profile_id, env.secrets)
    with pytest.raises(RuntimeError):
        active.transition(
            action="resume",
            expected_policy_revision=2,
            operation_id=str(uuid4()),
            user_confirmed=True,
        )
    assert active.current().state == "paused"
    active.transition(
        action="revoke", expected_policy_revision=2, operation_id=str(uuid4()), user_confirmed=True
    )
    with pytest.raises(RuntimeError, match="requires_new_enable"):
        active.transition(
            action="resume",
            expected_policy_revision=3,
            operation_id=str(uuid4()),
            user_confirmed=True,
        )


def test_same_operation_concurrent_submission_commits_once(env):
    args = request(env)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: service(env).enable(**args), range(2)))
    assert sum(result.created for result in results) == 1
    assert results[0].policy == results[1].policy


def test_no_sources_can_enable_future_and_backlog_is_explicit(env):
    with sqlite3.connect(env.database) as db:
        db.execute("DELETE FROM evidence")
        db.execute("DELETE FROM source_grants")
    result = service(env).enable(**request(env))
    assert result.policy.after_seq == 0
    assert result.policy.backlog_events == ()
    assert result.policy.state == "enabled"


def test_pre_d57_none_policy_is_upgraded_once_to_compact_backlog(env):
    activation = service(env)
    old = activation.enable(**request(env)).policy
    old = old.model_copy(update={"changed_at": "2026-10-06T12:00:00+00:00"})
    with sqlite3.connect(env.database) as db:
        db.execute(
            "UPDATE continuous_policy_history SET receipt_json=? WHERE user_id=? AND revision=?",
            (old.model_dump_json(), old.user_id, old.revision),
        )

    upgraded = activation.upgrade_legacy_none_policy()

    assert upgraded is not None and upgraded.created
    assert upgraded.policy.backlog_mode == "include"
    assert upgraded.policy.automatic_backlog_upgrade is True
    assert upgraded.policy.backlog_count == 1
    assert upgraded.policy.backlog_conversation_count == 1
    assert activation.upgrade_legacy_none_policy() is None


@pytest.mark.parametrize(
    ("failure", "error_code"),
    [
        (
            SecretStoreUnavailableError("Operating system secure storage read backoff"),
            "secure_storage_unavailable",
        ),
        (RuntimeError("unexpected startup failure"), "legacy_backlog_upgrade_failed"),
    ],
)
def test_legacy_policy_upgrade_defers_startup_failures(env, monkeypatch, failure, error_code):
    activation = service(env)
    old = activation.enable(**request(env)).policy.model_copy(
        update={"changed_at": "2026-10-06T12:00:00+00:00"}
    )
    with sqlite3.connect(env.database) as db:
        db.execute(
            "UPDATE continuous_policy_history SET receipt_json=? WHERE user_id=? AND revision=?",
            (old.model_dump_json(), old.user_id, old.revision),
        )

    events = []
    monkeypatch.setattr(activation, "enable", lambda **_kwargs: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(
        "one_memory.model_policy_activation.diagnostic_event",
        lambda **event: events.append(event),
    )

    assert activation.upgrade_legacy_none_policy() is None
    assert len(events) == 1
    assert events[0]["level"] == "warn"
    assert events[0]["event"] == "policy.legacy_backlog_upgrade_deferred"
    assert events[0]["error_code"] == error_code
    assert activation.current() == old


def test_post_d57_explicit_none_policy_is_not_upgraded(env):
    activation = service(env)
    current = activation.enable(**request(env)).policy
    assert current.backlog_mode == "none"
    assert activation.upgrade_legacy_none_policy() is None
    assert activation.current() == current


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_confirmed", False),
        ("user_confirmed", 1),
        ("backlog_mode", "all"),
        ("profile_revision", True),
        ("operation_id", "invalid"),
    ],
)
def test_invalid_input_never_initializes_or_switches(env, field, value):
    with pytest.raises((RuntimeError, ValueError)):
        service(env).enable(**request(env, **{field: value}))
    assert service(env).current() is None


def test_cloud_credential_is_checked_without_keychain_write(env, monkeypatch):
    other = new_profile(env, cloud=True, key="fixture-key")
    monkeypatch.setattr(env.secrets, "put", lambda *args: pytest.fail("no credential write"))
    monkeypatch.setattr(env.secrets, "delete", lambda *args: pytest.fail("no credential delete"))
    receipt = service(env).enable(**request(env, profile_id=other.profile_id))
    assert receipt.policy.profile.profile_id == other.profile_id


def test_profile_change_after_secret_preflight_does_not_commit_policy(env, monkeypatch):
    other = new_profile(env, cloud=True, key="fixture-key")
    original_has = env.secrets.has

    def changed(secret_ref):
        answer = original_has(secret_ref)
        with sqlite3.connect(env.database) as db:
            db.execute(
                "UPDATE model_profiles SET model_available=0 WHERE profile_id=?",
                (other.profile_id,),
            )
        return answer

    monkeypatch.setattr(env.secrets, "has", changed)
    with pytest.raises(RuntimeError, match="probe_required"):
        service(env).enable(**request(env, profile_id=other.profile_id))
    assert env.profiles.active_identity() == (env.profile.profile_id, 1)
    assert service(env).current() is None


def test_same_operation_does_not_switch_or_accept_lost_probe_readiness(env):
    active = service(env)
    args = request(env)
    active.enable(**args)
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE model_profiles SET embedding_available=0")
    with pytest.raises(RuntimeError, match="probe_required"):
        active.enable(**args)
    assert active.current().revision == 1


def test_two_different_operations_same_cas_only_one_commits(env):
    requests = [request(env), request(env)]

    def submit(value):
        try:
            return service(env).enable(**value).created
        except policy.ContinuousPolicyError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(submit, requests)) == 1
    assert service(env).current().revision == 1


def test_backlog_scan_does_not_hold_the_single_writer(env, monkeypatch):
    active = service(env)
    entered = threading.Event()
    release = threading.Event()
    original = policy.snapshot_backlog

    def paused_snapshot(db, *, user_id="local"):
        snapshot = original(db, user_id=user_id)
        entered.set()
        assert release.wait(3)
        return snapshot

    monkeypatch.setattr(policy, "snapshot_backlog", paused_snapshot)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(active.enable, **request(env, backlog_mode="include"))
        if not entered.wait(3):
            release.set()
            pending.result(timeout=3)
            pytest.fail("backlog snapshot was not reached")
        writer = sqlite3.connect(env.database, timeout=0.2, isolation_level=None)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("ROLLBACK")
        finally:
            writer.close()
            release.set()
        assert pending.result(timeout=3).created is True


def test_activation_helper_is_transactional_and_restores_row_factory(env, monkeypatch):
    from one_memory.model_content_permissions import ModelContentPermissionStore
    from one_memory.model_profiles import ModelProfileStore

    other = new_profile(env)
    snapshot = env.profiles.snapshot(other.profile_id, 1)
    with sqlite3.connect(env.database) as db:
        with pytest.raises(RuntimeError, match="caller_transaction_required"):
            ModelProfileStore.activate_in_connection(db, snapshot, expected_secret_ref=None)
        db.execute("BEGIN IMMEDIATE")
        monkeypatch.setattr(
            ModelContentPermissionStore,
            "_validate_profile",
            lambda *args: (_ for _ in ()).throw(RuntimeError("recipient rejected")),
        )
        with pytest.raises(RuntimeError, match="recipient rejected"):
            ModelProfileStore.activate_in_connection(db, snapshot, expected_secret_ref=None)
        assert db.row_factory is None
        assert (
            db.execute("SELECT profile_id FROM model_profiles WHERE is_active=1").fetchone()[0]
            == env.profile.profile_id
        )
        assert db.in_transaction


def test_missing_database_get_does_not_create_it(tmp_path, env):
    database = tmp_path / "must-not-be-created.sqlite3"
    with pytest.raises(sqlite3.Error):
        ModelPolicyActivation(database, env.secrets).current()
    assert not database.exists()
