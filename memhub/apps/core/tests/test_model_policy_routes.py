import sqlite3
from uuid import uuid4

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from one_memory.model_policy_activation import ModelPolicyActivation
from one_memory.model_policy_routes import continuous_policy_router
from tests.test_model_content_permissions import env as env  # noqa: F401
from tests.test_model_policy_activation import request

PREFIX = "/api/v1/continuous-processing-policy"
SAFE_KEYS = {
    "schema_version",
    "user_id",
    "revision",
    "epoch",
    "state",
    "after_seq",
    "profile_id",
    "profile_revision",
    "legacy_revision",
    "backlog_mode",
    "backlog_count",
    "operation_id",
    "changed_at",
}


def client(env, *, enabled=False, callback=None):
    app = FastAPI()
    app.state.model_policy_activation = ModelPolicyActivation(env.database, env.secrets)

    def auth(authorization: str | None = Header(None)):
        if authorization != "Bearer fixture":
            raise HTTPException(401, "unauthorized")

    app.include_router(
        continuous_policy_router([Depends(auth)], on_enabled=callback, enabled=enabled)
    )
    return TestClient(app, headers={"Authorization": "Bearer fixture"})


def test_get_does_not_initialize_and_capability_defaults_false(env):
    with client(env) as http:
        data = http.get(PREFIX).json()
        assert data == {
            "available": True,
            "policy": None,
            "legacy_revision": 0,
            "capability": {"enable": False, "reason": "processing_guards_not_integrated"},
            "backlog_upgrade_notice": {
                "pending": False,
                "epoch": None,
                "conversation_count": 0,
                "profile_id": None,
            },
        }
        assert http.post(PREFIX + "/enable", json=request(env)).status_code == 409
    with sqlite3.connect(env.database) as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='continuous_policy_history'"
        ).fetchone()


@pytest.mark.parametrize("route", ["", "/enable", "/pause", "/resume", "/revoke"])
def test_all_routes_require_auth(env, route):
    with client(env, enabled=True) as http:
        response = (
            http.get(PREFIX, headers={"Authorization": "bad"})
            if not route
            else http.post(PREFIX + route, json=request(env), headers={"Authorization": "bad"})
        )
        assert response.status_code == 401


def test_factory_refuses_no_auth_dependency():
    with pytest.raises(ValueError, match="authentication_required"):
        continuous_policy_router([])


@pytest.mark.parametrize(
    "mutation",
    [
        {"backlog_mode": None},
        {"user_confirmed": "true"},
        {"profile_revision": True},
        {"profile_revision": 1.0},
        {"operation_id": "not-uuid"},
        {"api_key": "forbidden"},
    ],
)
def test_request_validation_does_not_coerce_and_rejects_secret_fields(env, mutation):
    with client(env, enabled=True) as http:
        value = {**request(env), **mutation}
        assert http.post(PREFIX + "/enable", json=value).status_code == 422
        assert http.get(PREFIX).json()["policy"] is None


def test_backlog_is_required_not_defaulted(env):
    with client(env, enabled=True) as http:
        value = request(env)
        del value["backlog_mode"]
        assert http.post(PREFIX + "/enable", json=value).status_code == 422


def test_enable_and_transition_safe_shape_and_callback_once_after_commit(env):
    calls = []

    async def notified(receipt):
        assert ModelPolicyActivation(env.database, env.secrets).current() == receipt
        calls.append(receipt.revision)

    with client(env, enabled=True, callback=notified) as http:
        value = request(env, backlog_mode="include")
        first = http.post(PREFIX + "/enable", json=value)
        assert first.status_code == 200, first.text
        assert set(first.json()) == SAFE_KEYS
        assert first.json()["backlog_count"] == 1
        assert "secret_ref" not in first.text and '"profile":' not in first.text
        assert http.post(PREFIX + "/enable", json=value).json() == first.json()
        assert calls == [1]
        for revision, action in [(1, "pause"), (2, "resume"), (3, "revoke")]:
            result = http.post(
                PREFIX + "/" + action,
                json={
                    "operation_id": str(uuid4()),
                    "expected_policy_revision": revision,
                    "user_confirmed": True,
                },
            )
            assert result.status_code == 200, result.text
            assert set(result.json()) == SAFE_KEYS
        assert calls == [1, 3]
        assert http.get(PREFIX).json()["policy"]["state"] == "revoked"


def test_status_exposes_only_automatic_backlog_upgrade_notice(env):
    activation = service = ModelPolicyActivation(env.database, env.secrets)
    old = activation.enable(**request(env)).policy
    old = old.model_copy(update={"changed_at": "2026-10-06T12:00:00+00:00"})
    with sqlite3.connect(env.database) as db:
        db.execute(
            "UPDATE continuous_policy_history SET receipt_json=? WHERE user_id=? AND revision=?",
            (old.model_dump_json(), old.user_id, old.revision),
        )
    upgraded = activation.upgrade_legacy_none_policy()
    assert upgraded is not None

    app = FastAPI()
    app.state.model_policy_activation = service

    def auth():
        return None

    app.include_router(continuous_policy_router([Depends(auth)], enabled=True))
    with TestClient(app) as http:
        notice = http.get(PREFIX).json()["backlog_upgrade_notice"]
    assert notice == {
        "pending": True,
        "epoch": upgraded.policy.epoch,
        "conversation_count": 1,
        "profile_id": env.profile.profile_id,
    }


def test_notification_failure_does_not_rollback_or_repeat_enable(env):
    calls = []

    async def fail_notify(receipt):
        calls.append(receipt.revision)
        raise RuntimeError("private runtime failure")

    with client(env, enabled=True, callback=fail_notify) as http:
        value = request(env)
        assert http.post(PREFIX + "/enable", json=value).status_code == 200
        assert http.post(PREFIX + "/enable", json=value).status_code == 200
        assert calls == [1]
        assert http.get(PREFIX).json()["policy"]["state"] == "enabled"


def test_gate_off_allows_revoke_but_not_resume(env):
    ModelPolicyActivation(env.database, env.secrets).enable(**request(env))
    with client(env) as http:
        value = {
            "operation_id": str(uuid4()),
            "expected_policy_revision": 1,
            "user_confirmed": True,
        }
        assert http.post(PREFIX + "/pause", json=value).status_code == 200
        value.update(operation_id=str(uuid4()), expected_policy_revision=2)
        assert http.post(PREFIX + "/resume", json=value).status_code == 409
        assert http.post(PREFIX + "/revoke", json=value).status_code == 200


def test_store_failure_has_no_raw_errors_or_fake_empty_policy(env, monkeypatch):
    monkeypatch.setattr(
        ModelPolicyActivation,
        "read_current",
        lambda self: (_ for _ in ()).throw(sqlite3.OperationalError("secret raw error")),
    )
    with client(env) as http:
        result = http.get(PREFIX)
        assert result.status_code == 503
        assert "secret" not in result.text and '"policy":null' not in result.text
