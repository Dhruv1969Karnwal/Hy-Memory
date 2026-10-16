import json
import sqlite3
from contextlib import closing, contextmanager
from uuid import uuid4

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from one_memory.model_policy_activation import ModelPolicyActivation
from one_memory.model_policy_routes import continuous_policy_router, safe_policy
from tests.test_model_content_permissions import env as env  # noqa: F401
from tests.test_model_policy_activation import request

ROOT = "/api/v1/continuous-processing-policy/operations/"


def client(env, callback=None):
    app = FastAPI()
    app.state.model_policy_activation = ModelPolicyActivation(env.database, env.secrets)

    def auth(authorization: str | None = Header(None)):
        if authorization != "Bearer fixture":
            raise HTTPException(401, "unauthorized")

    app.include_router(continuous_policy_router([Depends(auth)], on_enabled=callback))
    return TestClient(app, headers={"Authorization": "Bearer fixture"})


def activate(env):
    return ModelPolicyActivation(env.database, env.secrets).enable(**request(env)).policy


def database_dump(database):
    # WAL checkpoints can change the main file bytes on closing a read handle.
    # Compare actual schema/data instead, keeping all rows rather than just counts.
    with closing(sqlite3.connect(database)) as db:
        return tuple(db.iterdump())


def test_missing_table_returns_not_found_without_initializing_or_reading_secrets(env, monkeypatch):
    def forbidden(*args):
        pytest.fail("lookup must not read or mutate secrets")

    for name in ("has", "resolve", "put", "delete"):
        monkeypatch.setattr(env.secrets, name, forbidden)
    before = database_dump(env.database)
    with client(env) as http:
        result = http.get(ROOT + str(uuid4()))
        assert result.status_code == 200, result.text
        assert result.json() == {"state": "not_found", "receipt": None, "current_revision": 0}
    assert database_dump(env.database) == before
    with sqlite3.connect(env.database) as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='continuous_policy_history'"
        ).fetchone()


def test_current_operation_uses_safe_receipt_without_callback_or_writes(env, monkeypatch):
    receipt = activate(env)

    async def forbidden_callback(_receipt):
        pytest.fail("GET cannot invoke activation callback")

    monkeypatch.setattr(env.secrets, "has", lambda *args: pytest.fail("no credential read"))
    before = database_dump(env.database)
    with client(env, forbidden_callback) as http:
        result = http.get(ROOT + receipt.operation_id)
        assert result.status_code == 200, result.text
        assert result.json() == {
            "state": "current",
            "receipt": safe_policy(receipt).model_dump(),
            "current_revision": 1,
        }
        assert not {"profile", "secret_ref", "backlog_events"} & result.json()["receipt"].keys()
    assert database_dump(env.database) == before


def test_superseded_operation_preserves_historical_receipt_not_current_activation(env):
    first = activate(env)
    latest = (
        ModelPolicyActivation(env.database, env.secrets)
        .transition(
            operation_id=str(uuid4()),
            expected_policy_revision=1,
            action="revoke",
            user_confirmed=True,
        )
        .policy
    )
    with client(env) as http:
        historical = http.get(ROOT + first.operation_id)
        assert historical.status_code == 200, historical.text
        assert historical.json() == {
            "state": "superseded",
            "receipt": safe_policy(first).model_dump(),
            "current_revision": latest.revision,
        }
        result = http.get(ROOT + latest.operation_id).json()
        assert result["state"] == "current" and result["receipt"]["state"] == "revoked"
        assert http.get(ROOT + str(uuid4())).json() == {
            "state": "not_found",
            "receipt": None,
            "current_revision": latest.revision,
        }
    assert ModelPolicyActivation(env.database, env.secrets).current() == latest


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
        "a" * 32,
        "ABCDEFAB-CDEF-ABCD-EFAB-CDEFABCDEFAB",
        "{abcdefab-cdef-abcd-efab-cdefabcdefab}",
    ],
)
def test_noncanonical_uuid_is_422(env, value):
    with client(env) as http:
        result = http.get(ROOT + value)
        assert result.status_code == 422, result.text


def test_lookup_requires_authentication(env):
    receipt = activate(env)
    with client(env) as http:
        result = http.get(ROOT + receipt.operation_id, headers={"Authorization": "bad"})
        assert result.status_code == 401
        assert receipt.profile.profile_id not in result.text


def test_other_user_operation_is_not_disclosed(env):
    own = activate(env)
    other_op = str(uuid4())
    with sqlite3.connect(env.database) as db:
        data = {**own.model_dump(mode="json"), "user_id": "other", "operation_id": other_op}
        db.execute(
            "INSERT INTO continuous_policy_history VALUES(?,?,?,?,?)",
            ("other", 1, other_op, "a" * 64, json.dumps(data)),
        )
    with client(env) as http:
        result = http.get(ROOT + other_op)
        assert result.status_code == 200
        assert result.json() == {"state": "not_found", "receipt": None, "current_revision": 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", 99),
        ("revision", "1"),
        ("operation_id", "different"),
        ("user_id", "other"),
        ("secret_ref", "must-not-leak"),
    ],
)
def test_row_receipt_mismatch_or_extra_fields_returns_safe_503(env, field, value):
    receipt = activate(env)
    with sqlite3.connect(env.database) as db:
        data = {**receipt.model_dump(mode="json"), field: value}
        db.execute("UPDATE continuous_policy_history SET receipt_json=?", (json.dumps(data),))
    with client(env) as http:
        result = http.get(ROOT + receipt.operation_id)
        assert result.status_code == 503, result.text
        assert "must-not-leak" not in result.text
        assert "receipt" not in result.json()


def test_corrupt_current_row_cannot_make_history_or_missing_operation_look_safe(env):
    first = activate(env)
    ModelPolicyActivation(env.database, env.secrets).transition(
        operation_id=str(uuid4()), expected_policy_revision=1, action="pause", user_confirmed=True
    )
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE continuous_policy_history SET receipt_json='broken' WHERE revision=2")
    with client(env) as http:
        assert http.get(ROOT + first.operation_id).status_code == 503
        assert http.get(ROOT + str(uuid4())).status_code == 503


def test_sql_failure_is_503_not_not_found(env):
    activate(env)
    with sqlite3.connect(env.database) as db:
        db.execute("ALTER TABLE continuous_policy_history RENAME COLUMN receipt_json TO broken")
    with client(env) as http:
        result = http.get(ROOT + str(uuid4()))
        assert result.status_code == 503
        assert "broken" not in result.text


def test_missing_database_is_503_and_is_not_created(env):
    missing = env.database.parent / "missing-policy.sqlite3"
    with client(env) as http:
        http.app.state.model_policy_activation = ModelPolicyActivation(missing, env.secrets)
        response = http.get(ROOT + str(uuid4()))
        assert response.status_code == 503
    assert not missing.exists()


def test_lookup_connection_rejects_writes_and_preserves_database(env, monkeypatch):
    receipt = activate(env)
    service = ModelPolicyActivation(env.database, env.secrets)
    exists = service._exists
    rejected = []

    def check_read_only(db):
        assert db.in_transaction
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM continuous_policy_history")
        rejected.append(True)
        return exists(db)

    monkeypatch.setattr(service, "_exists", check_read_only)
    assert service.lookup_operation(receipt.operation_id).receipt == receipt
    assert rejected == [True]
    assert ModelPolicyActivation(env.database, env.secrets).current() == receipt


def test_lookup_reads_operation_and_current_in_the_same_snapshot(env, monkeypatch):
    receipt = activate(env)
    service = ModelPolicyActivation(env.database, env.secrets)
    connect = service._connect
    writes = []

    @contextmanager
    def observing_connect(*, write=False):
        assert not write

        def trace(sql):
            if "AND operation_id=" in sql and not writes:
                writes.append("concurrent pause")
                ModelPolicyActivation(env.database, env.secrets).transition(
                    operation_id=str(uuid4()),
                    expected_policy_revision=1,
                    action="pause",
                    user_confirmed=True,
                )

        with connect() as db:
            db.set_trace_callback(trace)
            yield db

    monkeypatch.setattr(service, "_connect", observing_connect)
    result = service.lookup_operation(receipt.operation_id)
    assert writes == ["concurrent pause"]
    assert result.state == "current" and result.current_revision == 1
    assert result.receipt == receipt
    assert ModelPolicyActivation(env.database, env.secrets).current().revision == 2
