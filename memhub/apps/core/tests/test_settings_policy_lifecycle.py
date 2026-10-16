"""Settings data-plane lifecycle: real temporary Core, fake probe/HY, no UI or package."""

import asyncio
import sqlite3
from contextlib import closing
from uuid import uuid4

import httpx
import pytest

from one_memory.continuous_permission_bridge import ContinuousPermissionReceipt
from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.memory_tasks import MemoryTaskStatus
from one_memory.model_probe import ModelProber
from tests.test_continuous_processing_integration import API_ROOT
from tests.test_continuous_processing_integration import core as core  # noqa: F401


async def save(core, name="model-a", *, probe=True):
    response = await core.client.post(
        "/api/v1/model-profiles",
        json={
            "name": name,
            "provider_kind": "openai_compatible",
            "base_url": f"https://{name}.invalid/v1",
            "model": name,
            "api_key": "synthetic-key-not-a-credential",
            "embedding_provider_kind": "openai_compatible",
            "embedding_base_url": f"https://{name}.invalid/v1",
            "embedding_model": "integration-vector",
            "disable_thinking": True,
            "max_concurrency": 1,
        },
    )
    assert response.status_code == 200, response.text
    profile = response.json()
    assert profile["is_active"] is False
    return await probe_profile(core, profile) if probe else profile


async def probe_profile(core, profile):
    # Extend the fixture's synthetic model list for two genuinely distinct models;
    # all other probe stages keep the existing controlled boundary and payload checks.
    def respond(request):
        if request.url.path.endswith("/models"):
            core.probe_requests.append((request.url.path, ""))
            return httpx.Response(200, json={"data": [{"id": profile["model"]}]})
        return core.probe_transport(request)

    core.app.state.model_prober = ModelProber(
        store=core.app.state.model_profile_store,
        secrets=core.secrets,
        transport=httpx.MockTransport(respond),
    )
    result = await core.client.post(f"/api/v1/model-profiles/{profile['profile_id']}/probe")
    assert result.status_code == 200, result.text
    assert result.json()["input_profile_revision"] == profile["revision"]
    profiles = await core.client.get("/api/v1/model-profiles")
    assert profiles.status_code == 200
    measured = next(p for p in profiles.json() if p["profile_id"] == profile["profile_id"])
    assert measured["revision"] == result.json()["profile_revision"]
    assert measured["embedding_dimensions"] == 3
    assert all(
        measured["capabilities"][key]
        for key in ("endpoint_reachable", "model_available", "embedding_available")
    )
    return measured


async def current(core):
    response = await core.client.get(API_ROOT)
    assert response.status_code == 200, response.text
    return response.json()


async def enable(core, profile, **overrides):
    before = await current(core)
    body = {
        "operation_id": str(uuid4()),
        "profile_id": profile["profile_id"],
        "profile_revision": profile["revision"],
        "expected_policy_revision": before["policy"]["revision"] if before["policy"] else 0,
        "expected_legacy_revision": before["legacy_revision"],
        "backlog_mode": "none",
        "user_confirmed": True,
        **overrides,
    }
    return await core.client.post(API_ROOT + "/enable", json=body), body


async def enabled(core, profile, **overrides):
    response, body = await enable(core, profile, **overrides)
    assert response.status_code == 200, response.text
    value = response.json()
    # One DB read snapshot observes policy and active model together.
    with closing(sqlite3.connect(core.database)) as db:
        db.execute("BEGIN")
        active = db.execute("SELECT profile_id,revision FROM model_profiles WHERE is_active=1")
        assert active.fetchall() == [(profile["profile_id"], profile["revision"])]
        saved = db.execute(
            "SELECT receipt_json FROM continuous_policy_history WHERE user_id='local' "
            "ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        assert saved is not None
        assert core.policy_current().profile.profile_id == value["profile_id"]
    assert value["profile_id"] == profile["profile_id"]
    assert value["profile_revision"] == profile["revision"]
    return value, body


def guard(core):
    result = core.app.state.memory_processing_supervisor._permission_guard
    assert result is not None
    return result


def check_current(core, record):
    authority = guard(core).current("local")
    assert isinstance(authority, ContinuousPermissionReceipt), "Must never fall back to legacy"
    guard(core).check(authority, authority.policy.profile, (record,))
    return authority


async def assert_processed_by(core, record, profile):
    tasks = await core.run_prepared(record)
    assert tasks and all(task.status == MemoryTaskStatus.COMPLETED for task in tasks)
    assert all(
        (task.profile_id, task.profile_revision) == (profile["profile_id"], profile["revision"])
        for task in tasks
    )
    assert core.bridge.requests and record.evidence_id in core.bridge.requests[-1].evidence_ids
    assert core.bridge.leases[-1].profile.profile_id == profile["profile_id"]
    await core.assert_governed(record)


async def legacy_grant(core, profile):
    active = await core.client.post(f"/api/v1/model-profiles/{profile['profile_id']}/activate")
    assert active.status_code == 200, active.text
    preview = await core.client.post(
        "/api/v1/model-content-permission/preview",
        json={
            "profile_id": profile["profile_id"],
            "profile_revision": profile["revision"],
            "source_ids": ["cursor"],
            "all_imported_from_selected_sources": True,
            "include_future_additions": True,
        },
    )
    assert preview.status_code == 200, preview.text
    result = await core.client.post(
        "/api/v1/model-content-permission",
        json={"snapshot": preview.json(), "expected_revision": 0, "user_confirmed": True},
    )
    assert result.status_code == 200, result.text
    return core.app.state.model_content_permissions.receipt(user_id="local")


def test_no_onboarding_or_legacy_permission_required_for_save_probe_enable(core):
    async def scenario():
        async with core.running():
            onboarding = (await core.client.get("/api/v1/onboarding")).json()
            assert onboarding["status"] != "completed"
            profile = await save(core)
            assert profile["revision"] == 2  # probe measured width from unspecified to 3
            assert (await current(core))["policy"] is None
            assert not core.bridge.requests and not core.app.state.memory_task_store.list()
            await enabled(core, profile)
            assert (await core.client.get("/api/v1/onboarding")).json() == onboarding
            assert core.app.state.model_content_permissions.receipt(user_id="local") is None
            core.grant_source()
            record = core.put_complete("settings-first", conversation="settings-first")
            check_current(core, record)
            await assert_processed_by(core, record, profile)

    asyncio.run(scenario())


def test_saved_unprobed_draft_does_not_activate_or_create_policy(core):
    async def scenario():
        async with core.running():
            core.grant_source()
            record = core.put("draft-only")
            profile = await save(core, probe=False)
            assert not profile["is_active"]
            assert core.app.state.model_profile_store.active_identity() is None
            assert (await current(core))["policy"] is None
            with pytest.raises(HyMemoryBridgeError, match="permission_required"):
                await asyncio.to_thread(
                    core.app.state.memory_processing_supervisor.prepare_conversation, record
                )
            assert not core.probe_requests and not core.bridge.requests
            assert core.app.state.memory_task_store.list() == []

    asyncio.run(scenario())


def test_second_saved_probed_profile_does_not_replace_enabled_model(core):
    async def scenario():
        async with core.running():
            first = await save(core)
            original, _ = await enabled(core, first)
            second = await save(core, "model-b")
            assert second["profile_id"] != first["profile_id"]
            assert (await current(core))["policy"] == original
            assert core.app.state.model_profile_store.active_identity() == (
                first["profile_id"],
                first["revision"],
            )
            assert not core.bridge.requests
            core.grant_source()
            record = core.put_complete("after-second-draft", conversation="second-draft")
            await assert_processed_by(core, record, first)

    asyncio.run(scenario())


def test_switch_distinct_profiles_atomically_binds_new_policy_and_rejects_old_authority(core):
    async def scenario():
        async with core.running():
            first = await save(core)
            old, old_body = await enabled(core, first)
            core.grant_source()
            before = core.put_complete("before-switch", conversation="old-model")
            old_authority = check_current(core, before)
            second = await save(core, "model-b")
            new, body = await enabled(core, second)
            assert new["revision"] == old["revision"] + 1 and new["epoch"] != old["epoch"]
            with pytest.raises(HyMemoryBridgeError):
                guard(core).check(old_authority, old_authority.policy.profile, (before,))
            repeat = await core.client.post(API_ROOT + "/enable", json=body)
            assert repeat.status_code == 200 and repeat.json() == new
            assert (await core.client.post(API_ROOT + "/enable", json=old_body)).status_code == 409
            fresh = core.put_complete("after-switch", conversation="new-model")
            await assert_processed_by(core, fresh, second)
            assert all(before.evidence_id not in req.evidence_ids for req in core.bridge.requests)

    asyncio.run(scenario())


def test_failed_policy_cas_leaves_previous_profile_and_policy_unchanged(core):
    async def scenario():
        async with core.running():
            first = await save(core)
            original, _ = await enabled(core, first)
            second = await save(core, "model-b")
            rejected, _ = await enable(core, second, expected_policy_revision=0)
            assert rejected.status_code == 409, rejected.text
            assert (await current(core))["policy"] == original
            assert core.app.state.model_profile_store.active_identity() == (
                first["profile_id"],
                first["revision"],
            )
            assert not core.bridge.requests

    asyncio.run(scenario())


def test_unprobed_new_model_cannot_replace_a_working_policy(core):
    async def scenario():
        async with core.running():
            first = await save(core)
            original, _ = await enabled(core, first)
            second = await save(core, "model-b", probe=False)
            rejected, _ = await enable(core, second)
            assert rejected.status_code == 409 and "probe_required" in rejected.text
            assert (await current(core))["policy"] == original
            assert core.app.state.model_profile_store.active_identity() == (
                first["profile_id"],
                first["revision"],
            )
            assert not core.bridge.requests

    asyncio.run(scenario())


def test_active_profile_edit_invalidates_guard_until_explicit_reenable(core):
    async def scenario():
        async with core.running():
            first = await save(core)
            original, _ = await enabled(core, first)
            core.grant_source()
            patch = await core.client.patch(
                f"/api/v1/model-profiles/{first['profile_id']}",
                json={"expected_revision": first["revision"], "model": "edited-model"},
            )
            assert patch.status_code == 200, patch.text
            edited = patch.json()
            assert not edited["is_active"] and edited["revision"] == first["revision"] + 1
            blocked = core.put_complete("after-edit", conversation="edited")
            with pytest.raises(HyMemoryBridgeError):
                check_current(core, blocked)
            edited = await probe_profile(core, edited)
            assert (await current(core))["policy"] == original
            with pytest.raises(HyMemoryBridgeError):
                check_current(core, blocked)
            assert not core.bridge.requests
            renewed, _ = await enabled(core, edited, backlog_mode="include")
            assert renewed["revision"] == original["revision"] + 1
            await core.wait_completed(timeout=4)
            await core.assert_governed(blocked)
            assert all(lease.profile.model == "edited-model" for lease in core.bridge.leases)

    asyncio.run(scenario())


def test_revoke_then_new_enable_changes_epoch_without_restoring_revoked_period(core):
    async def scenario():
        async with core.running():
            first = await save(core)
            original, _ = await enabled(core, first)
            revoked = await core.client.post(
                API_ROOT + "/revoke",
                json={
                    "operation_id": str(uuid4()),
                    "expected_policy_revision": original["revision"],
                    "user_confirmed": True,
                },
            )
            assert revoked.status_code == 200, revoked.text
            denied = await core.client.post(
                API_ROOT + "/resume",
                json={
                    "operation_id": str(uuid4()),
                    "expected_policy_revision": revoked.json()["revision"],
                    "user_confirmed": True,
                },
            )
            assert denied.status_code == 409 and "requires_new_enable" in denied.text
            core.grant_source()
            excluded = core.put("during-revoke", conversation="revoked-period")
            with pytest.raises(HyMemoryBridgeError):
                check_current(core, excluded)
            second = await save(core, "model-b")
            renewed, _ = await enabled(core, second)
            assert renewed["epoch"] != original["epoch"]
            with pytest.raises(HyMemoryBridgeError, match="outside_policy"):
                check_current(core, excluded)
            fresh = core.put_complete("after-reenable", conversation="renewed")
            await assert_processed_by(core, fresh, second)
            assert all(excluded.evidence_id not in req.evidence_ids for req in core.bridge.requests)

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["delete", "clear_key"])
def test_unavailable_policy_profile_never_falls_back_to_real_legacy_grant(core, change):
    async def scenario():
        async with core.running():
            core.grant_source()
            first = await save(core)
            legacy = await legacy_grant(core, first)
            original, _ = await enabled(core, first)
            assert original["legacy_revision"] == legacy.revision
            initial = core.put("positive-authority", conversation="before-unavailable")
            authority = check_current(core, initial)
            patch = await core.client.patch(
                f"/api/v1/model-profiles/{first['profile_id']}",
                json={"expected_revision": first["revision"], "clear_api_key": True},
            )
            assert patch.status_code == 200 and not patch.json()["is_active"]
            if change == "delete":
                deleted = await core.client.delete(f"/api/v1/model-profiles/{first['profile_id']}")
                assert deleted.status_code == 204, deleted.text
                assert (
                    await core.client.post(f"/api/v1/model-profiles/{first['profile_id']}/probe")
                ).status_code == 404
            else:
                missing = await core.client.post(
                    f"/api/v1/model-profiles/{first['profile_id']}/probe"
                )
                assert missing.status_code == 200
                assert missing.json()["stages"][0]["code"] == "missing_api_key"
            assert core.app.state.model_content_permissions.receipt(user_id="local") == legacy
            assert core.policy_current().epoch == original["epoch"]
            next_record = core.put("after-unavailable", conversation="after-unavailable")
            with pytest.raises(HyMemoryBridgeError):
                guard(core).check(authority, authority.policy.profile, (next_record,))
            with pytest.raises(HyMemoryBridgeError):
                await asyncio.to_thread(
                    core.app.state.memory_processing_supervisor.prepare_conversation, next_record
                )
            assert not core.bridge.requests
            assert core.app.state.memory_governance_store.list(user_id="local") == []

    asyncio.run(scenario())
