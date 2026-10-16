from types import SimpleNamespace

import pytest

from one_memory.hy_memory_bridge import HyMemoryBridgeError
from one_memory.memory_server.config_state import ConfigStore
from one_memory.model_processing_permission_guard import ModelProcessingPermissionGuard
from one_memory.model_profiles import ModelProfileSnapshot, ProviderKind


def snapshot(profile_id: str = "profile") -> ModelProfileSnapshot:
    return ModelProfileSnapshot(
        profile_id=profile_id,
        revision=1,
        provider_kind=ProviderKind.OLLAMA,
        base_url="http://127.0.0.1:11434",
        model="test-model",
        embedding_model="test-embedding",
        embedding_provider_kind=ProviderKind.OLLAMA,
        embedding_base_url="http://127.0.0.1:11434",
        timeout_seconds=30,
        max_concurrency=1,
        context_window=None,
        has_api_key=False,
    )


def test_lease_reads_current_allowed_before_each_boundary(tmp_path) -> None:
    config = ConfigStore(tmp_path)
    profile = snapshot()
    profiles = SimpleNamespace(active_identity=lambda: (profile.profile_id, profile.revision))
    guard = ModelProcessingPermissionGuard(config, SimpleNamespace(), profiles)
    task = SimpleNamespace(
        task_id="task",
        user_id="local",
        profile_snapshot=profile,
    )

    with pytest.raises(HyMemoryBridgeError, match="model_content_permission_required"):
        guard.lease(task, (), None)

    config.set_model_content_allowed(True)
    lease = guard.lease(task, (), None)
    lease.recheck()

    config.set_model_content_allowed(False)
    with pytest.raises(HyMemoryBridgeError, match="model_content_permission_required"):
        lease.recheck()


def test_lease_refuses_a_task_profile_that_is_no_longer_active(tmp_path) -> None:
    config = ConfigStore(tmp_path)
    config.set_model_content_allowed(True)
    profile = snapshot()
    profiles = SimpleNamespace(active_identity=lambda: ("new-profile", 1))
    guard = ModelProcessingPermissionGuard(config, SimpleNamespace(), profiles)

    with pytest.raises(HyMemoryBridgeError, match="model_content_profile_changed"):
        guard.check(profile)
