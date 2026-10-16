import json

import pytest

from one_memory.memory_server.config_state import ConfigFileError, ConfigStore


def test_sync_config_defaults_off_and_survives_restart(tmp_path):
    store = ConfigStore(tmp_path)
    assert not store.sync_enabled("cursor")

    store.set_sync("cursor", True)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved == {
        "model_content_allowed": False,
        "schema_version": 1,
        "sources": {"cursor": {"sync_enabled": True}},
    }
    assert ConfigStore(tmp_path).sync_enabled("cursor")


def test_sync_config_updates_one_source_without_losing_another(tmp_path):
    store = ConfigStore(tmp_path)
    store.set_sync("cursor", True)
    store.set_sync("claude-code", True)
    store.set_sync("cursor", False)

    reopened = ConfigStore(tmp_path)
    assert not reopened.sync_enabled("cursor")
    assert reopened.sync_enabled("claude-code")


def test_model_content_permission_is_one_global_final_state(tmp_path):
    store = ConfigStore(tmp_path)
    assert not store.model_content_allowed()

    store.set_model_content_allowed(True)
    store.set_sync("cursor", True)
    reopened = ConfigStore(tmp_path)
    assert reopened.model_content_allowed()
    assert reopened.sync_enabled("cursor")

    reopened.set_model_content_allowed(False)
    assert not ConfigStore(tmp_path).model_content_allowed()
    assert ConfigStore(tmp_path).sync_enabled("cursor")


def test_invalid_config_fails_loudly_instead_of_silently_disabling_sources(tmp_path):
    (tmp_path / "config.json").write_text('{"schema_version":1,"sources":{"cursor":{}}')
    with pytest.raises(ConfigFileError, match="config_invalid"):
        ConfigStore(tmp_path)
