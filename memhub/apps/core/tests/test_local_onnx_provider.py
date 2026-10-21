"""Expect local ONNX embedding to be the default and to feed HY Memory.

HY's EmbedService only speaks OpenAI-compatible HTTP and offers no injection
point. After the client is built we replace `_embed_service` with a thin
adapter around LocalEmbedder — in every place that holds one, including the
pipelines HY pre-warms during construction.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from one_memory.embedding_assets import (
    BUILTIN_EMBEDDING_DIMENSIONS,
    BUILTIN_EMBEDDING_MODEL,
)
from one_memory.hy_memory_bridge import (
    LocalHyEmbedService,
    build_hy_memory_config,
    create_hy_memory_client,
)
from one_memory.local_embedding import LocalEmbedder
from one_memory.model_profiles import (
    ModelProfileInput,
    ModelProfileStore,
    ProviderKind,
    apply_builtin_embedding_defaults,
)
from one_memory.secrets import InMemorySecretStore
from tests.test_local_embedding import requires_assets


def test_default_local_hy_embed_services_share_one_encoder() -> None:
    first = LocalHyEmbedService()
    second = LocalHyEmbedService()

    assert first._embedder is second._embedder  # noqa: SLF001


def test_provider_kind_includes_local_onnx() -> None:
    assert ProviderKind.LOCAL_ONNX.value == "local_onnx"


def test_missing_embedding_defaults_to_the_builtin_model() -> None:
    provider, model, base_url, dimensions = apply_builtin_embedding_defaults(
        embedding_model=None,
        embedding_provider_kind=None,
        embedding_base_url=None,
        embedding_dimensions=None,
    )
    assert provider is ProviderKind.LOCAL_ONNX
    assert model == BUILTIN_EMBEDDING_MODEL
    assert base_url is None
    assert dimensions == BUILTIN_EMBEDDING_DIMENSIONS


def test_an_explicit_embedding_is_left_alone() -> None:
    provider, model, base_url, dimensions = apply_builtin_embedding_defaults(
        embedding_model="text-embedding-3-small",
        embedding_provider_kind=ProviderKind.OPENAI_COMPATIBLE,
        embedding_base_url="https://api.openai.com/v1",
        embedding_dimensions=1536,
    )
    assert provider is ProviderKind.OPENAI_COMPATIBLE
    assert model == "text-embedding-3-small"
    assert base_url == "https://api.openai.com/v1"
    assert dimensions == 1536


def test_creating_a_profile_without_embedding_gets_the_builtin(
    tmp_path: Path,
) -> None:
    store = ModelProfileStore(tmp_path / "profiles.db")
    secrets = InMemorySecretStore()
    profile = store.create(
        ModelProfileInput(
            name="gen-only",
            provider_kind=ProviderKind.OLLAMA,
            model="qwen2.5",
        ),
        secrets,
    )
    assert profile.embedding_provider_kind is ProviderKind.LOCAL_ONNX
    assert profile.embedding_model == BUILTIN_EMBEDDING_MODEL
    assert profile.embedding_dimensions == BUILTIN_EMBEDDING_DIMENSIONS


def test_local_embedding_is_exempt_from_the_shared_origin_rule(
    tmp_path: Path,
) -> None:
    # Remote embedding must share the generation origin and key. Local
    # embedding never leaves the machine, so the same rule would block the
    # exact combination this feature exists for: remote generation + local
    # embedding.
    store = ModelProfileStore(tmp_path / "profiles.db")
    secrets = InMemorySecretStore()
    profile = store.create(
        ModelProfileInput(
            name="mixed",
            provider_kind=ProviderKind.OPENAI_COMPATIBLE,
            base_url="https://api.example.com/v1",
            model="gpt-4o-mini",
            embedding_provider_kind=ProviderKind.LOCAL_ONNX,
            embedding_model=BUILTIN_EMBEDDING_MODEL,
            embedding_dimensions=BUILTIN_EMBEDDING_DIMENSIONS,
            api_key="sk-test",
        ),
        secrets,
    )
    assert profile.embedding_provider_kind is ProviderKind.LOCAL_ONNX


def test_build_config_for_local_embedding_needs_no_embedding_key(
    tmp_path: Path,
) -> None:
    from tests.test_hy_memory_bridge import snapshot

    profile = snapshot(
        ProviderKind.OPENAI_COMPATIBLE,
        base_url="https://api.example.com/v1",
        has_api_key=True,
        embedding_provider=ProviderKind.LOCAL_ONNX,
    ).model_copy(
        update={
            "embedding_model": BUILTIN_EMBEDDING_MODEL,
            "embedding_dimensions": BUILTIN_EMBEDDING_DIMENSIONS,
        }
    )
    config = build_hy_memory_config(profile, api_key="sk-gen", data_dir=tmp_path)
    assert config.vector_store.embedding_dims == BUILTIN_EMBEDDING_DIMENSIONS
    assert config.embedder.embedding_dims == BUILTIN_EMBEDDING_DIMENSIONS


@requires_assets
def test_local_hy_embed_service_returns_unit_512_vectors() -> None:
    import asyncio

    service = LocalHyEmbedService(LocalEmbedder())
    vector = asyncio.run(service.embed("local embedding needs no network egress"))
    assert len(vector) == 512
    length = sum(value * value for value in vector) ** 0.5
    assert length == pytest.approx(1.0, abs=1e-4)


@requires_assets
def test_create_client_swaps_in_the_local_embed_service(tmp_path: Path) -> None:
    from tests.test_hy_memory_bridge import snapshot

    profile = snapshot(
        ProviderKind.OPENAI_COMPATIBLE,
        base_url="https://api.example.com/v1",
        has_api_key=True,
        embedding_provider=ProviderKind.LOCAL_ONNX,
    ).model_copy(
        update={
            "embedding_model": BUILTIN_EMBEDDING_MODEL,
            "embedding_dimensions": BUILTIN_EMBEDDING_DIMENSIONS,
        }
    )
    # HyMemoryClient opens local chroma/kuzu on a fresh data_dir and leaves
    # the embedder alone until first call, so this does not reach the network.
    client = create_hy_memory_client(profile, api_key="sk-gen", data_dir=tmp_path)
    try:
        assert isinstance(client._embed_service, LocalHyEmbedService)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def test_local_embed_service_covers_every_method_hy_calls() -> None:
    """Our replacement must cover every method HY actually calls -- the list is counted from \
HY's source, not hardcoded.

    The previous version of this class implemented only `embed`, and the comment still read "HY \
only ever calls .embed on it".
    In the HY version installed here, the write \
pipeline calls `embed_queued` (17 sites), `embed_batch`
    (12 sites), `embed` (11 sites) and `embed_with_meta` (1 \
site). The missing ones do not raise a missing-method error;
    instead HY falls back to its own service pointing at a placeholder address, so extraction \
ends in a connection failure and
    not one memory is produced.

    The list is counted rather than copied: a copied list rots with each HY upgrade,
    while this test's whole purpose is to watch that upgrade for us.
    """
    import re
    from pathlib import Path as _Path

    import hy_memory

    root = _Path(hy_memory.__file__).parent
    called: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name == "embed_service.py":
            continue
        called.update(re.findall(r"embed_service\.([a-z_]+)", path.read_text(encoding="utf-8")))

    assert called, "not a single call site was counted, which means this test itself is broken"
    missing = {name for name in called if not callable(getattr(LocalHyEmbedService, name, None))}
    assert missing == set(), f"HY calls these but we never implemented them: {sorted(missing)}"


@requires_assets
def test_the_pipelines_get_the_local_embed_service_not_only_the_client(
    tmp_path: Path,
) -> None:
    """The modules that do the real work must receive the local one as well.

    This was the missed step: in __init__ the client hands its own EmbedService to
    the component factory (client.py:300), and every pipeline takes it from the factory \
(registry.py:103/137/177).
    Replacing only the reference on client leaves \
the factory holding the original one -- pointing at
    http://127.0.0.1/local-onnx, where not even a TCP connection can be established.

    The old test only asserted the reference on client, \
so it stayed green while extraction kept producing nothing.
    """
    from tests.test_hy_memory_bridge import snapshot

    profile = snapshot(
        ProviderKind.OPENAI_COMPATIBLE,
        base_url="https://api.example.com/v1",
        has_api_key=True,
        embedding_provider=ProviderKind.LOCAL_ONNX,
    ).model_copy(
        update={
            "embedding_model": BUILTIN_EMBEDDING_MODEL,
            "embedding_dimensions": BUILTIN_EMBEDDING_DIMENSIONS,
        }
    )
    client = create_hy_memory_client(profile, api_key="sk-gen", data_dir=tmp_path)
    try:
        shared = client._registry._shared_embed_service
        assert isinstance(shared, LocalHyEmbedService)
        # The same object, not two separately cached replacements: HY's service carries its own
        # batch queue,
        # and two copies would each build a queue, while the point of a queue is that one batch
        # travels together.
        assert shared is client._embed_service
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


@requires_assets
def test_the_write_pipeline_itself_gets_the_local_embed_service(tmp_path: Path) -> None:
    """The pipeline that actually writes memories must hold the local one.

    This is the layer neither of the previous two guards covers, and it is the one that decides \
the outcome. Measured: after
    fixing both the client and factory references, real \
imports still hit 67 ConnectErrors and produced zero memories.
    The cause: client.__init__ already pre-warms System2Writer at client.py:332 --
    the replacement has not happened yet, so it takes HY's \
original service and passes it to the inner writer, and
    registry._system2_writer is never rebuilt once it holds a value.

    The call chain was measured, not read: attaching \
a stack trace to the real EmbedService's embed_queued
    makes it report client.async_add -> system2_writer.write ->
    writer.write -> embed_queued.
    """
    from tests.test_hy_memory_bridge import snapshot

    profile = snapshot(
        ProviderKind.OPENAI_COMPATIBLE,
        base_url="https://api.example.com/v1",
        has_api_key=True,
        embedding_provider=ProviderKind.LOCAL_ONNX,
    ).model_copy(
        update={
            "embedding_model": BUILTIN_EMBEDDING_MODEL,
            "embedding_dimensions": BUILTIN_EMBEDDING_DIMENSIONS,
        }
    )
    client = create_hy_memory_client(profile, api_key="sk-gen", data_dir=tmp_path)
    try:
        writer = asyncio.run(client._registry.get_system2_writer())
        assert isinstance(writer._embed_service, LocalHyEmbedService)
        # The inner writer too -- the stack trace shows embed_queued is called by it, not by
        # System2.
        inner = getattr(writer, "_writer", None) or getattr(writer, "writer", None)
        assert inner is not None, "inner writer not found, which means this test itself is broken"
        assert isinstance(inner._embed_service, LocalHyEmbedService)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


@requires_assets
def test_named_read_pipeline_uses_local_embedding_after_cache_replacement(tmp_path: Path) -> None:
    from tests.test_hy_memory_bridge import snapshot

    profile = snapshot(
        ProviderKind.OPENAI_COMPATIBLE,
        base_url="https://api.example.com/v1",
        has_api_key=True,
        embedding_provider=ProviderKind.LOCAL_ONNX,
    ).model_copy(
        update={
            "embedding_model": BUILTIN_EMBEDDING_MODEL,
            "embedding_dimensions": BUILTIN_EMBEDDING_DIMENSIONS,
        }
    )
    client = create_hy_memory_client(profile, api_key="test-generation-key", data_dir=tmp_path)
    try:
        reader = client._loop_thread.run(client._registry.get_reader(reader_name="legacy"))
        assert isinstance(reader.embed_service, LocalHyEmbedService)
        assert reader.embed_service is client._embed_service
        assert client._registry._readers["legacy"] is reader
    finally:
        client.close()
