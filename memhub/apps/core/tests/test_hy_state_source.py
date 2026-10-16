"""RC4：记忆生命周期（有效 / 被取代 / 去重隐藏 / 过期）直接读 HY，产品账本不再逐类镜像。

产品只保留 HY 没有的三样：使用范围、来源撤权后的隐藏过滤、用户删除记录。
这里钉住：HY 自己改了状态（导入写回从未见过的也算），读路径立刻按 HY 走；
HY 读不到时读路径退回账本，而不是把"没读到"当成"全部有效"。
"""

import asyncio
from pathlib import Path

import pytest
from hy_memory.models.memory import MemoryStatus as HyStatus

from one_memory.hy_lifecycle import HyLifecycle, HyLifecycleUnreadable
from one_memory.hy_memory_bridge import HyMemoryBridge
from one_memory.imports.grants import SourceGrantStore
from one_memory.imports.models import EvidenceInput, EvidenceOrigin, OriginMetadata, WrittenBy
from one_memory.imports.sources.registry import RootCandidateKind
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_governance import (
    HyCreate,
    MemoryDraft,
    MemoryGovernanceStore,
    MemoryKind,
    MemoryMutationConflictError,
    MemoryStatus,
)
from one_memory.memory_library import read_memory_library
from tests.person_door import as_person
from tests.test_hy_permission_raw_context import (
    AGENT_TEXT,
    USER_TEXT,
    ScriptedOracle,
    assert_outputs,
    bind_oracle,
    native_client,
    on_native_loop,
    raw_client,
)

FACT = "用户的发布前检查要求一次权限评审"


def _evidence(database: Path, message_id: str, *, conversation: str = "conversation") -> str:
    return (
        EvidenceStore(database)
        .put(
            EvidenceInput(
                source_id="cursor",
                conversation_id=conversation,
                message_id=message_id,
                correlation_id=message_id,
                role="user",
                written_by=WrittenBy.PERSON,
                verbatim=f"原话 {message_id}",
                origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
            )
        )
        .record.evidence_id
    )


def _library(tmp_path: Path) -> tuple[Path, MemoryGovernanceStore, dict[str, str]]:
    """Four memories, newest last: a, b, c, d. Each rests on its own evidence."""
    database = tmp_path / "one-memory.sqlite3"
    SourceGrantStore(database, home=tmp_path)
    store = MemoryGovernanceStore(database)
    ids: dict[str, str] = {}
    for key in ("a", "b", "c", "d"):
        evidence_id = _evidence(database, key)
        as_person(store).create(
            MemoryDraft(
                user_id="local",
                kind=MemoryKind.CHAT,
                content=f"{FACT} {key}",
                scopes=("all-apps",),
                evidence_ids=(evidence_id,),
            ),
            request_id=f"create-{key}",
            memory_id=key,
        )
        ids[key] = evidence_id
    return database, store, ids


def _ids(memories) -> list[str]:
    return [memory.memory_id for memory in memories]


# ---------------------------------------------------------------- 账本读路径


def test_what_hy_hides_leaves_every_current_read_and_shows_hy_state_by_id(tmp_path) -> None:
    _, store, _ = _library(tmp_path)
    # HY deduplicated d into b during a search, and replaced a with c.
    lifecycle = HyLifecycle(hidden={"d": ("b",), "a": ("c",)})

    assert _ids(store.list(user_id="local")) == ["d", "c", "b", "a"]
    assert _ids(store.list(user_id="local", lifecycle=lifecycle)) == ["c", "b"]
    found, more = store.find_confirmed_matching_terms(
        user_id="local", terms=(FACT,), lifecycle=lifecycle
    )
    assert _ids(found) == ["c", "b"] and more is False

    hidden = store.get("d", user_id="local", lifecycle=lifecycle)
    assert hidden.status is MemoryStatus.SUPERSEDED
    assert hidden.superseded_by == ("b",) and hidden.deleted_at is None
    kept = store.get("b", user_id="local", lifecycle=lifecycle)
    assert kept.status is MemoryStatus.CONFIRMED and kept.supersedes == ("d",)
    assert _ids(
        store.list(user_id="local", status=MemoryStatus.SUPERSEDED, lifecycle=lifecycle)
    ) == [
        "d",
        "a",
    ]
    assert _ids(
        store.list(user_id="local", status=MemoryStatus.CONFIRMED, lifecycle=lifecycle)
    ) == [
        "c",
        "b",
    ]


def test_hidden_rows_are_excluded_before_the_page_is_cut(tmp_path) -> None:
    _, store, _ = _library(tmp_path)
    lifecycle = HyLifecycle(hidden={"d": ("b",), "c": ("b",)})

    assert _ids(store.list(user_id="local", limit=1, lifecycle=lifecycle)) == ["b"]
    assert _ids(store.list(user_id="local", limit=1, offset=1, lifecycle=lifecycle)) == ["a"]
    found, more = store.find_confirmed_matching_terms(
        user_id="local", terms=(FACT,), limit=1, lifecycle=lifecycle
    )
    assert _ids(found) == ["b"] and more is True


def test_a_user_deletion_stays_a_deletion_whatever_hy_says(tmp_path) -> None:
    _, store, _ = _library(tmp_path)
    b = store.get("b", user_id="local")
    as_person(store).delete(
        "b", user_id="local", request_id="delete-b", expected_revision=b.revision
    )
    # HY still holds b as active until the deletion outbox reaches it.
    lifecycle = HyLifecycle(hidden={})

    assert "b" not in _ids(store.list(user_id="local", lifecycle=lifecycle))
    assert store.get("b", user_id="local", lifecycle=lifecycle).status is MemoryStatus.DELETED
    assert _ids(store.list(user_id="local", status=MemoryStatus.DELETED, lifecycle=lifecycle)) == [
        "b"
    ]


def test_hy_restoring_a_memory_it_had_replaced_brings_it_back(tmp_path) -> None:
    """A row an older build mirrored as superseded is current again once HY says so.

    HY restores the predecessor when the head of a chain is deleted; no write
    reaches the ledger for that, so only reading HY can see it.
    """
    from one_memory.memory_governance import HySupersede

    _, store, _ = _library(tmp_path)
    a = store.get("a", user_id="local")
    store.write_from_hy(
        HySupersede(
            memory_id="a",
            request_id="hy-supersede:t:a",
            expected_revision=a.revision,
            superseded_by=("c",),
        ),
        user_id="local",
        derivation=store.hy_derivation(),
    )
    assert "a" not in _ids(store.list(user_id="local"))

    restored = HyLifecycle(hidden={})
    assert "a" in _ids(store.list(user_id="local", lifecycle=restored))
    memory = store.get("a", user_id="local", lifecycle=restored)
    assert memory.status is MemoryStatus.CONFIRMED and memory.superseded_by == ()
    assert store.get("c", user_id="local", lifecycle=restored).supersedes == ()


def test_without_a_snapshot_the_ledger_reads_as_before(tmp_path) -> None:
    _, store, _ = _library(tmp_path)

    assert _ids(store.list(user_id="local", lifecycle=None)) == ["d", "c", "b", "a"]
    assert store.get("d", user_id="local", lifecycle=None).status is MemoryStatus.CONFIRMED


def test_a_superseded_memory_does_not_come_back_after_reopen(tmp_path) -> None:
    """Reopening the library may not resurrect what HY already replaced.

    The ledger's own mirrored supersession is the floor that holds while HY is
    not open; a second read path must not treat "no snapshot yet" as "active".
    """
    from one_memory.memory_governance import HySupersede

    database, store, _ = _library(tmp_path)
    a = store.get("a", user_id="local")
    store.write_from_hy(
        HySupersede(
            memory_id="a",
            request_id="hy-supersede:reopen:a",
            expected_revision=a.revision,
            superseded_by=("c",),
        ),
        user_id="local",
        derivation=store.hy_derivation(),
    )
    reopened = MemoryGovernanceStore(database)
    assert "a" not in _ids(reopened.list(user_id="local"))
    page = read_memory_library(database, limit=100)
    assert "a" not in [item.memory_id for item in page.items]


def test_revoked_source_stays_hidden_even_with_a_stale_lifecycle_snapshot(tmp_path) -> None:
    """Authority outranks an old HY snapshot: no snapshot may re-expose it."""
    database, store, _ = _library(tmp_path)
    stale = HyLifecycle(hidden={})
    assert _ids(store.list(user_id="local", lifecycle=stale)) == ["d", "c", "b", "a"]

    grants = SourceGrantStore(database, home=tmp_path)
    root = tmp_path / "cursor-root"
    root.mkdir()
    grants.grant(
        user_id="local",
        source_id="cursor",
        root=root,
        candidate_kind=RootCandidateKind.DEFAULT,
    )
    grants.revoke(user_id="local", source_id="cursor")
    # Documented contract: the ledger itself still lists the rows it owns.
    # Revocation is enforced by the read projections and external authority,
    # so a stale lifecycle snapshot must not re-expose them there.
    assert _ids(store.list(user_id="local", lifecycle=stale)) == ["d", "c", "b", "a"]
    page = read_memory_library(database, source="cursor", limit=100, lifecycle=stale)
    assert page.total == 0


def test_revocation_survives_reopen_and_keeps_the_ledger_intact(tmp_path) -> None:
    """Revocation hides reads; it does not delete, and it must outlive a restart.

    The governance store keeps its row by design, so a downstream consumer has
    to read through authority and must not treat the surviving row as current.
    """
    database, _, _ = _library(tmp_path)
    grants = SourceGrantStore(database, home=tmp_path)
    root = tmp_path / "cursor-root"
    root.mkdir()
    grants.grant(
        user_id="local",
        source_id="cursor",
        root=root,
        candidate_kind=RootCandidateKind.DEFAULT,
    )
    grants.revoke(user_id="local", source_id="cursor")

    reopened = MemoryGovernanceStore(database)
    receipt = SourceGrantStore(database, home=tmp_path).receipt(user_id="local", source_id="cursor")

    assert receipt is not None and receipt.revoked_at is not None
    # Rows survive; they are simply not readable through this source.
    assert len(reopened.list(user_id="local")) == 4
    assert read_memory_library(database, source="cursor", limit=100).total == 0


def test_the_memory_library_page_and_total_follow_hy(tmp_path) -> None:
    database, _, _ = _library(tmp_path)
    lifecycle = HyLifecycle(hidden={"d": ("b",)})

    page = read_memory_library(database, lifecycle=lifecycle)
    assert _ids(page.items) == ["c", "b", "a"] and page.total == 3
    replaced = read_memory_library(database, status=MemoryStatus.SUPERSEDED, lifecycle=lifecycle)
    assert _ids(replaced.items) == ["d"] and replaced.total == 1
    assert replaced.items[0].superseded_by == ("b",)
    active = read_memory_library(database, active_only=True, lifecycle=lifecycle, limit=1)
    assert _ids(active.items) == ["c"] and active.total == 3


# ---------------------------------------------------------------- 删除语义


def test_a_new_conversation_saying_a_deleted_fact_is_remembered_again(tmp_path) -> None:
    """Deleting removes that memory, not the fact: no content tombstone.

    Only the exact old evidence stays guarded (RC10); a different conversation
    saying the same thing later is a new statement.
    """
    database, store, _ = _library(tmp_path)
    a = store.get("a", user_id="local")
    as_person(store).delete(
        "a", user_id="local", request_id="delete-a", expected_revision=a.revision
    )
    again = _evidence(database, "again", conversation="later-conversation")

    created = store.write_from_hy(
        HyCreate(
            draft=MemoryDraft(
                user_id="local",
                kind=MemoryKind.CHAT,
                content=a.content,
                scopes=("all-apps",),
                evidence_ids=(again,),
            ),
            request_id="extract:later:new-a",
            memory_id="new-a",
        ),
        user_id="local",
        derivation=store.hy_derivation((again,)),
    )

    assert created.status is MemoryStatus.CONFIRMED and created.content == a.content
    assert "new-a" in _ids(store.list(user_id="local"))
    assert store.blocked_source_ids(user_id="local", evidence_ids=(again,)) == set()


def test_the_store_uses_its_lifecycle_source_for_reads_and_person_edits(tmp_path) -> None:
    _, store, _ = _library(tmp_path)
    hidden = HyLifecycle(hidden={"d": ("b",)})
    calls: list[tuple[str, tuple[str, ...] | None]] = []

    def source(user_id: str, ids: tuple[str, ...] | None = None) -> HyLifecycle:
        calls.append((user_id, ids))
        return hidden

    store.read_lifecycle_from(source)
    assert _ids(store.list(user_id="local")) == ["c", "b", "a"]
    current = store.get("d", user_id="local")
    assert current is not None and current.status is MemoryStatus.SUPERSEDED
    assert calls == [("local", None), ("local", ("d",))]
    # Editing what HY has set aside would fork the chain it already replaced.
    with pytest.raises(MemoryMutationConflictError, match="memory_superseded"):
        as_person(store).correct(
            "d",
            user_id="local",
            request_id="correct-hidden",
            expected_revision=current.revision,
            content=f"{FACT} corrected",
        )


def test_a_rejected_or_missing_source_answers_degraded_not_all_active(tmp_path) -> None:
    """Wiring contract for C0/C4: unreadable means unavailable, never 'active'.

    A source that raises must be visible as an error to the composition root; a
    source that answers `None` keeps the ledger floor rather than inventing
    freshness. Neither may be read as 'HY says everything is current'.
    """
    _, store, _ = _library(tmp_path)

    def broken(_user_id: str, _ids: tuple[str, ...] | None = None) -> HyLifecycle:
        raise HyLifecycleUnreadable("hy_lifecycle_unreadable")

    store.read_lifecycle_from(broken)
    with pytest.raises(HyLifecycleUnreadable):
        store.list(user_id="local")
    with pytest.raises(HyLifecycleUnreadable):
        store.get("d", user_id="local")

    # Before HY is open, `None` keeps the ledger's own supersession floor:
    # a memory HY replaced stays hidden because the write was observed here.
    from one_memory.memory_governance import HySupersede

    _, store, _ = _library(tmp_path)
    a = store.get("a", user_id="local")
    store.write_from_hy(
        HySupersede(
            memory_id="a",
            request_id="hy-supersede:floor:a",
            expected_revision=a.revision,
            superseded_by=("c",),
        ),
        user_id="local",
        derivation=store.hy_derivation(),
    )
    store.read_lifecycle_from(lambda _user_id, _ids=None: None)
    assert "a" not in _ids(store.list(user_id="local"))


def test_one_snapshot_per_page_not_one_per_row(tmp_path) -> None:
    """Wiring contract for C4: a page read asks HY once, not once per row."""
    _, store, _ = _library(tmp_path)
    calls: list[tuple[str, tuple[str, ...] | None]] = []

    def source(user_id: str, ids: tuple[str, ...] | None = None) -> HyLifecycle:
        calls.append((user_id, ids))
        return HyLifecycle(hidden={"d": ("b",)})

    store.read_lifecycle_from(source)
    assert len(store.list(user_id="local")) == 3
    assert calls == [("local", None)]


def test_an_unreadable_lifecycle_blocks_edits_but_not_a_local_delete(tmp_path) -> None:
    _, store, _ = _library(tmp_path)
    current = store.get("d", user_id="local", lifecycle=None)
    assert current is not None

    def unreadable(_user_id: str, _ids: tuple[str, ...] | None = None) -> HyLifecycle:
        raise HyLifecycleUnreadable("hy_lifecycle_unreadable")

    store.read_lifecycle_from(unreadable)
    with pytest.raises(HyLifecycleUnreadable):
        as_person(store).correct(
            "d",
            user_id="local",
            request_id="correct-unreadable",
            expected_revision=current.revision,
            content=f"{FACT} corrected",
        )

    deleted = as_person(store).delete(
        "d",
        user_id="local",
        request_id="delete-unreadable",
        expected_revision=current.revision,
    )
    assert deleted.status is MemoryStatus.DELETED


# ---------------------------------------------------------------- 真 HY


async def _first_turn(supervisor, records):
    first = supervisor.prepare_conversation(records[0])[0]
    client = await native_client(supervisor, first)
    bind_oracle(supervisor, ScriptedOracle(records))
    assert (await supervisor._create_runner().run(first.task_id)).status == "completed"
    await assert_outputs(supervisor, first, records, client)
    by_content = {m.content: m for m in supervisor.governance.list(user_id="local")}
    bridge = supervisor._bridge(first.profile_snapshot, None, supervisor.data_dir)
    return client, bridge, by_content[USER_TEXT], by_content[AGENT_TEXT]


def test_hy_search_dedup_is_read_straight_from_hy(tmp_path) -> None:
    """HY shadows a duplicate while searching; no ingest write-back ever sees it."""

    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, _):
            client, bridge, dup, kept = await _first_turn(supervisor, records)
            assert isinstance(bridge, HyMemoryBridge)
            assert bridge.lifecycle_now("local") == HyLifecycle(hidden={})

            await on_native_loop(
                client,
                client._vector_store.update_payload(
                    dup.memory_id,
                    {
                        "status": HyStatus.SHADOW.value,
                        "is_latest": False,
                        "shadow_reason": "dedup:search",
                        "dedup_kept_id": kept.memory_id,
                    },
                ),
            )

            lifecycle = bridge.lifecycle_now("local")
            assert lifecycle == HyLifecycle(hidden={dup.memory_id: (kept.memory_id,)})
            assert _ids(supervisor.governance.list(user_id="local", lifecycle=lifecycle)) == [
                kept.memory_id
            ]
            assert bridge.lifecycle_now("someone-else") == HyLifecycle(hidden={})

    asyncio.run(scenario())


def test_a_replacement_hy_records_on_the_old_node_is_read_from_hy(tmp_path) -> None:
    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, _):
            client, bridge, old, head = await _first_turn(supervisor, records)
            await on_native_loop(
                client,
                client._vector_store.update_payload(
                    old.memory_id,
                    {
                        "status": HyStatus.SUPERSEDED.value,
                        "is_latest": False,
                        "superseded_by": [head.memory_id],
                    },
                ),
            )

            lifecycle = bridge.lifecycle_now("local")
            assert lifecycle == HyLifecycle(hidden={old.memory_id: (head.memory_id,)})
            memory = supervisor.governance.get(old.memory_id, user_id="local", lifecycle=lifecycle)
            assert memory.status is MemoryStatus.SUPERSEDED
            assert memory.superseded_by == (head.memory_id,)

    asyncio.run(scenario())


def test_an_unreadable_hy_store_raises_instead_of_reading_as_all_active(tmp_path) -> None:
    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, _):
            client, bridge, dup, kept = await _first_turn(supervisor, records)
            await on_native_loop(
                client,
                client._vector_store.update_payload(
                    dup.memory_id,
                    {"status": HyStatus.SHADOW.value, "is_latest": False},
                ),
            )
            # Positive first: this store does hold something hidden.
            assert bridge.lifecycle_now("local") == HyLifecycle(hidden={dup.memory_id: ()})

            collection = client._vector_store._collection

            def broken(*args, **kwargs):
                raise RuntimeError("chroma is gone")

            original = collection.get
            collection.get = broken
            try:
                with pytest.raises(HyLifecycleUnreadable):
                    bridge.lifecycle_now("local")
            finally:
                collection.get = original

    asyncio.run(scenario())


def test_hy_not_opened_yet_is_only_safe_when_no_store_exists(tmp_path) -> None:
    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, _):
            first = supervisor.prepare_conversation(records[0])[0]
            bridge = supervisor._bridge(first.profile_snapshot, None, supervisor.data_dir)
            assert bridge.lifecycle_now("local") is None

            (supervisor.data_dir / "hy-memory" / "governed-v2" / "graph").mkdir(parents=True)
            with pytest.raises(HyLifecycleUnreadable, match="hy_lifecycle_not_ready"):
                bridge.lifecycle_now("local")

    asyncio.run(scenario())


def test_malformed_hy_lifecycle_rows_fail_closed(tmp_path) -> None:
    async def scenario():
        async with raw_client(tmp_path) as (supervisor, records, _):
            client, bridge, _dup, _kept = await _first_turn(supervisor, records)
            collection = client._vector_store._collection
            original = collection.get
            collection.get = lambda **_kwargs: {"ids": ["missing-metadata"], "metadatas": []}
            try:
                with pytest.raises(HyLifecycleUnreadable, match="hy_lifecycle_store_mismatch"):
                    bridge.lifecycle_now("local")
            finally:
                collection.get = original

    asyncio.run(scenario())
