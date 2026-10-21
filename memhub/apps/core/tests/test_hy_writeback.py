"""HY results enter the governance tables through a single door, and every pass carries the \
evidence snapshot taken when the call began.

The regression scenarios cover several write-back paths once missed (four in extraction, digest, \
retired, supersede),
showing the problem is scattered paths, not one spot. A static scan guards "only one door" here, \
while
the digest, retired and supersede cases where a human overrules mid-HY-call are each reproduced.
"""

import ast
import asyncio
import re
import sqlite3
from pathlib import Path

import pytest

import one_memory
from one_memory.hy_memory_bridge import HyGraphNode, HyMemoryIngestResult, HyMemoryOutput
from one_memory.imports.models import WrittenBy
from one_memory.memory_governance import (
    HyCreate,
    MemoryEvidenceOverriddenError,
    MemoryGovernanceStore,
    MemoryStatus,
)
from one_memory.memory_processing import MemoryProcessingSupervisor
from tests.person_door import as_person
from tests.test_memory_no_candidate import _HeldBridge, _manual, _overrule_mid_call
from tests.test_memory_processing import FakeBridge, _put, _runner, _setup, _task

SOURCE = Path(one_memory.__file__).parent
#: What a person does to a memory.
PERSON_WRITES = ("create", "correct", "set_scopes", "undo", "delete")
#: Ways HY results used to reach governance directly. None may exist any more.
DIRECT_HY_WRITES = {
    "reconcile_hy",
    "supersede",
    "link_evidence",
    "complete_deletion",
    "fail_deletion",
}


# # ------------------------------------------------------------ The runtime door: does not rely on
# # static scanning
#
# The regressions showed that scanning by attribute name plus receiver text cannot recognize
# aliases, getattr, calls through the class,
# or a stored bound method. So the safety boundary sits at runtime: the governance store itself has
# no manual write methods,
# and manual writing can only come from the PersonWrites it issues once (claimed by Core's API at
# startup).


def _seed_record(tmp_path):
    stores = _setup(tmp_path)
    evidence, _profiles, _tasks, governance, _secrets, _profile = stores
    record = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I live in Hangzhou.",
    )
    return governance, record


def _alias(governance, draft):
    store = governance
    return store.create(draft, request_id="alias")


def _getattr(governance, draft):
    return getattr(governance, "create")(draft, request_id="getattr")  # noqa: B009


def _through_the_class(governance, draft):
    return MemoryGovernanceStore.create(governance, draft, request_id="class")


def _bound_first(governance, draft):
    write = governance.create
    return write(draft, request_id="bound")


@pytest.mark.parametrize(
    "bypass",
    [_alias, _getattr, _through_the_class, _bound_first],
    ids=["alias", "getattr", "through-class", "bound-method"],
)
def test_code_holding_only_the_store_cannot_write_as_the_person(tmp_path, bypass) -> None:
    governance, record = _seed_record(tmp_path)

    with pytest.raises(AttributeError):
        bypass(governance, _draft(record, "memory written past the person entry point"))

    assert governance.list(user_id="local") == []


@pytest.mark.parametrize("name", PERSON_WRITES)
def test_the_store_has_no_person_write_of_its_own(name) -> None:
    assert not hasattr(MemoryGovernanceStore, name)


def test_the_person_door_is_issued_once_and_cannot_be_forged(tmp_path) -> None:
    from one_memory.memory_governance import PersonWrites

    governance, record = _seed_record(tmp_path)
    with pytest.raises(ValueError, match="actor"):
        governance.claim_person_writes("hy")
    door = governance.claim_person_writes("user")
    # Positive premise: the claimed door really can write.
    written = door.create(
        _draft(record, "memory written by the person themselves"), request_id="door"
    )
    assert governance.get(written.memory_id, user_id="local") == written

    with pytest.raises(PermissionError, match="person_writes_already_claimed"):
        governance.claim_person_writes("user")
    with pytest.raises(PermissionError, match="person_writes_not_issued"):
        PersonWrites(governance, "user", key=object())


def test_core_takes_the_person_door_before_anything_else_can(tmp_path) -> None:
    from tests.test_memory_no_candidate import _app

    async def scenario():
        app = _app(tmp_path)
        async with app.router.lifespan_context(app):
            assert app.state.person_writes.actor == "api"
            with pytest.raises(PermissionError, match="person_writes_already_claimed"):
                app.state.memory_governance_store.claim_person_writes("user")

    asyncio.run(scenario())


# # ------------------------------------------------------------ Static scanning: a supplement only


def _governance_private_names() -> set[str]:
    """Of the governance layer's own private members, the ones no other module defines under the \
same name -- appearing elsewhere can only mean reaching in."""
    from one_memory.memory_governance import PersonWrites

    own = {
        name
        for cls in (MemoryGovernanceStore, PersonWrites)
        for name in vars(cls)
        if name.startswith("_") and not name.startswith("__")
    } | {"_person_key", "_person_claimed"}
    elsewhere = set()
    for path in SOURCE.rglob("*.py"):
        if path.name == "memory_governance.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                elsewhere.add(node.name)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                elsewhere.add(node.attr)
    return own - elsewhere


def _reaches_into_governance(source: str, private: set[str]) -> list[tuple[int, str]]:
    """Attribute references and string constants both count: aliases, getattr, calls through the \
class and stored bound methods cannot escape the name itself."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in private | {"claim_person_writes"}:
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.Constant) and node.value in private | {"claim_person_writes"}:
            found.append((node.lineno, node.value))
        elif isinstance(node, ast.Call) and ast.unparse(node.func).endswith("PersonWrites"):
            # Naming the type is harmless; building one is reaching in.
            found.append((node.lineno, "PersonWrites"))
    return found


def test_the_scan_sees_every_bypass_shape() -> None:
    private = _governance_private_names()
    # Positive premise: the write entry point's own private names are all in the list and were not
    # deleted as "same name elsewhere".
    assert {"_create", "_mutate", "_person_key"} <= private
    for shape in (
        "store = processor.governance\nstore._create(d, request_id='r')",
        "getattr(processor.governance, '_mutate')(m)",
        "MemoryGovernanceStore._create(processor.governance, d)",
        "write = processor.governance._mutate\nwrite(m)",
        "processor.governance.claim_person_writes('api')",
        "PersonWrites(store, 'api', key=store._person_key)",
    ):
        assert _reaches_into_governance(shape, private), shape


def test_only_the_api_takes_the_person_door_and_nothing_reaches_into_governance() -> None:
    private = _governance_private_names()
    found = {
        path.relative_to(SOURCE).as_posix(): hits
        for path in sorted(SOURCE.rglob("*.py"))
        if path.name != "memory_governance.py"
        and (hits := _reaches_into_governance(path.read_text(encoding="utf-8"), private))
    }
    api_hits = {name for _line, name in found.pop("api.py", [])}
    assert api_hits == {"claim_person_writes"}
    assert found == {}


def test_governance_has_one_public_door_for_hy_results() -> None:
    public = {
        name
        for name, value in vars(MemoryGovernanceStore).items()
        if callable(value) and not name.startswith("_")
    }
    assert "write_from_hy" in public
    assert public & DIRECT_HY_WRITES == set()


# # ------------------------------------------------------------ Writing SQL straight past the door
#
# The regressions showed: the door guards Python calls, but not someone hand-writing one UPDATE
# governed_memories.
# So this scans SQL text in production source as a supplement; it cannot see table names assembled
# at runtime, so it is not the only defense.

GOVERNANCE_TABLES = (
    "governed_memories",
    "governed_memory_versions",
    "governed_memory_evidence",
    "governed_memory_mutations",
    "governed_memory_supersessions",
    "memory_deletions",
    "memory_index_syncs",
)
_GOVERNANCE_WRITE = re.compile(
    r"\b(?:INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE(?:\s+OR\s+\w+)?|DELETE\s+FROM"
    r"|DROP\s+TABLE(?:\s+IF\s+EXISTS)?|ALTER\s+TABLE)\s+[\"`\[]?(?:main\.)?(?:"
    + "|".join(GOVERNANCE_TABLES)
    + r")\b",
    re.IGNORECASE,
)


def _governance_writes(source: str) -> list[int]:
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value if isinstance(part, ast.Constant) else "?" for part in node.values
            )
        else:
            continue
        if _GOVERNANCE_WRITE.search(text):
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize(
    "sample",
    [
        'db.execute("UPDATE governed_memories SET status = ?", (s,))',
        "db.execute('delete from memory_deletions where memory_id = ?', (m,))",
        'db.execute("INSERT OR REPLACE INTO governed_memory_evidence VALUES (?, ?, 1)")',
        'db.execute("REPLACE INTO \\"governed_memory_versions\\" VALUES (?)")',
        'db.execute("UPDATE main.governed_memory_mutations SET x = 1")',
        'db.execute("UPDATE "\n    "governed_memories SET content = ?")',
        'db.execute(f"DELETE FROM governed_memory_supersessions WHERE id = {i}")',
        'db.executescript("DROP TABLE IF EXISTS memory_index_syncs;")',
        'db.execute("ALTER TABLE governed_memories ADD COLUMN x")',
    ],
)
def test_the_sql_scan_sees_a_write_to_a_governance_table(sample) -> None:
    assert _governance_writes(sample) != []


@pytest.mark.parametrize(
    "sample",
    [
        'db.execute("SELECT * FROM governed_memories WHERE memory_id = ?")',
        'db.execute("SELECT 1 FROM x JOIN governed_memory_evidence USING(memory_id)")',
        'db.execute("CREATE TRIGGER t AFTER UPDATE ON governed_memory_versions BEGIN END")',
        'db.execute("INSERT INTO evidence VALUES (?) ON CONFLICT DO UPDATE SET x = 1")',
    ],
)
def test_reading_a_governance_table_is_not_a_write(sample) -> None:
    assert _governance_writes(sample) == []


def test_only_governance_writes_to_its_own_tables() -> None:
    offenders = []
    own = []
    for path in sorted(SOURCE.rglob("*.py")):
        found = _governance_writes(path.read_text(encoding="utf-8"))
        if path.name in {"memory_governance.py", "migrations.py"}:
            own = found
            continue
        offenders += [f"{path.relative_to(SOURCE)}:{line}" for line in found]
    # Non-zero cardinality: the scan does recognize write statements inside the governance module
    # itself, which is what makes the empty table below meaningful.
    assert len(own) >= 10
    assert offenders == []


# ------------------------------------------------------------------ digest


class _HeldDigest(FakeBridge):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def digest(self, *, user_id: str, agent_id: str) -> dict[str, object]:
        self.entered.set()
        await self.release.wait()
        return await super().digest(user_id=user_id, agent_id=agent_id)


def _digest_scene(tmp_path):
    stores = _setup(tmp_path)
    evidence, profiles, tasks, governance, secrets, profile = stores
    record = _put(
        evidence,
        message_id="m1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="A fact inside the project scope.",
        workspace="workspace:v1:project",
    )
    for key in ("seed", "doomed"):
        as_person(governance).create(
            _draft(record, f"memory {key}", scopes=("workspace:v1:project",)),
            request_id=f"create-{key}",
            memory_id=key,
        )
    supervisor = MemoryProcessingSupervisor(
        tasks=tasks,
        profiles=profiles,
        secrets=secrets,
        evidence=evidence,
        governance=governance,
        data_dir=tmp_path,
    )
    bridge = _HeldDigest(
        outputs=(),
        graph_nodes=(
            HyGraphNode(
                node_id="schema-1",
                content="The user prefers verifiable conclusions.",
                layer="l6_schema",
            ),
        ),
    )
    supervisor._bridges[(profile.profile_id, profile.revision, None)] = bridge
    return governance, supervisor, bridge


def _draft(record, content, *, scopes=("all-apps",)):
    from one_memory.memory_governance import MemoryDraft, MemoryKind

    return MemoryDraft(
        user_id="local",
        kind=MemoryKind.CHAT,
        content=content,
        scopes=scopes,
        evidence_ids=(record.evidence_id,),
    )


def _delete(governance, memory_id):
    memory = governance.get(memory_id, user_id="local")
    as_person(governance).delete(
        memory_id,
        user_id="local",
        request_id=f"delete-{memory_id}",
        expected_revision=memory.revision,
    )


def test_a_schema_digested_while_its_evidence_is_overruled_is_not_recorded(tmp_path) -> None:
    governance, supervisor, bridge = _digest_scene(tmp_path)

    async def scenario():
        running = asyncio.create_task(supervisor.digest())
        await bridge.entered.wait()
        _delete(governance, "doomed")
        bridge.release.set()
        return await running

    result = asyncio.run(scenario())

    assert bridge.digest_calls, "HY must actually have been called"
    assert governance.get("schema-1", user_id="local") is None
    assert result["schemas_recorded"] == 0


def test_a_deletion_before_the_digest_began_does_not_hold_the_schema_back(tmp_path) -> None:
    """Overruling before the snapshot was already gated by the context; the write only looks at \
mid-call overruling, otherwise evidence touched by a deletion could never be written in."""
    governance, supervisor, bridge = _digest_scene(tmp_path)
    _delete(governance, "doomed")
    bridge.release.set()

    result = asyncio.run(supervisor.digest())

    assert result["schemas_recorded"] == 1
    assert governance.get("schema-1", user_id="local").status is MemoryStatus.CONFIRMED


# ------------------------------------------------------------ retired / supersede


class _HeldResult(_HeldBridge):
    """The HY call stopped halfway; on the way back it reports which memories it deleted or \
superseded."""

    def __init__(self, *, retired=(), superseded=None) -> None:
        super().__init__(outputs=())
        self.retired = retired
        self.superseded = superseded or {}

    async def ingest(self, request):
        self.entered.set()
        await self.release.wait()
        self.requests.append(request)
        return HyMemoryIngestResult(
            success=True,
            retired_memory_ids=self.retired,
            superseded_memory_ids=self.superseded,
        )


def _turn(stores):
    evidence, _profiles, _tasks, governance, _secrets, _profile = stores
    elsewhere = _put(
        evidence,
        message_id="u0",
        role="user",
        written_by=WrittenBy.PERSON,
        content="source text from elsewhere",
    )
    record = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I live in Hangzhou.",
    )
    doomed = _manual(governance, record, "The user lives in Hangzhou.", "seed-doomed")
    return elsewhere, record, doomed


def test_hy_retiring_a_memory_mid_call_is_refused_once_its_evidence_is_overruled(tmp_path) -> None:
    stores = _setup(tmp_path)
    _evidence, _profiles, tasks, governance, _secrets, profile = stores
    elsewhere, record, doomed = _turn(stores)
    retiring = _manual(governance, elsewhere, "The user lives in Shanghai.", "seed-retiring")
    bridge = _HeldResult(retired=(retiring.memory_id,))
    task = _task(tasks, profile, record)

    _overrule_mid_call(
        stores, bridge, doomed, lambda: _runner(tmp_path, stores, bridge).run(task.task_id)
    )

    assert governance.get(retiring.memory_id, user_id="local").status is MemoryStatus.CONFIRMED
    assert tasks.get(task.task_id).status == "unknown"
    assert tasks.get(task.task_id).last_error_code == "hy_memory_evidence_changed"


def test_hy_superseding_a_memory_mid_call_is_refused_once_its_evidence_is_overruled(
    tmp_path,
) -> None:
    stores = _setup(tmp_path)
    _evidence, _profiles, tasks, governance, _secrets, profile = stores
    elsewhere, record, doomed = _turn(stores)
    old = _manual(governance, elsewhere, "The user lives in Shanghai.", "seed-old")
    head = _manual(governance, elsewhere, "The user moved to Hangzhou.", "seed-head")
    bridge = _HeldResult(superseded={old.memory_id: (head.memory_id,)})
    task = _task(tasks, profile, record)

    _overrule_mid_call(
        stores, bridge, doomed, lambda: _runner(tmp_path, stores, bridge).run(task.task_id)
    )

    assert governance.get(old.memory_id, user_id="local").status is MemoryStatus.CONFIRMED
    assert tasks.get(task.task_id).status == "unknown"
    assert tasks.get(task.task_id).last_error_code == "hy_memory_evidence_changed"


def test_what_hy_itself_retired_in_a_turn_does_not_block_the_rest_of_that_turn(tmp_path) -> None:
    """HY deleting on its own is not a human overruling: in one round it deletes one and \
supersedes one, and both must land."""
    stores = _setup(tmp_path)
    evidence, _profiles, tasks, governance, _secrets, profile = stores
    record = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I live in Hangzhou.",
    )
    retired = _manual(governance, record, "The user lives in Shanghai.", "seed-retired")
    old = _manual(governance, record, "The user works in Shanghai.", "seed-old")
    head = _manual(governance, record, "The user works in Hangzhou.", "seed-head")
    bridge = _HeldResult(
        retired=(retired.memory_id,), superseded={old.memory_id: (head.memory_id,)}
    )
    bridge.release.set()
    task = _task(tasks, profile, record)

    finished = asyncio.run(_runner(tmp_path, stores, bridge).run(task.task_id))

    assert (finished.status, finished.last_error_code) == ("completed", None)
    assert governance.get(retired.memory_id, user_id="local").status is MemoryStatus.DELETED
    assert governance.get(old.memory_id, user_id="local").status is MemoryStatus.SUPERSEDED


# # ------------------------------------------------------------ Tombstone judgement only looks at
# # evidence in use


@pytest.mark.parametrize("still_resting", [True, False], ids=["resting", "moved-away"])
def test_only_evidence_a_deleted_memory_still_rests_on_is_held_back(
    tmp_path, still_resting
) -> None:
    stores = _setup(tmp_path)
    evidence, _profiles, tasks, governance, _secrets, profile = stores
    first = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I live in Hangzhou.",
    )
    later = _put(
        evidence,
        message_id="u9",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I moved to Shanghai.",
        conversation_id="elsewhere",
    )
    memory = _manual(governance, first, "The user lives in Hangzhou.", "seed-memory")
    if not still_resting:
        # This is how HY swaps a memory's evidence when reconciling again: the old link stays as
        # history and is no longer in use.
        with sqlite3.connect(tmp_path / "one-memory.sqlite3") as db:
            db.execute(
                "UPDATE governed_memory_evidence SET active=0 WHERE memory_id=?",
                (memory.memory_id,),
            )
            db.execute(
                "INSERT INTO governed_memory_evidence (memory_id,evidence_id,active) "
                "VALUES (?,?,1)",
                (memory.memory_id, later.evidence_id),
            )
    _delete(governance, memory.memory_id)
    bridge = FakeBridge(
        outputs=(
            HyMemoryOutput(
                memory_id="hy-new", content="The user is living in Hangzhou.", layer="L0_BASIC_INFO"
            ),
        )
    )
    task = _task(tasks, profile, first)

    finished = asyncio.run(_runner(tmp_path, stores, bridge).run(task.task_id))

    if still_resting:
        assert (finished.status, finished.last_error_code) == (
            "cancelled",
            "hy_memory_evidence_changed",
        )
    else:
        assert (finished.status, finished.last_error_code) == ("completed", None)
        assert governance.get("hy-new", user_id="local").status is MemoryStatus.CONFIRMED


@pytest.mark.parametrize("still_resting", [True, False], ids=["resting", "moved-away"])
def test_a_deletion_mid_call_holds_back_only_evidence_the_memory_still_rests_on(
    tmp_path, still_resting
) -> None:
    """The same criterion is re-checked inside the write transaction: a memory deleted mid-call \
whose evidence already changed leaves the old evidence untouched."""
    stores = _setup(tmp_path)
    evidence, _profiles, _tasks, governance, _secrets, _profile = stores
    first = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I live in Hangzhou.",
    )
    later = _put(
        evidence,
        message_id="u9",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I moved to Shanghai.",
        conversation_id="elsewhere",
    )
    memory = _manual(governance, first, "The user lives in Hangzhou.", "seed-memory")
    if not still_resting:
        with sqlite3.connect(tmp_path / "one-memory.sqlite3") as db:
            db.execute(
                "UPDATE governed_memory_evidence SET active=0 WHERE memory_id=?",
                (memory.memory_id,),
            )
            db.execute(
                "INSERT INTO governed_memory_evidence (memory_id,evidence_id,active) "
                "VALUES (?,?,1)",
                (memory.memory_id, later.evidence_id),
            )
    derivation = governance.hy_derivation((first.evidence_id,))
    _delete(governance, memory.memory_id)

    def write():
        return governance.write_from_hy(
            HyCreate(
                draft=_draft(first, "The user is living in Hangzhou."),
                request_id="hy-create",
                memory_id="hy-new",
            ),
            user_id="local",
            derivation=derivation,
        )

    if still_resting:
        with pytest.raises(MemoryEvidenceOverriddenError):
            write()
        assert governance.get("hy-new", user_id="local") is None
    else:
        assert write().status is MemoryStatus.CONFIRMED


# # ------------------------------------------------------------ Matter discovery backfills links


def test_matter_discovery_does_not_link_evidence_overruled_while_it_reviewed(
    tmp_path, monkeypatch
) -> None:
    """Citations that pass matter review are backfilled as memory evidence; if a human deletes \
the memory built on that source text during review, it can no longer be backfilled."""
    from one_memory.matters import discovery as discovery_module
    from one_memory.matters.discovery import MatterDiscovery
    from one_memory.matters.model import MatterModel
    from one_memory.matters.models import Citation, Claim, DiscoveryInput
    from one_memory.matters.store import MatterStore
    from tests.test_matter_hy_integration import Bridge
    from tests.test_matter_hy_integration import setup as matter_setup

    def run(overrule: bool) -> tuple[object, object, object]:
        root = tmp_path / ("overruled" if overrule else "plain")
        root.mkdir()

        async def scenario():
            processor, _record, _ = matter_setup(root, Bridge())
            linked: dict[str, object] = {}

            async def classify(_model, memories, refs, _records, **_kwargs):
                # Only after extraction does this source text exist; HY's memory was never built on
                # it, so backfilling the link has work to do.
                quoted = _put(
                    processor.evidence,
                    message_id="q1",
                    role="user",
                    written_by=WrittenBy.PERSON,
                    content="Atlas stores data only on this machine.",
                    conversation_id="atlas",
                )
                linked["resting"] = _manual(
                    processor.governance,
                    quoted,
                    "The user stores data only on this machine.",
                    "resting",
                )
                key = next(iter(memories))
                assert quoted.evidence_id not in memories[key].evidence_ids
                linked["memory_id"], linked["quoted"] = key, quoted
                claim = Claim(
                    statement=memories[key].content,
                    kind="constraint",
                    citations=[
                        Citation(evidence_id=quoted.evidence_id, quote="Atlas stores data only")
                    ],
                    memory_refs=[refs[key]],
                )
                return {"c0": claim}, {}, 0

            async def review(_self, claims, *_args, **_kwargs):
                if overrule:
                    _delete(processor.governance, linked["resting"].memory_id)
                return claims, 0

            monkeypatch.setattr(discovery_module, "classify_memories", classify)
            monkeypatch.setattr(MatterDiscovery, "_review_and_repair", review)
            store = MatterStore(processor.tasks.database_path)
            service = MatterDiscovery(
                store, MatterModel(processor.profiles, processor.secrets), processor
            )
            try:
                result = await service.run(DiscoveryInput(max_chars=10_000))
                memory = processor.governance.get(linked["memory_id"], user_id="local")
                return result, memory, linked["quoted"]
            finally:
                await processor.shutdown()

        return asyncio.run(scenario())

    # Positive premise: with no overruling, this citation really is backfilled as evidence.
    _plain, memory, quoted = run(overrule=False)
    assert quoted.evidence_id in memory.evidence_ids

    result, memory, quoted = run(overrule=True)
    assert quoted.evidence_id not in memory.evidence_ids
    assert (result.status, result.error) == ("failed", "matter_memory_changed_during_generation")


# # ------------------------------------------------------------ The check covers every evidence
# # being written
#
# The regressions showed: the encoding-layer revision writes (old evidence UNION new evidence), but
# the caller only checked the new ones.
# Mid-call a human deleted another memory built on the old evidence; the revision was written anyway
# and the overruled source text was grafted back.
# Making every caller remember to pass everything is the same disease as the last round's "each path
# remembers to check", so the door completes it itself.


def _written_evidence_scene(tmp_path):
    stores = _setup(tmp_path)
    evidence, _profiles, _tasks, governance, _secrets, _profile = stores
    old = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I live in Hangzhou.",
    )
    new = _put(
        evidence,
        message_id="u2",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I work in Hangzhou.",
        conversation_id="elsewhere",
    )
    target = _manual(governance, old, "The user lives in Hangzhou.", "seed-target")
    doomed = _manual(governance, old, "The user's home is in Hangzhou.", "seed-doomed")
    return governance, old, new, target, doomed


def _change(kind, old, new, target):
    from one_memory.memory_governance import HyLink, HyRevise

    if kind == "create":
        return HyCreate(
            draft=_draft(old, "The user is living in Hangzhou."),
            request_id="hy-create",
            memory_id="hy-new",
        )
    if kind == "revise":
        return HyRevise(
            memory_id=target.memory_id,
            request_id="hy-revise",
            expected_revision=target.revision,
            content="The user lives in Hangzhou and works there.",
            evidence_ids=(old.evidence_id, new.evidence_id),
        )
    return HyLink(memory_id="hy-new", evidence_id=old.evidence_id)


@pytest.mark.parametrize("kind", ["create", "revise", "link"])
@pytest.mark.parametrize("cited", ["everything", "only-new"])
def test_a_write_is_checked_against_every_evidence_it_writes(tmp_path, kind, cited) -> None:
    governance, old, new, target, doomed = _written_evidence_scene(tmp_path)
    if kind == "link":
        governance.write_from_hy(
            HyCreate(
                draft=_draft(new, "The user works in Hangzhou."),
                request_id="seed",
                memory_id="hy-new",
            ),
            user_id="local",
            derivation=governance.hy_derivation((new.evidence_id,)),
        )
    derivation = governance.hy_derivation((old.evidence_id, new.evidence_id))
    if cited == "only-new":
        derivation = derivation.citing((new.evidence_id,))
    before = governance.get("hy-new" if kind != "revise" else target.memory_id, user_id="local")
    _delete(governance, doomed.memory_id)

    with pytest.raises(MemoryEvidenceOverriddenError):
        governance.write_from_hy(
            _change(kind, old, new, target), user_id="local", derivation=derivation
        )

    assert (
        governance.get("hy-new" if kind != "revise" else target.memory_id, user_id="local")
        == before
    )


@pytest.mark.parametrize("kind", ["create", "revise", "link"])
def test_citing_only_new_evidence_still_writes_when_nothing_was_overruled(tmp_path, kind) -> None:
    """Completing evidence only adds checks, not blocks: when nobody overrules, things land as \
usual."""
    governance, old, new, target, _doomed = _written_evidence_scene(tmp_path)
    if kind == "link":
        governance.write_from_hy(
            HyCreate(
                draft=_draft(new, "The user works in Hangzhou."),
                request_id="seed",
                memory_id="hy-new",
            ),
            user_id="local",
            derivation=governance.hy_derivation((new.evidence_id,)),
        )
    derivation = governance.hy_derivation((old.evidence_id, new.evidence_id)).citing(
        (new.evidence_id,)
    )

    written = governance.write_from_hy(
        _change(kind, old, new, target), user_id="local", derivation=derivation
    )

    assert old.evidence_id in written.evidence_ids


# # ------------------------------------------------------------ HY's own deletions are not human
# # overruling
#
# The regressions showed: the pre-call gate treated any memory with status=deleted as human-deleted,
# so memories retired by HY itself made their source text held back in every later round. The
# write-transaction check already
# excludes by_model; the pre-call gate must use the same criterion.


@pytest.mark.parametrize("by", ["person", "hy"])
def test_only_the_persons_deletion_holds_evidence_back_from_the_next_call(tmp_path, by) -> None:
    from one_memory.memory_governance import HyRetire

    stores = _setup(tmp_path)
    evidence, _profiles, tasks, governance, _secrets, profile = stores
    record = _put(
        evidence,
        message_id="u1",
        role="user",
        written_by=WrittenBy.PERSON,
        content="I live in Hangzhou.",
    )
    memory = _manual(governance, record, "The user lives in Hangzhou.", "seed-memory")
    if by == "person":
        _delete(governance, memory.memory_id)
    else:
        governance.write_from_hy(
            HyRetire(
                memory_id=memory.memory_id,
                request_id="hy-retire",
                expected_revision=memory.revision,
            ),
            user_id="local",
            derivation=governance.hy_derivation((record.evidence_id,)),
        )
    assert governance.get(memory.memory_id, user_id="local").status is MemoryStatus.DELETED
    bridge = FakeBridge(
        outputs=(
            HyMemoryOutput(
                memory_id="hy-new", content="The user is living in Hangzhou.", layer="L0_BASIC_INFO"
            ),
        )
    )
    task = _task(tasks, profile, record)

    blocked = governance.blocked_source_ids(user_id="local", evidence_ids=(record.evidence_id,))
    finished = asyncio.run(_runner(tmp_path, stores, bridge).run(task.task_id))

    if by == "person":
        assert blocked == {record.evidence_id}
        assert (finished.status, finished.last_error_code) == (
            "cancelled",
            "hy_memory_evidence_changed",
        )
    else:
        assert blocked == set()
        assert (finished.status, finished.last_error_code) == ("completed", None)
