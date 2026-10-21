import asyncio
import hashlib

import pytest

from one_memory.imports.models import EvidenceInput, EvidenceOrigin, OriginMetadata, WrittenBy
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_governance import MemoryDraft, MemoryGovernanceStore, MemoryKind
from one_memory.recall import GovernedRecallTarget, RecallRequest, RecallState
from tests.person_door import as_person


def _confirmed_memory(governance, evidence, *, content: str, scope: str):
    request_key = hashlib.sha256(f"{scope}:{content}".encode()).hexdigest()
    memory = as_person(governance).create(
        MemoryDraft(
            user_id="local",
            kind=MemoryKind.CHAT,
            content=content,
            scopes=(scope,),
            evidence_ids=(evidence.evidence_id,),
        ),
        request_id=f"create:{request_key}",
    )
    return memory


def test_recall_is_confirmed_only_grounded_and_scope_bound(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="I am a product manager, and the project uses a local-first architecture.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    personal = _confirmed_memory(
        governance,
        evidence,
        content="The user is a product manager",
        scope="all-apps",
    )
    _confirmed_memory(
        governance,
        evidence,
        content="The project uses a local-first architecture",
        scope="workspace:v1:project",
    )
    target = GovernedRecallTarget(governance)

    result = target.recall(RecallRequest(query="product manager"))

    assert result.state is RecallState.GROUNDED
    assert [hit.memory_id for hit in result.hits] == [personal.memory_id]
    assert result.hits[0].evidence_ids == (evidence.evidence_id,)
    assert all("workspace:v1:project" not in hit.scopes for hit in result.hits)

    workspace = target.recall(
        RecallRequest(
            query="What architecture does the project use?",
            scope="workspace:v1:project",
        )
    )
    assert [hit.content for hit in workspace.hits] == [
        "The project uses a local-first architecture"
    ]


def test_wildcard_scope_means_every_scope_just_like_it_does_in_qa(tmp_path) -> None:
    """`*` must mean the same thing on both read paths.

    Hosts recall through this path while the in-app answerer uses its own, and
    there `*` is a wildcard. Treating `*` here as a literal scope name matches
    nothing, so a host asking across scopes gets a silent empty result rather
    than the memory it was promised.
    """
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="Login authentication design discussion: weighing option A, option B and"
            " option C.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    project = _confirmed_memory(
        governance,
        evidence,
        content="Login authentication design discussion: weighing option A, option B and option C",
        scope="workspace:v1:project",
    )
    target = GovernedRecallTarget(governance)

    # The default personal scope must still not see project memories.
    assert (
        target.recall(RecallRequest(query="login authentication design")).state is RecallState.EMPTY
    )

    across = target.recall(RecallRequest(query="login authentication design", scope="*"))

    assert across.state is RecallState.GROUNDED
    assert [hit.memory_id for hit in across.hits] == [project.memory_id]


def test_recall_returns_empty_instead_of_unrelated_fallback(tmp_path) -> None:
    target = GovernedRecallTarget(MemoryGovernanceStore(tmp_path / "one-memory.sqlite3"))

    result = target.recall(RecallRequest(query="next week's flight"))

    assert result.state is RecallState.EMPTY
    assert result.hits == ()


def test_recall_finds_a_single_chinese_character_query(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="\u5f20 \u7684\u504f\u597d",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    memory = _confirmed_memory(
        governance,
        evidence,
        content="\u5f20 \u7684\u504f\u597d",
        scope="all-apps",
    )

    result = GovernedRecallTarget(governance).recall(RecallRequest(query="\u5f20"))
    stopped = GovernedRecallTarget(governance).recall(RecallRequest(query="\u6211"))

    assert result.state is RecallState.GROUNDED
    assert [hit.memory_id for hit in result.hits] == [memory.memory_id]
    assert stopped.state is RecallState.EMPTY
    assert stopped.hits == ()


def test_recall_stop_term_does_not_query_governance(tmp_path, monkeypatch) -> None:
    governance = MemoryGovernanceStore(tmp_path / "one-memory.sqlite3")

    def forbidden(*args, **kwargs):
        pytest.fail("Stop-term-only recall must not query governed memory storage")

    monkeypatch.setattr(governance, "find_confirmed_matching_terms", forbidden)

    result = GovernedRecallTarget(governance).recall(RecallRequest(query="\u6211"))

    assert result.state is RecallState.EMPTY
    assert result.hits == ()


def test_recall_rejects_single_generic_term_overlap(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="Hand the plan to WorkBuddy for a design review and optimization.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    _confirmed_memory(
        governance,
        evidence,
        content="Hand the plan to WorkBuddy for a design review and optimization",
        scope="workspace:v1:project",
    )

    result = GovernedRecallTarget(governance).recall(
        RecallRequest(
            query="What are this project's legal and open source license plans?",
            scope="workspace:v1:project",
        )
    )

    assert result.state is RecallState.EMPTY


def test_recall_uses_semantic_ids_when_words_do_not_overlap(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="I am a product manager.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    personal = _confirmed_memory(
        governance,
        evidence,
        content="The user is a product manager",
        scope="all-apps",
    )

    async def semantic_search(_: str) -> tuple[str, ...]:
        return (personal.memory_id,)

    target = GovernedRecallTarget(governance, semantic_search=semantic_search)
    result = asyncio.run(target.recall_async(RecallRequest(query="what is my job?")))

    assert [hit.memory_id for hit in result.hits] == [personal.memory_id]
    assert result.hits[0].score == 1


def test_recall_skips_oversized_hit_and_keeps_smaller_matches(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="Product managers prefer clear communication",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    small = _confirmed_memory(
        governance,
        evidence,
        content="Product managers prefer clear communication",
        scope="all-apps",
    )
    _confirmed_memory(
        governance,
        evidence,
        content="Product managers prefer clear communication" + ("x" * 250),
        scope="all-apps",
    )

    result = GovernedRecallTarget(governance).recall(
        RecallRequest(query="product manager", character_budget=200)
    )

    assert [hit.memory_id for hit in result.hits] == [small.memory_id]
    assert result.truncated is True


def test_recall_async_keeps_strong_semantic_when_weak_lexical_already_hits(tmp_path) -> None:
    """A weak substring hit must not skip the semantic channel.

    Chinese bigrams make everyday phrasing look grounded: "how to pick an open source license"
    overlaps "open source" and "license" on a recent review note, while the actual MIT
    decision has no literal overlap. Returning at the first non-empty lexical
    page would make that decision unreachable.
    """
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="Open source license discussion.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    strong = _confirmed_memory(
        governance,
        evidence,
        content="The user decided One Memory will ship under the MIT license.",
        scope="all-apps",
    )
    weak = _confirmed_memory(
        governance,
        evidence,
        content="This open source project's license plan is still under review, not final.",
        scope="all-apps",
    )
    searched: list[str] = []

    async def semantic_search(query: str) -> tuple[str, ...]:
        searched.append(query)
        return (strong.memory_id,)

    result = asyncio.run(
        GovernedRecallTarget(governance, semantic_search=semantic_search).recall_async(
            RecallRequest(query="how to pick an open source license")
        )
    )

    assert searched == ["how to pick an open source license"]
    ids = [hit.memory_id for hit in result.hits]
    assert strong.memory_id in ids
    assert weak.memory_id in ids
    assert ids[0] == strong.memory_id


def test_recall_async_keeps_lexical_hits_when_semantic_times_out(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="Open source license discussion.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    strong = _confirmed_memory(
        governance,
        evidence,
        content="The user decided One Memory will ship under the MIT license.",
        scope="all-apps",
    )
    weak = _confirmed_memory(
        governance,
        evidence,
        content="This open source project's license plan is still under review, not final.",
        scope="all-apps",
    )

    started = False

    async def semantic_search(_query: str) -> tuple[str, ...]:
        nonlocal started
        started = True
        await asyncio.sleep(4)
        return (strong.memory_id,)

    result = asyncio.run(
        GovernedRecallTarget(governance, semantic_search=semantic_search).recall_async(
            RecallRequest(query="how to pick an open source license")
        )
    )

    assert started is True
    ids = [hit.memory_id for hit in result.hits]
    assert ids == [weak.memory_id]
    assert strong.memory_id not in ids
