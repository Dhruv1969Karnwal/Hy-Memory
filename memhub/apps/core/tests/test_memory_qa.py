import asyncio
import json

import httpx
import pytest

from one_memory.imports.models import EvidenceInput, EvidenceOrigin, OriginMetadata, WrittenBy
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_governance import MemoryDraft, MemoryGovernanceStore, MemoryKind
from one_memory.memory_qa import MemoryAnswerer, MemoryAnswerError
from one_memory.model_profiles import (
    ModelCapabilities,
    ModelProfileInput,
    ModelProfileStore,
    ProviderKind,
)
from one_memory.secrets import InMemorySecretStore
from tests.person_door import as_person


@pytest.mark.parametrize(
    "stop_term",
    tuple(
        "\u6211\u4f60\u4ed6\u5979\u5b83\u7684\u4e86\u662f\u5728\u548c\u4e0e\u6216\u8fd9\u90a3\u4e5f\u90fd\u5f88\u6ca1\u6709\u5c31\u8fd8\u800c\u4f46\u5374\u56e0\u4e14\u628a\u88ab\u4ece\u7ed9\u5230\u53c8\u8457\u7740\u8fc7\u554a\u5462\u5417\u5427"
    ),
)
def test_search_terms_keep_meaningful_single_chinese_character(stop_term) -> None:
    from one_memory.memory_qa import _search_terms

    assert _search_terms("\u5f20") == {"\u5f20"}
    assert _search_terms(stop_term) == set()
    assert _search_terms(f"{stop_term} \u5f20") == {"\u5f20"}


def _grounded_setup(tmp_path, *, memory_id: str = "memory-real-data"):
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    profiles = ModelProfileStore(database)
    secrets = InMemorySecretStore()
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            correlation_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="Every dashboard must use real data, never mock data.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    as_person(governance).create(
        MemoryDraft(
            user_id="local",
            kind=MemoryKind.WORKING,
            content=(
                "The user requires the product dashboards to use real data and never mock data."
            ),
            scopes=("all-apps",),
            evidence_ids=(evidence.evidence_id,),
        ),
        request_id="create-memory",
        memory_id=memory_id,
    )
    profile = profiles.create(
        ModelProfileInput(
            name="Local",
            provider_kind=ProviderKind.OLLAMA,
            base_url="http://127.0.0.1:11434",
            model="test-model",
            embedding_model="test-embedding",
        ),
        secrets,
    )
    profiles.record_capabilities(
        profile.profile_id,
        expected_revision=profile.revision,
        capabilities=ModelCapabilities(
            endpoint_reachable=True,
            model_available=True,
            structured_output=True,
            embedding_available=True,
        ),
        secrets=secrets,
    )
    profiles.activate(profile.profile_id, secrets)
    return governance, profiles, secrets


def _ndjson_stream(chunks: tuple[str, ...]) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["think"] is False
        body = "".join(
            json.dumps({"message": {"content": chunk}}, ensure_ascii=False) + "\n"
            for chunk in chunks
        )
        return httpx.Response(200, text=body)

    return httpx.MockTransport(respond)


def _collect(stream) -> list[dict]:
    async def drain() -> list[dict]:
        return [event async for event in stream]

    return asyncio.run(drain())


def test_streamed_answer_emits_recall_then_incremental_grounded_text(tmp_path) -> None:
    governance, profiles, secrets = _grounded_setup(tmp_path)
    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        transport=_ndjson_stream(
            (
                '{"citations": ["memory-real-data"], ',
                '"answer": "You asked that the dashboards',
                ' show real data only."}',
            )
        ),
    )

    events = _collect(answerer.answer_stream("What is my requirement about mock data?"))

    assert events[0]["type"] == "recall"
    assert [memory.memory_id for memory in events[0]["memories"]] == ["memory-real-data"]
    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == ["You asked that the dashboards", " show real data only."]
    final = events[-1]
    assert final["type"] == "final"
    assert final["mode"] == "generated"
    assert final["answer"] == "You asked that the dashboards show real data only."
    assert [memory.memory_id for memory in final["memories"]] == ["memory-real-data"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("change", ["set_scopes", "correct", "delete"])
def test_governance_change_during_generation_stops_the_old_answer(tmp_path, streaming, change):
    governance, profiles, secrets = _grounded_setup(tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        memory = governance.get("memory-real-data", user_id="local")
        kwargs = {
            "user_id": "local",
            "request_id": "during-model",
            "expected_revision": memory.revision,
        }
        if change == "set_scopes":
            as_person(governance).set_scopes(memory.memory_id, scopes=("private",), **kwargs)
        elif change == "correct":
            as_person(governance).correct(
                memory.memory_id, content="The current decision is different", **kwargs
            )
        else:
            as_person(governance).delete(memory.memory_id, **kwargs)
        content = json.dumps({"citations": ["memory-real-data"], "answer": "stale answer"})
        return httpx.Response(200, json={"message": {"content": content}})

    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        transport=httpx.MockTransport(respond),
    )
    events = []

    async def run():
        if streaming:
            async for event in answerer.answer_stream("What is my requirement about mock data?"):
                events.append(event)
        else:
            await answerer.answer("What is my requirement about mock data?")

    with pytest.raises(MemoryAnswerError, match="memory_answer_context_changed"):
        asyncio.run(run())
    assert all(event["type"] == "recall" for event in events)


def test_streamed_deltas_always_reconstruct_the_final_answer(tmp_path) -> None:
    governance, profiles, secrets = _grounded_setup(tmp_path)

    def stream_for(chunks: tuple[str, ...]) -> list[dict]:
        answerer = MemoryAnswerer(
            governance=governance,
            profiles=profiles,
            secrets=secrets,
            transport=_ndjson_stream(chunks),
        )
        return _collect(answerer.answer_stream("What is my requirement about mock data?"))

    grounded = stream_for(
        ('{"citations": ["memory-real-data"], "answer": "You asked for real data."}',)
    )
    uncited = stream_for(('{"citations": [], "answer": "Answer invented by the model"}',))

    for events in (grounded, uncited):
        final = events[-1]
        streamed = "".join(event["text"] for event in events if event["type"] == "delta")
        assert streamed == final["answer"]

    # The uncited answer text must never have reached the reader.
    assert "Answer invented by the model" not in "".join(
        event["text"] for event in uncited if event["type"] == "delta"
    )
    assert uncited[-1]["mode"] == "empty"


def test_streamed_answer_withholds_text_when_citation_is_not_retrieved(tmp_path) -> None:
    governance, profiles, secrets = _grounded_setup(tmp_path)
    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        transport=_ndjson_stream(
            (
                '{"citations": ["workspace-memory"], ',
                '"answer": "out-of-scope answer"}',
            )
        ),
    )

    events: list[dict] = []

    async def drain() -> None:
        async for event in answerer.answer_stream("What is my requirement about mock data?"):
            events.append(event)

    with pytest.raises(MemoryAnswerError, match="memory_answer_invalid_citation"):
        asyncio.run(drain())

    assert [event["type"] for event in events] == ["recall"]


def test_answer_uses_only_grounded_confirmed_memory_citations(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    profiles = ModelProfileStore(database)
    secrets = InMemorySecretStore()
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            correlation_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="Every dashboard must use real data, never mock data.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    as_person(governance).create(
        MemoryDraft(
            user_id="local",
            kind=MemoryKind.WORKING,
            content=(
                "The user requires the product dashboards to use real data and never mock data."
            ),
            scopes=("all-apps",),
            evidence_ids=(evidence.evidence_id,),
        ),
        request_id="create-memory",
        memory_id="memory-real-data",
    )
    profile = profiles.create(
        ModelProfileInput(
            name="Local",
            provider_kind=ProviderKind.OLLAMA,
            base_url="http://127.0.0.1:11434",
            model="test-model",
            embedding_model="test-embedding",
        ),
        secrets,
    )
    profiles.record_capabilities(
        profile.profile_id,
        expected_revision=profile.revision,
        capabilities=ModelCapabilities(
            endpoint_reachable=True,
            model_available=True,
            structured_output=True,
            embedding_available=True,
        ),
        secrets=secrets,
    )
    profiles.activate(profile.profile_id, secrets)

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "must address the user as 'you'" in body["messages"][0]["content"]
        assert body["messages"][1]["content"].find("memory-real-data") >= 0
        question = json.loads(body["messages"][1]["content"])["question"]
        model_answer = (
            {
                "answer": "The available memories are not enough to answer this question.",
                "citations": [],
            }
            if question == "\u6211\u4e0b\u5468\u7684\u822a\u73ed\u662f\u4ec0\u4e48\uff1f"
            else {
                "answer": "You asked that the dashboards show real data only and never mock data.",
                "citations": ["memory-real-data"],
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                model_answer,
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    async def semantic_search(query: str) -> tuple[str, ...]:
        return () if "\u822a\u73ed" in query else ("memory-real-data",)

    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        semantic_search=semantic_search,
        transport=httpx.MockTransport(respond),
    )

    result = asyncio.run(answerer.answer("What principles of mine are non-negotiable?"))

    assert result.answer == "You asked that the dashboards show real data only and never mock data."
    assert [item.memory_id for item in result.memories] == ["memory-real-data"]
    assert result.mode == "generated"

    refused = asyncio.run(
        answerer.answer("\u6211\u4e0b\u5468\u7684\u822a\u73ed\u662f\u4ec0\u4e48\uff1f")
    )

    assert refused.answer == "The available memories are not enough to answer this question."
    assert refused.memories == ()
    assert refused.mode == "empty"


def test_answer_refuses_question_without_matching_confirmed_memory(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    answerer = MemoryAnswerer(
        governance=MemoryGovernanceStore(database),
        profiles=ModelProfileStore(database),
        secrets=InMemorySecretStore(),
    )

    result = asyncio.run(
        answerer.answer("\u6211\u4e0b\u5468\u7684\u822a\u73ed\u662f\u4ec0\u4e48\uff1f")
    )

    assert result.answer == "The available memories are not enough to answer this question."
    assert result.memories == ()
    assert result.mode == "empty"


def test_answer_does_not_turn_a_failed_semantic_lane_into_no_grounding(tmp_path) -> None:
    governance, profiles, secrets = _grounded_setup(tmp_path)

    async def unavailable(_query: str) -> tuple[str, ...]:
        raise RuntimeError("semantic backend unavailable")

    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        semantic_search=unavailable,
    )

    with pytest.raises(MemoryAnswerError, match="memory_answer_semantic_unavailable"):
        asyncio.run(answerer.answer("\u6211\u4e0b\u5468\u7684\u822a\u73ed\u662f\u4ec0\u4e48\uff1f"))


def test_answer_keeps_lexical_hits_but_marks_failed_semantic_coverage(tmp_path) -> None:
    governance, profiles, secrets = _grounded_setup(tmp_path)

    async def unavailable(_query: str) -> tuple[str, ...]:
        raise RuntimeError("semantic backend unavailable")

    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        semantic_search=unavailable,
        transport=_ndjson_stream(
            (
                '{"citations": ["memory-real-data"], ',
                '"answer": "You asked that the dashboards show real data only."}',
            )
        ),
    )

    events = _collect(answerer.answer_stream("What is my requirement about mock data?"))
    final = events[-1]
    assert final["mode"] == "generated"
    assert final["retrieval"]["semantic_state"] == "failed"


def test_default_qa_scope_excludes_confirmed_workspace_memory(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            workspace="workspace:v1:project",
            conversation_id="conversation",
            message_id="message",
            correlation_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="The project decided to use a local-first architecture.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    as_person(governance).create(
        MemoryDraft(
            user_id="local",
            kind=MemoryKind.WORKING,
            content=evidence.verbatim,
            scopes=("workspace:v1:project",),
            evidence_ids=(evidence.evidence_id,),
        ),
        request_id="create-workspace-memory",
    )
    answerer = MemoryAnswerer(
        governance=governance,
        profiles=ModelProfileStore(database),
        secrets=InMemorySecretStore(),
    )

    result = asyncio.run(answerer.answer("What architecture does the project use?"))

    assert result.answer == "The available memories are not enough to answer this question."
    assert result.memories == ()
    assert result.mode == "empty"


def test_answer_rejects_model_citation_outside_retrieved_scope(tmp_path) -> None:
    database = tmp_path / "one-memory.sqlite3"
    evidence_store = EvidenceStore(database)
    governance = MemoryGovernanceStore(database)
    profiles = ModelProfileStore(database)
    secrets = InMemorySecretStore()
    evidence = evidence_store.put(
        EvidenceInput(
            source_id="cursor",
            conversation_id="conversation",
            message_id="message",
            correlation_id="message",
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="I prefer concise answers.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record
    as_person(governance).create(
        MemoryDraft(
            user_id="local",
            kind=MemoryKind.CHAT,
            content="The user prefers concise answers.",
            scopes=("all-apps",),
            evidence_ids=(evidence.evidence_id,),
        ),
        request_id="create",
        memory_id="allowed-memory",
    )
    profile = profiles.create(
        ModelProfileInput(
            name="Local",
            provider_kind=ProviderKind.OLLAMA,
            base_url="http://127.0.0.1:11434",
            model="test-model",
            embedding_model="test-embedding",
        ),
        secrets,
    )
    profiles.record_capabilities(
        profile.profile_id,
        expected_revision=profile.revision,
        capabilities=ModelCapabilities(
            endpoint_reachable=True,
            model_available=True,
            structured_output=True,
            embedding_available=True,
        ),
        secrets=secrets,
    )
    profiles.activate(profile.profile_id, secrets)

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "answer": "out-of-scope answer",
                            "citations": ["workspace-memory"],
                        }
                    )
                }
            },
        )

    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        semantic_search=lambda _query: asyncio.sleep(
            0,
            result=("allowed-memory",),
        ),
        transport=httpx.MockTransport(respond),
    )

    with pytest.raises(MemoryAnswerError, match="memory_answer_invalid_citation"):
        asyncio.run(answerer.answer("What are my answering preferences?"))


def _compatible_gateway_setup(tmp_path):
    """A generator that works but does not accept response_format.

    Must be openai_compatible. The first version of this fixture reused the Ollama card from
    _grounded_setup, and the Ollama branch calls the field format rather than response_format, so
    'no response_format was sent' was vacuously true -- the test never reached the code under test.
    Mutation testing caught it. The assert below is what makes that premise speak for itself.
    """
    governance, profiles, secrets = _grounded_setup(tmp_path)
    profile = profiles.create(
        ModelProfileInput(
            name="Compatible provider",
            provider_kind=ProviderKind.OPENAI_COMPATIBLE,
            base_url="https://api.example.invalid/standard/v1",
            model="deepseek-v4-pro",
            embedding_model="text-embedding",
            api_key="probe-key",
        ),
        secrets,
    )
    profiles.record_capabilities(
        profile.profile_id,
        expected_revision=profile.revision,
        capabilities=ModelCapabilities(
            endpoint_reachable=True,
            model_available=True,
            structured_output=False,
            embedding_available=True,
        ),
        secrets=secrets,
    )
    profiles.activate(profile.profile_id, secrets)
    active = profiles.active(secrets)
    assert active.provider_kind is ProviderKind.OPENAI_COMPATIBLE
    return governance, profiles, secrets


def test_a_model_that_cannot_be_schema_constrained_still_answers(tmp_path) -> None:
    """Not being able to constrain the schema does not mean the model cannot answer.

    A compatible gateway may answer response_format with HTTP 400, while the model still returns
    valid JSON without the parameter. Treating 'cannot be constrained' as 'unusable' here meant \
that on
    such gateways 'ask memory' only ever quoted the source text and never got a generated answer \
-- even though the model could answer.
    """
    governance, profiles, secrets = _compatible_gateway_setup(tmp_path)
    sent: list[bool] = []
    urls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        sent.append(b"response_format" in request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "citations": ["memory-real-data"],
                                    "answer": "You asked that the dashboards show real data only.",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        transport=httpx.MockTransport(respond),
    )
    result = asyncio.run(answerer.answer("What data do the dashboards use?"))

    # It was called, on chat/completions, and without the parameter that gateway rejects.
    # All three are asserted: checking only the last one lets a card on another branch keep it
    # vacuously true.
    assert urls == ["https://api.example.invalid/standard/v1/chat/completions"]
    assert sent == [False]
    assert result.mode == "generated"
    assert "real data" in result.answer


def test_no_model_at_all_still_falls_back_to_quoting(tmp_path) -> None:
    """Positive control: with no usable model at all it still only quotes the source text.

    Without this, the test above could be faked by always sending a generation request.
    """
    governance, profiles, secrets = _grounded_setup(tmp_path)
    profile = profiles.active(secrets)
    profiles.record_capabilities(
        profile.profile_id,
        expected_revision=profile.revision,
        capabilities=ModelCapabilities(
            endpoint_reachable=True,
            model_available=False,
            structured_output=False,
            embedding_available=True,
        ),
        secrets=secrets,
    )

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no generation request may be sent when no model is available")

    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        transport=httpx.MockTransport(refuse),
    )
    result = asyncio.run(answerer.answer("What data do the dashboards use?"))

    assert result.mode == "extractive"


def test_json_wrapped_in_a_code_fence_is_still_read(tmp_path) -> None:
    """Without a schema constraint the model very often wraps the JSON in ```.

    A strict json.loads would call that invalid output, so a model that can answer gets nothing.
    """
    governance, profiles, secrets = _compatible_gateway_setup(tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        inner = json.dumps(
            {
                "citations": ["memory-real-data"],
                "answer": "You asked that the dashboards show real data only.",
            },
            ensure_ascii=False,
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"```json\n{inner}\n```"}}]},
        )

    answerer = MemoryAnswerer(
        governance=governance,
        profiles=profiles,
        secrets=secrets,
        transport=httpx.MockTransport(respond),
    )
    result = asyncio.run(answerer.answer("What data do the dashboards use?"))

    assert result.mode == "generated"
    assert "real data" in result.answer
