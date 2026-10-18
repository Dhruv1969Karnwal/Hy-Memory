"""R-09: the first-install onboarding records the profile without asking the user a word.

Onboarding used to end once the sources were done, never inducing a profile. The only
caller of `digest()` in the whole repo is that button in the memory space (`MemorySpace.tsx:265`), \
so after a first install
not one System 2 induced profile sits in the governance layer, and a fresh library knows
nothing about "who am I", even though the evidence is already imported.

Induction itself also drops half: `_record_schemas` only picks `L6_SCHEMA` nodes and
skips `L0_BASIC_INFO` outright. The `_read_graph_snapshot` query carries no layer filter,
so L0 nodes are already in the snapshot, fetched and then discarded. HY itself calls these two \
layers the profile together
(`_PROFILE_LAYERS = {"l0_basic_info", "l6_schema"}),
the basic info half stays behind in Kuzu, invisible locally.
"""

import asyncio
import sqlite3

from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from one_memory.api import create_app
from one_memory.config import Settings
from one_memory.hy_memory_bridge import HyGraphNode, HyGraphSnapshot, HyMemoryBridgeError
from one_memory.imports.models import (
    EvidenceContentKind,
    EvidenceInput,
    EvidenceOrigin,
    OriginMetadata,
    WrittenBy,
)
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_governance import (
    MemoryDraft,
    MemoryGovernanceStore,
    MemoryKind,
    MemoryStatus,
)
from one_memory.memory_processing import MemoryProcessingSupervisor
from one_memory.memory_tasks import MemoryTaskStore
from one_memory.model_profiles import (
    ModelCapabilities,
    ModelProfileInput,
    ModelProfileStore,
    ProviderKind,
)
from one_memory.secrets import InMemorySecretStore
from tests.person_door import as_person


class FakeGraph:
    """Induction only uses four methods; the rest are unimplemented, one fake less and one \
distortion less."""

    def __init__(self, graph_nodes: tuple[HyGraphNode, ...]) -> None:
        self.graph_nodes = graph_nodes
        self.graph_reads: list[tuple[str, str]] = []
        self.digest_calls: list[tuple[str, str]] = []

    async def digest(self, *, user_id: str, agent_id: str) -> dict[str, object]:
        self.digest_calls.append((user_id, agent_id))
        return {"success": True, "tasks_processed": 1}

    async def graph_snapshot(
        self,
        *,
        user_id: str,
        agent_id: str,
        limit: int = 300,
    ) -> HyGraphSnapshot:
        self.graph_reads.append((user_id, agent_id))
        return HyGraphSnapshot(nodes=self.graph_nodes, edges=())

    async def list_agent_ids(self, *, user_id: str, limit: int = 1_000) -> tuple[str, ...]:
        return ()

    async def close(self) -> None:
        return None


def _setup(tmp_path):
    database = tmp_path / "one-memory.sqlite3"
    secrets = InMemorySecretStore()
    evidence = EvidenceStore(database)
    profiles = ModelProfileStore(database)
    tasks = MemoryTaskStore(database)
    governance = MemoryGovernanceStore(database)
    profile = profiles.create(
        ModelProfileInput(
            name="Local",
            provider_kind=ProviderKind.OLLAMA,
            model="hy-bot-phi4",
            embedding_model="nomic-embed-text",
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
    profile = profiles.activate(profile.profile_id, secrets)
    return evidence, profiles, tasks, governance, secrets, profile


def _imported(evidence: EvidenceStore, *, workspace: str | None = "workspace:v1:project"):
    return evidence.put(
        EvidenceInput(
            source_id="cursor",
            workspace=workspace,
            conversation_id="conversation",
            message_id="m1",
            correlation_id="m1",
            content_kind=EvidenceContentKind.RAW_CONVERSATION,
            role="user",
            written_by=WrittenBy.PERSON,
            verbatim="A conversation imported in.",
            origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
        )
    ).record


def _confirmed_from(governance: MemoryGovernanceStore, evidence_id: str) -> None:
    """Induction only acts on "scopes that still have local evidence", so a confirmed memory must \
exist first."""
    as_person(governance).create(
        MemoryDraft(
            user_id="local",
            kind=MemoryKind.CHAT,
            content="An existing memory inside the project.",
            scopes=("workspace:v1:project",),
            evidence_ids=(evidence_id,),
        ),
        request_id="candidate-1",
    )


def _supervisor(tmp_path, stores, bridge) -> MemoryProcessingSupervisor:
    evidence, profiles, tasks, governance, secrets, profile = stores
    supervisor = MemoryProcessingSupervisor(
        tasks=tasks,
        profiles=profiles,
        secrets=secrets,
        evidence=evidence,
        governance=governance,
        data_dir=tmp_path,
    )
    supervisor._bridges[(profile.profile_id, profile.revision, None)] = bridge
    return supervisor


def test_the_basic_info_half_of_the_profile_is_written_down_too(tmp_path) -> None:
    """L0_BASIC_INFO comes back and is dropped again, so only half the profile lands.

    HY itself calls L0_basic_info and L6_schema the profile; both layers are in the snapshot,
    yet `_record_schemas` keeps only L6. Zero-question profile capture is precisely about this L0 \
half:
    facts like "the user is a product manager" only ever appear in the basic info layer.
    """
    stores = _setup(tmp_path)
    evidence, _profiles, _tasks, governance, _secrets, _profile = stores
    record = _imported(evidence)
    _confirmed_from(governance, record.evidence_id)
    bridge = FakeGraph(
        graph_nodes=(
            HyGraphNode(
                node_id="basic-1",
                content="The user is a product manager without a technical background.",
                layer="l0_basic_info",
            ),
            HyGraphNode(
                node_id="schema-1",
                content="The user prefers verifiable conclusions.",
                layer="l6_schema",
            ),
        ),
    )
    supervisor = _supervisor(tmp_path, stores, bridge)

    result = asyncio.run(supervisor.digest())

    profile_contents = {
        memory.content: memory
        for memory in governance.list(user_id="local")
        if memory.scopes == ("all-apps",)
    }
    assert "The user is a product manager without a technical background." in profile_contents
    assert "The user prefers verifiable conclusions." in profile_contents
    assert result["schemas_recorded"] == 2
    basic = profile_contents["The user is a product manager without a technical background."]
    assert basic.status is MemoryStatus.CONFIRMED
    # It must say why we believe it, the same bar as the L6 half.
    assert basic.evidence_ids == (record.evidence_id,)
    # kind uses the repo's existing layer-to-kind map; no new kind is invented for profiles
    # (the kind is part of the API contract, adding one needs a human sign-off).
    assert basic.kind is MemoryKind.CHAT


def test_a_layer_hy_does_not_call_profile_still_never_reaches_governance(tmp_path) -> None:
    """Green control, and the fake-fix guard for this item.

    Turning "only pick L6" into "bring back everything in the snapshot" would turn the test above \
green at the cost of
    whole raw conversations (L1_RAW) broadcast to every app as profiles. This one watches that.
    """
    stores = _setup(tmp_path)
    evidence, _profiles, _tasks, governance, _secrets, _profile = stores
    record = _imported(evidence)
    _confirmed_from(governance, record.evidence_id)
    bridge = FakeGraph(
        graph_nodes=tuple(
            HyGraphNode(node_id=f"node-{layer}", content=f"content of {layer}", layer=layer)
            for layer in ("l1_raw", "l2_fact", "l3_summary", "l4_identity", "l7_intention")
        ),
    )
    supervisor = _supervisor(tmp_path, stores, bridge)

    result = asyncio.run(supervisor.digest())

    assert result["schemas_recorded"] == 0
    assert [memory.content for memory in governance.list(user_id="local")] == [
        "An existing memory inside the project."
    ]


def test_the_same_basic_info_is_not_written_twice(tmp_path) -> None:
    """Green control: induction is triggered repeatedly (once on onboarding, anytime via the \
button); it must not pile up."""
    stores = _setup(tmp_path)
    evidence, _profiles, _tasks, governance, _secrets, _profile = stores
    record = _imported(evidence)
    _confirmed_from(governance, record.evidence_id)
    nodes = (
        HyGraphNode(
            node_id="basic-1",
            content="The user is a product manager without a technical background.",
            layer="l0_basic_info",
        ),
    )
    supervisor = _supervisor(tmp_path, stores, FakeGraph(graph_nodes=nodes))

    first = asyncio.run(supervisor.digest())
    second = asyncio.run(supervisor.digest())

    assert first["schemas_recorded"] == 1
    assert second["schemas_recorded"] == 0
    assert [
        memory.content
        for memory in governance.list(user_id="local")
        if memory.scopes == ("all-apps",)
    ] == ["The user is a product manager without a technical background."]


def _app(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        source_home=tmp_path / "home",
        api_token=SecretStr("test-token"),
    )
    return create_app(settings, source_catalog={})


def _client(app) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={
            "Authorization": "Bearer test-token",
            "Origin": "http://127.0.0.1:5173",
        },
    )


def test_finishing_onboarding_is_not_model_content_consent(tmp_path) -> None:
    """S1.3: completion and skip cannot authorize profile induction."""

    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.router.lifespan_context(app), _client(app) as client:
            calls: list[str] = []

            async def spy() -> dict[str, object]:
                calls.append("digest")
                return {"success": True, "schemas_recorded": 0}

            app.state.memory_processing_supervisor.digest = spy  # type: ignore[method-assign]

            # Just glancing at the onboarding state must not trigger induction.
            assert (await client.get("/api/v1/onboarding")).status_code == 200
            assert calls == []

            finished = await client.post("/api/v1/onboarding/complete")

            assert finished.json()["status"] == "completed"
            assert calls == []

    asyncio.run(scenario())


def test_the_way_back_to_the_first_run_deletes_nothing(tmp_path) -> None:
    """Reset only moves the completion flag; not one memory is lost.

    Keeping the completion state and the memories in one library is deliberate: reinstalling the app
    must not treat someone who already has memories as a new user. The price is that reinstalling \
does not show the cold start again. This watches that exit:
    the state goes back to first run and nothing else moves.
    """

    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.router.lifespan_context(app), _client(app) as client:
            database = tmp_path / "data" / "one-memory.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "INSERT INTO governed_memories "
                    "(memory_id,user_id,kind,status,content,scopes_json,"
                    "revision,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        "keep-me",
                        "local",
                        "chat",
                        "confirmed",
                        "memory must be kept",
                        '["all-apps"]',
                        1,
                        "2026-10-17T00:00:00+00:00",
                        "2026-10-17T00:00:00+00:00",
                    ),
                )
            finished = await client.post("/api/v1/onboarding/complete")

            assert finished.json()["status"] == "completed"

            reset = await client.post("/api/v1/onboarding/reset")

            assert reset.status_code == 200
            assert reset.json()["status"] == "not_started"
            assert reset.json()["completed_at"] is None
            # It really landed on disk, not just in this response: the next startup also reads first
            # run.
            assert (await client.get("/api/v1/onboarding")).json()["status"] == "not_started"
            with sqlite3.connect(database) as connection:
                assert connection.execute(
                    "SELECT content,revision FROM governed_memories WHERE memory_id='keep-me'"
                ).fetchone() == ("memory must be kept", 1)

    asyncio.run(scenario())


def test_resetting_before_anything_was_completed_is_not_an_error(tmp_path) -> None:
    """Green control: with no such row yet in a fresh library, resetting must not blow up."""

    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.router.lifespan_context(app), _client(app) as client:
            reset = await client.post("/api/v1/onboarding/reset")

            assert reset.status_code == 200
            assert reset.json()["status"] == "not_started"

    asyncio.run(scenario())


def test_hy_being_down_does_not_leave_the_first_install_unfinished(tmp_path) -> None:
    """Guard: this step is skippable, an induction failure must not wedge onboarding.

    On a first install the model profile may not be configured yet, and general outbound \
networking may be unavailable too.
    An induction failure should only cost one profile, never leave the install unfinished.
    """

    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.router.lifespan_context(app), _client(app) as client:

            async def unavailable() -> dict[str, object]:
                raise HyMemoryBridgeError("hy_memory_unavailable")

            app.state.memory_processing_supervisor.digest = unavailable  # type: ignore[method-assign]

            finished = await client.post("/api/v1/onboarding/complete")

            assert finished.status_code == 200
            assert finished.json()["status"] == "completed"
            assert finished.json()["completed_at"] is not None
            # The state really persisted, it is not just in this response.
            assert (await client.get("/api/v1/onboarding")).json()["status"] == "completed"

    asyncio.run(scenario())
