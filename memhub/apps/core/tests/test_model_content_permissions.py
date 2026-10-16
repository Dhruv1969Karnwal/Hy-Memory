"""Content permission tests use only a temporary database and synthetic sources."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from one_memory.imports.grants import SourceGrantStore
from one_memory.imports.models import (
    EvidenceInput,
    EvidenceOrigin,
    EvidenceRecord,
    OriginMetadata,
    WrittenBy,
)
from one_memory.imports.sources.registry import RootCandidateKind
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_server.config_state import ConfigStore
from one_memory.model_content_permissions import (
    ModelContentCostEstimate,
    ModelContentPermissionError,
    ModelContentPermissionReceipt,
    ModelContentPermissionSnapshot,
    ModelContentPermissionStore,
)
from one_memory.model_profiles import (
    ModelCapabilities,
    ModelProfileInput,
    ModelProfilePatch,
    ModelProfileSnapshot,
    ModelProfileStore,
    ProviderKind,
)
from one_memory.secrets import InMemorySecretStore

USER = "local"
SOURCE = "claude-code"
MANUAL = "one-memory-manual"


def evidence_value(
    message: str,
    *,
    source_id: str = SOURCE,
    user_id: str = USER,
    text: str | None = None,
) -> EvidenceInput:
    return EvidenceInput(
        user_id=user_id,
        source_id=source_id,
        conversation_id="synthetic-conversation",
        message_id=message,
        role="user",
        written_by=WrittenBy.PERSON,
        verbatim=text or f"Synthetic content {message}",
        origin_meta={EvidenceOrigin.MANUAL_IMPORT: OriginMetadata()},
    )


@dataclass
class Environment:
    database: Path
    root: Path
    sources: SourceGrantStore
    evidence: EvidenceStore
    profiles: ModelProfileStore
    profile: ModelProfileSnapshot
    permissions: ModelContentPermissionStore
    record: EvidenceRecord
    secrets: InMemorySecretStore

    def snapshot(
        self,
        *,
        records: tuple[EvidenceRecord, ...] | None = None,
        source_ids: tuple[str, ...] = (SOURCE,),
        future: bool = False,
    ) -> ModelContentPermissionSnapshot:
        return self.permissions.snapshot(
            user_id=USER,
            source_ids=source_ids,
            profile=self.profile,
            evidence=records if records is not None else (self.record,),
            include_future_additions=future,
        )

    def grant(self, *, future: bool = False) -> ModelContentPermissionReceipt:
        return self.permissions.grant(
            self.snapshot(future=future), user_confirmed=True, expected_revision=0
        )

    def check(
        self,
        receipt: ModelContentPermissionReceipt,
        *,
        records: tuple[EvidenceRecord, ...] | None = None,
        profile: ModelProfileSnapshot | None = None,
        user_id: str = USER,
    ) -> ModelContentPermissionReceipt:
        return self.permissions.check(
            user_id=user_id,
            expected_revision=receipt.revision,
            expected_revocation_revision=receipt.revocation_revision,
            profile=profile or self.profile,
            evidence=records if records is not None else (self.record,),
        )


@pytest.fixture
def env(tmp_path: Path) -> Environment:
    database = tmp_path / "one-memory.sqlite3"
    home = tmp_path / "home"
    root = home / ".claude" / "projects"
    root.mkdir(parents=True)
    sources = SourceGrantStore(database, home=home)
    sources.grant(
        user_id=USER, source_id=SOURCE, root=root, candidate_kind=RootCandidateKind.DEFAULT
    )
    evidence = EvidenceStore(database)
    record = evidence.put(evidence_value("selected")).record
    profiles = ModelProfileStore(database)
    secrets = InMemorySecretStore()
    profile = profiles.create(
        ModelProfileInput(
            name="Synthetic local model",
            provider_kind=ProviderKind.OLLAMA,
            base_url="http://127.0.0.1:11434/",
            model="test-generation",
            embedding_provider_kind=ProviderKind.OLLAMA,
            embedding_base_url="http://127.0.0.1:11435/",
            embedding_model="test-embedding",
        ),
        secrets,
    )
    profiles.record_capabilities(
        profile.profile_id,
        expected_revision=profile.revision,
        capabilities=ModelCapabilities(
            endpoint_reachable=True, model_available=True, embedding_available=True
        ),
        secrets=secrets,
    )
    profiles.activate(profile.profile_id, secrets)
    return Environment(
        database,
        root,
        sources,
        evidence,
        profiles,
        profiles.snapshot(profile.profile_id, profile.revision),
        ModelContentPermissionStore(database),
        record,
        secrets,
    )


def test_model_selection_and_source_grant_are_not_content_consent(env: Environment) -> None:
    assert env.permissions.receipt(user_id=USER) is None


def test_cost_disclosure_is_explicit_unknown_without_a_pinned_price_source(
    env: Environment,
) -> None:
    snapshot = env.snapshot()
    assert snapshot.cost_estimate.status == "unknown"
    assert snapshot.cost_estimate.currency is None
    assert snapshot.cost_estimate.amount is None
    assert snapshot.cost_estimate.price_source is None
    with pytest.raises(ValidationError):
        ModelContentCostEstimate(status="estimated", currency="USD", amount=1.0)
    with pytest.raises(ValidationError):
        ModelContentCostEstimate(status="unknown", amount=0.0)
    estimate = ModelContentCostEstimate(
        status="estimated",
        currency="USD",
        amount=0.25,
        price_source="https://example.invalid/pinned-price",
    )
    assert estimate.amount == 0.25
    with pytest.raises(ModelContentPermissionError, match="permission_required"):
        env.permissions.check(
            user_id=USER,
            expected_revision=0,
            expected_revocation_revision=0,
            profile=env.profile,
            evidence=(env.record,),
        )
    snapshot = env.snapshot()
    for confirmation in (False, 1, "true", None):
        with pytest.raises(ModelContentPermissionError, match="confirmation_required"):
            env.permissions.grant(snapshot, user_confirmed=confirmation, expected_revision=0)
    assert env.permissions.receipt(user_id=USER) is None


def test_snapshot_is_immutable_precise_and_durable_in_the_same_database(env: Environment) -> None:
    receipt = env.grant()
    assert receipt.revision == 1 and receipt.revocation_revision == 0
    assert receipt.revoked_at is None
    assert env.check(receipt) == receipt
    scope = receipt.snapshot
    assert scope.profile == env.profile
    assert scope.generation.endpoint == "http://127.0.0.1:11434/v1"
    assert scope.embedding.endpoint == "http://127.0.0.1:11435/v1"
    assert scope.generation.model == "test-generation"
    assert scope.embedding.model == "test-embedding"
    source = env.sources.require_receipt(user_id=USER, source_id=SOURCE)
    assert scope.sources[0].granted_at == source.granted_at
    assert scope.sources[0].revision == source.revision
    assert scope.sources[0].root_identity == source.root_id
    assert scope.evidence[0].evidence_id == env.record.evidence_id
    assert scope.evidence[0].revision == env.record.revision
    assert scope.evidence[0].content_hash == env.record.content_hash
    assert scope.evidence[0].identity_hash
    assert env.record.verbatim not in receipt.model_dump_json()
    with pytest.raises(ValidationError):
        scope.evidence[0].revision = 999
    with pytest.raises(ValidationError):
        scope.sources[0].root_identity = "replacement"
    with pytest.raises(ValidationError):
        scope.profile.model = "replacement"
    env.permissions = ModelContentPermissionStore(env.database)
    assert env.permissions.receipt(user_id=USER) == receipt
    assert env.check(receipt) == receipt
    with sqlite3.connect(env.database) as db:
        assert db.execute("SELECT count(*) FROM model_content_permissions").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM evidence").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM source_grants").fetchone()[0] == 1


def test_revoke_and_regrant_have_distinct_monotonic_revisions(env: Environment) -> None:
    first = env.grant()
    with pytest.raises(ModelContentPermissionError, match="confirmation_required"):
        env.permissions.revoke(user_id=USER, expected_revision=1, user_confirmed=False)
    revoked = env.permissions.revoke(user_id=USER, expected_revision=1, user_confirmed=True)
    assert revoked.revision == 2 and revoked.revocation_revision == 1
    assert revoked.revoked_at is not None
    assert revoked.snapshot == first.snapshot and first.revoked_at is None
    env.permissions = ModelContentPermissionStore(env.database)
    with pytest.raises(ModelContentPermissionError, match="permission_revoked"):
        env.check(revoked)
    with pytest.raises(ModelContentPermissionError):
        env.check(first)
    with pytest.raises(ModelContentPermissionError, match="permission_changed"):
        env.permissions.grant(env.snapshot(), expected_revision=1, user_confirmed=True)
    renewed = env.permissions.grant(env.snapshot(), expected_revision=2, user_confirmed=True)
    assert renewed.revision == 3 and renewed.revocation_revision == 1
    assert env.check(renewed) == renewed
    assert env.permissions.receipt(user_id=USER, revision=1) == first
    with pytest.raises(ModelContentPermissionError, match="permission_changed"):
        env.permissions.check(
            user_id=USER,
            expected_revision=3,
            expected_revocation_revision=0,
            profile=env.profile,
            evidence=(env.record,),
        )


def test_permission_changes_are_compare_and_swap_and_do_not_expand_old_receipts(
    env: Environment,
) -> None:
    first = env.grant()
    other = env.evidence.put(evidence_value("second-selected")).record
    changed = env.snapshot(records=(env.record, other))
    with pytest.raises(ModelContentPermissionError, match="permission_changed"):
        env.permissions.grant(changed, expected_revision=0, user_confirmed=True)
    assert env.check(first) == first
    second = env.permissions.grant(changed, expected_revision=1, user_confirmed=True)
    assert env.check(second, records=(other,)) == second
    with pytest.raises(ModelContentPermissionError, match="permission_changed"):
        env.check(first)
    assert len(first.snapshot.evidence) == 1


def test_two_concurrent_confirmations_cannot_both_replace_the_same_revision(
    env: Environment,
) -> None:
    snapshot = env.snapshot()
    barrier = Barrier(2)

    def confirm() -> str:
        barrier.wait(timeout=5)
        try:
            env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
            return "granted"
        except ModelContentPermissionError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: confirm(), range(2)))
    assert sorted(results) == ["granted", "permission_changed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "revoke",
        "regrant",
        "grant_revision",
        "grant_granted_at",
        "root_identity",
        "requested_path",
        "root_path",
        "candidate_kind",
        "zero_revision",
        "unknown_identity",
        "missing_grant",
        "replace_directory",
    ],
)
def test_source_authority_changes_invalidate_existing_consent(
    env: Environment,
    mutation: str,
) -> None:
    receipt = env.grant()
    assert env.check(receipt) == receipt
    if mutation == "revoke":
        env.sources.revoke(user_id=USER, source_id=SOURCE)
    elif mutation == "regrant":
        env.sources.revoke(user_id=USER, source_id=SOURCE)
        env.sources.grant(
            user_id=USER,
            source_id=SOURCE,
            root=env.root,
            candidate_kind=RootCandidateKind.DEFAULT,
        )
    elif mutation == "replace_directory":
        env.root.rename(env.root.with_name("original-projects"))
        env.root.mkdir()
    else:
        sql = {
            "grant_revision": "UPDATE source_grants SET revision=revision+1",
            "grant_granted_at": "UPDATE source_grants SET granted_at='2099-01-01'",
            "root_identity": "UPDATE source_grants SET root_identity='root:v2:replaced'",
            "requested_path": "UPDATE source_grants SET requested_path='/missing-root'",
            "root_path": "UPDATE source_grants SET root_path='/another-root'",
            "candidate_kind": "UPDATE source_grants SET candidate_kind='environment_override'",
            "zero_revision": "UPDATE source_grants SET revision=0",
            "unknown_identity": "UPDATE source_grants SET root_identity='root:v99:unknown'",
            "missing_grant": "DELETE FROM source_grants",
        }[mutation]
        with sqlite3.connect(env.database) as db:
            db.execute(sql)
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt)
    assert env.permissions.receipt(user_id=USER) == receipt


def test_sync_permission_is_independent_and_does_not_grant_model_content(env: Environment) -> None:
    config = ConfigStore(env.database.parent)
    config.set_sync(SOURCE, True)
    assert env.permissions.receipt(user_id=USER) is None
    receipt = env.grant()
    config.set_sync(SOURCE, False)
    assert env.check(receipt) == receipt


def test_missing_source_grant_cannot_be_confirmed(env: Environment) -> None:
    with sqlite3.connect(env.database) as db:
        db.execute("DELETE FROM source_grants")
    with pytest.raises(ModelContentPermissionError):
        env.snapshot()
    assert env.permissions.receipt(user_id=USER) is None


@pytest.mark.parametrize("mutation", ["source", "profile", "evidence"])
def test_confirmation_revalidates_the_displayed_snapshot(
    env: Environment,
    mutation: str,
) -> None:
    snapshot = env.snapshot()
    if mutation == "source":
        env.sources.grant(
            user_id=USER,
            source_id=SOURCE,
            root=env.root,
            candidate_kind=RootCandidateKind.DEFAULT,
        )
    elif mutation == "profile":
        env.profiles.update(
            env.profile.profile_id,
            ModelProfilePatch(expected_revision=1, model="changed"),
            env.secrets,
        )
    else:
        env.evidence.put(evidence_value("selected", text="Changed after the consent preview"))
    with pytest.raises(ModelContentPermissionError):
        env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
    assert env.permissions.receipt(user_id=USER) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_url", "http://127.0.0.1:19000/v1"),
        ("model", "different-generation"),
        ("embedding_base_url", "http://127.0.0.1:19001/v1"),
        ("embedding_model", "different-embedding"),
        ("embedding_provider_kind", "local_onnx"),
        ("embedding_dimensions", 777),
        ("timeout_seconds", 120),
        ("revision", 2),
    ],
)
def test_changed_profile_is_rejected_even_when_caller_supplies_the_old_snapshot(
    env: Environment,
    field: str,
    value: object,
) -> None:
    receipt = env.grant()
    assert env.check(receipt) == receipt
    with sqlite3.connect(env.database) as db:
        db.execute(
            f"UPDATE model_profiles SET {field}=? WHERE profile_id=?",  # fixed test fields
            (value, env.profile.profile_id),
        )
    with pytest.raises(ModelContentPermissionError, match="profile_changed"):
        env.check(receipt)


def test_other_profile_or_recipient_cannot_be_substituted_in_a_request(env: Environment) -> None:
    receipt = env.grant()
    for mutation in (
        {"profile_id": "not-the-approved-profile"},
        {"base_url": "https://different.example/v1"},
        {"embedding_base_url": "https://vectors.example/v1"},
    ):
        with pytest.raises(ModelContentPermissionError, match="profile_changed"):
            env.check(receipt, profile=env.profile.model_copy(update=mutation))
    tampered = receipt.snapshot.model_copy(
        update={
            "generation": receipt.snapshot.generation.model_copy(
                update={
                    "endpoint": "https://different.example/v1",
                }
            ),
        }
    )
    with pytest.raises(ModelContentPermissionError):
        env.permissions.grant(tampered, expected_revision=1, user_confirmed=True)
    assert env.check(receipt) == receipt


def test_switching_or_deleting_the_current_profile_invalidates_checks(env: Environment) -> None:
    receipt = env.grant()
    other = env.profiles.create(
        ModelProfileInput(name="Other", provider_kind=ProviderKind.OLLAMA, model="other"),
        env.secrets,
    )
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE model_profiles SET is_active=0")
        db.execute("UPDATE model_profiles SET is_active=1 WHERE profile_id=?", (other.profile_id,))
    with pytest.raises(ModelContentPermissionError, match="profile_changed"):
        env.check(receipt)
    with sqlite3.connect(env.database) as db:
        db.execute("DELETE FROM model_profiles WHERE profile_id=?", (env.profile.profile_id,))
    with pytest.raises(ModelContentPermissionError, match="profile_changed"):
        env.check(receipt)


@pytest.mark.parametrize(
    "mutation",
    ["revision", "verbatim", "written_by", "workspace", "source", "user", "origins", "delete"],
)
def test_changed_evidence_never_passes_a_saved_exact_snapshot(
    env: Environment,
    mutation: str,
) -> None:
    receipt = env.grant()
    assert env.check(receipt) == receipt
    sql = {
        "revision": "UPDATE evidence SET revision=revision+1",
        "verbatim": "UPDATE evidence SET verbatim='same revision but different body'",
        "written_by": "UPDATE evidence SET written_by='agent'",
        "workspace": "UPDATE evidence SET workspace='different-workspace'",
        "source": "UPDATE evidence SET source_id='cursor'",
        "user": "UPDATE evidence SET user_id='other-user'",
        "origins": 'UPDATE evidence_origins SET raw_meta_json=\'{"extra":"changed"}\'',
        "delete": "DELETE FROM evidence",
    }[mutation]
    with sqlite3.connect(env.database) as db:
        db.execute(sql)
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt)
    current = env.evidence.get(env.record.evidence_id)
    if current is not None:
        with pytest.raises(ModelContentPermissionError):
            env.check(receipt, records=(current,))


def test_missing_cross_user_and_unselected_evidence_scopes_are_denied(env: Environment) -> None:
    receipt = env.grant()
    other = env.evidence.put(evidence_value("unselected")).record
    other_source = env.evidence.put(evidence_value("other-source", source_id="cursor")).record
    other_user = env.evidence.put(evidence_value("other-user", user_id="someone-else")).record
    for records in ((), (other,), (other_source,), (other_user,), (env.record, other)):
        with pytest.raises(ModelContentPermissionError):
            env.check(receipt, records=records)
    with pytest.raises(ModelContentPermissionError, match="permission_required"):
        env.check(receipt, user_id="someone-else")
    with pytest.raises(ModelContentPermissionError):
        env.snapshot(records=(other_user,))
    with pytest.raises(ModelContentPermissionError):
        env.snapshot(source_ids=())
    with pytest.raises(ModelContentPermissionError):
        env.snapshot(records=())


def test_exact_mode_does_not_silently_authorize_later_imports(env: Environment) -> None:
    receipt = env.grant()
    later = env.evidence.put(evidence_value("after-confirmation")).record
    assert env.check(receipt) == receipt
    with pytest.raises(ModelContentPermissionError, match="evidence_not_permitted"):
        env.check(receipt, records=(later,))


def test_future_mode_only_authorizes_explicit_sources_and_truly_new_additions(
    env: Environment,
) -> None:
    old = env.evidence.put(evidence_value("older-unselected")).record
    receipt = env.grant(future=True)
    assert receipt.snapshot.include_future_additions is True
    assert receipt.snapshot.future_after_sequence is not None
    later = env.evidence.put(evidence_value("new-addition")).record
    assert env.check(receipt, records=(later,)) == receipt
    env.permissions = ModelContentPermissionStore(env.database)
    assert env.check(receipt, records=(env.record, later)) == receipt
    outside = env.evidence.put(evidence_value("outside", source_id="cursor")).record
    for record in (old, outside):
        with pytest.raises(ModelContentPermissionError, match="evidence_not_permitted"):
            env.check(receipt, records=(record,))
    edited = env.evidence.put(evidence_value("new-addition", text="Edited, not added")).record
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt, records=(later,))
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt, records=(edited,))


def test_future_only_consent_never_backfills_preexisting_evidence(env: Environment) -> None:
    future = env.snapshot(records=(), future=True)
    receipt = env.permissions.grant(future, user_confirmed=True, expected_revision=0)
    with pytest.raises(ModelContentPermissionError, match="evidence_not_permitted"):
        env.check(receipt)
    later = env.evidence.put(evidence_value("first-future")).record
    assert env.check(receipt, records=(later,)) == receipt


def test_future_sequence_is_not_based_on_source_timestamps_or_reused_rowids(
    env: Environment,
) -> None:
    before = env.evidence.put(evidence_value("before")).record
    receipt = env.grant(future=True)
    with sqlite3.connect(env.database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("DELETE FROM evidence WHERE evidence_id=?", (before.evidence_id,))
    after = env.evidence.put(evidence_value("after")).record
    assert env.check(receipt, records=(after,)) == receipt
    reinserted = env.evidence.put(evidence_value("before")).record
    with pytest.raises(ModelContentPermissionError, match="evidence_not_permitted"):
        env.check(receipt, records=(reinserted,))


@pytest.mark.parametrize(
    "mutation",
    ["verbatim", "workspace", "origins", "new_origin", "deleted_origin"],
)
def test_future_additions_reject_mutations_even_without_a_revision_bump(
    env: Environment,
    mutation: str,
) -> None:
    receipt = env.grant(future=True)
    later = env.evidence.put(evidence_value("later")).record
    assert env.check(receipt, records=(later,)) == receipt
    with sqlite3.connect(env.database) as db:
        if mutation == "verbatim":
            db.execute(
                "UPDATE evidence SET verbatim='changed' WHERE evidence_id=?", (later.evidence_id,)
            )
        elif mutation == "workspace":
            db.execute(
                "UPDATE evidence SET workspace='changed' WHERE evidence_id=?", (later.evidence_id,)
            )
        elif mutation == "origins":
            db.execute(
                "UPDATE evidence_origins SET raw_meta_json='{}', source_version='new' "
                "WHERE evidence_id=?",
                (later.evidence_id,),
            )
        elif mutation == "deleted_origin":
            db.execute("DELETE FROM evidence_origins WHERE evidence_id=?", (later.evidence_id,))
        else:
            db.execute(
                "INSERT INTO evidence_origins VALUES (?,'hook','{}',NULL,?)",
                (later.evidence_id, datetime.now(UTC).isoformat()),
            )
    current = env.evidence.get(later.evidence_id)
    assert current is not None
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt, records=(current,))


def test_identical_reimport_does_not_consume_or_expand_future_consent(env: Environment) -> None:
    receipt = env.grant(future=True)
    later = env.evidence.put(evidence_value("later")).record
    assert env.check(receipt, records=(later,)) == receipt
    duplicate = env.evidence.put(evidence_value("later")).record
    assert env.check(receipt, records=(duplicate,)) == receipt


def test_future_only_consent_does_not_reclassify_deleted_legacy_evidence(
    env: Environment,
) -> None:
    receipt = env.permissions.grant(
        env.snapshot(records=(), future=True),
        expected_revision=0,
        user_confirmed=True,
    )
    with sqlite3.connect(env.database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("DELETE FROM evidence WHERE evidence_id=?", (env.record.evidence_id,))
    recreated = env.evidence.put(evidence_value("selected")).record
    with pytest.raises(ModelContentPermissionError, match="evidence_not_permitted"):
        env.check(receipt, records=(recreated,))


@pytest.mark.parametrize(
    "trigger", ["model_content_track_evidence_addition", "model_content_invalidate_evidence_update"]
)
def test_future_tracking_that_cannot_prove_additions_is_denied(
    env: Environment,
    trigger: str,
) -> None:
    receipt = env.grant(future=True)
    later = env.evidence.put(evidence_value("later")).record
    assert env.check(receipt, records=(later,)) == receipt
    with sqlite3.connect(env.database) as db:
        db.execute(f"DROP TRIGGER {trigger}")  # fixed test identifiers
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt, records=(later,))


def test_unknown_saved_policy_schema_is_rejected(env: Environment) -> None:
    receipt = env.grant()
    with sqlite3.connect(env.database) as db:
        db.execute(
            "UPDATE model_content_permissions SET snapshot_json="
            "json_set(snapshot_json, '$.schema_version', 999)"
        )
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt)


def test_profile_version_row_cannot_diverge_from_the_current_profile(env: Environment) -> None:
    receipt = env.grant()
    assert env.check(receipt) == receipt
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE model_profile_versions SET embedding_base_url='http://127.0.0.1:19999'")
    with pytest.raises(ModelContentPermissionError, match="profile_changed"):
        env.check(receipt)


def test_all_selected_sources_and_evidence_are_bound_even_for_a_subset_request(
    env: Environment,
) -> None:
    second_root = env.root.parent / "cursor"
    second_root.mkdir()
    env.sources.grant(
        user_id=USER, source_id="cursor", root=second_root, candidate_kind=RootCandidateKind.DEFAULT
    )
    other = env.evidence.put(evidence_value("cursor-turn", source_id="cursor")).record
    snapshot = env.snapshot(source_ids=("cursor", SOURCE), records=(env.record, other))
    receipt = env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
    assert env.check(receipt) == receipt
    env.sources.revoke(user_id=USER, source_id="cursor")
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt)


def test_permission_rescoping_replaces_instead_of_unions_previous_sources(env: Environment) -> None:
    first = env.grant()
    second_root = env.root.parent / "cursor"
    second_root.mkdir()
    env.sources.grant(
        user_id=USER, source_id="cursor", root=second_root, candidate_kind=RootCandidateKind.DEFAULT
    )
    other = env.evidence.put(evidence_value("cursor-turn", source_id="cursor")).record
    changed = env.snapshot(source_ids=("cursor",), records=(other,))
    receipt = env.permissions.grant(changed, expected_revision=first.revision, user_confirmed=True)
    assert env.check(receipt, records=(other,)) == receipt
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt)


def test_forged_future_watermark_and_edited_preview_are_denied(env: Environment) -> None:
    env.evidence.put(evidence_value("not-selected-before-preview"))
    snapshot = env.snapshot(future=True)
    forged = snapshot.model_copy(update={"future_after_sequence": 0})
    with pytest.raises(ModelContentPermissionError):
        env.permissions.grant(forged, expected_revision=0, user_confirmed=True)
    env.evidence.put(evidence_value("selected", text="Changed since preview"))
    env.evidence.put(evidence_value("arrived-after-preview"))
    with pytest.raises(ModelContentPermissionError):
        env.permissions.grant(snapshot, expected_revision=0, user_confirmed=True)
    assert env.permissions.receipt(user_id=USER) is None


def test_manual_source_uses_exact_evidence_without_filesystem_authority(env: Environment) -> None:
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(env.database) as db:
        db.execute(
            "INSERT INTO source_grants VALUES (?,?,?,'default',?,NULL)",
            (USER, MANUAL, "/deliberately-not-a-filesystem-source", now),
        )
    manual = env.evidence.put(evidence_value("manual-note", source_id=MANUAL)).record
    snapshot = env.snapshot(records=(manual,), source_ids=(MANUAL,))
    assert snapshot.sources[0].authority == "manual"
    assert snapshot.sources[0].root_identity is None and snapshot.sources[0].revision == 0
    receipt = env.permissions.grant(snapshot, user_confirmed=True, expected_revision=0)
    assert env.check(receipt, records=(manual,)) == receipt
    with pytest.raises(ModelContentPermissionError):
        env.snapshot(records=(manual,), source_ids=(MANUAL,), future=True)
    env.sources.revoke(user_id=USER, source_id=MANUAL)
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt, records=(manual,))
    with sqlite3.connect(env.database) as db:
        db.execute(
            "UPDATE source_grants SET revoked_at=NULL, granted_at=? WHERE source_id=?",
            (datetime.now(UTC).isoformat(), MANUAL),
        )
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt, records=(manual,))


def test_local_embedding_recipient_is_explicit_and_still_needs_consent(env: Environment) -> None:
    updated = env.profiles.update(
        env.profile.profile_id,
        ModelProfilePatch(expected_revision=1, embedding_provider_kind=ProviderKind.LOCAL_ONNX),
        env.secrets,
    )
    env.profile = env.profiles.snapshot(updated.profile_id, updated.revision)
    snapshot = env.snapshot()
    assert snapshot.embedding.provider_kind is ProviderKind.LOCAL_ONNX
    assert snapshot.embedding.endpoint is None
    assert env.permissions.receipt(user_id=USER) is None
    receipt = env.permissions.grant(snapshot, user_confirmed=True, expected_revision=0)
    assert env.check(receipt) == receipt


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:pass@example.test/v1",
        "https://example.test/v1?token=example",
        "https://example.test/v1#fragment",
        "not-an-endpoint",
    ],
)
def test_ambiguous_or_credential_bearing_recipients_fail_closed(
    env: Environment,
    endpoint: str,
) -> None:
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE model_profiles SET base_url=?", (endpoint,))
    profile = env.profile.model_copy(update={"base_url": endpoint})
    with pytest.raises(ModelContentPermissionError):
        env.permissions.snapshot(
            user_id=USER,
            source_ids=(SOURCE,),
            profile=profile,
            evidence=(env.record,),
        )


def test_no_dependencies_or_corrupt_receipt_never_produce_authority(tmp_path: Path) -> None:
    store = ModelContentPermissionStore(tmp_path / "empty.sqlite3")
    assert store.receipt(user_id=USER) is None
    with pytest.raises(ModelContentPermissionError, match="permission_required"):
        store.revoke(user_id=USER, expected_revision=0, user_confirmed=True)


def test_corrupt_or_unknown_snapshot_versions_fail_closed(env: Environment) -> None:
    receipt = env.grant()
    with sqlite3.connect(env.database) as db:
        db.execute("UPDATE model_content_permissions SET snapshot_json='{}'")
    with pytest.raises(ModelContentPermissionError):
        env.check(receipt)
