"""BUG-03: the hook lane and the file scan must share one identity for one message.

Cursor has no shared native message id (review 2300: transcripts carry no id and
the hook payload only offers ``generation_id``, which is a turn id). So the two
lanes have to agree on a derived key built from the same normalized text.

The ordering constraint that shapes these tests: BUG-04 strips trailing thinking
text from assistant bodies, which changes the stored verbatim. If the derived key
were hashed over the cleaned body, one BUG-04 rescan would turn every record into
a new identity and duplicate the whole batch. The key must be hashed over the
pre-cleaning text, so these fixtures run the same strip that BUG-04 will add.
"""

from datetime import UTC, datetime

from one_memory.imports.models import EvidenceInput, EvidenceOrigin, OriginMetadata, WrittenBy
from one_memory.imports.storage import EvidenceStore

THINKING_SUFFIX = (
    "\n\n\u6211\u9700\u8981\u5148\u786e\u8ba4\u8fd9\u6761\u6d88\u606f"
    "\u7684\u843d\u70b9\uff0c\u518d\u51b3\u5b9a\u8981\u4e0d\u8981\u6539"
    "\u626b\u63cf\u5668\u7684\u53d6\u820d\uff0c\u5426\u5219\u4f1a\u628a"
    "\u5df2\u7ecf\u5165\u5e93\u7684\u5386\u53f2\u53c8\u7b97\u4e00\u904d"
    "\u3002"
)


def strip_thinking(verbatim: str) -> str:
    """The BUG-04 cleaning these tests must survive: drop a trailing thinking tail."""
    marker = "\n\n"
    if marker not in verbatim:
        return verbatim
    head, _, tail = verbatim.rpartition(marker)
    return head if tail.startswith("\u6211\u9700\u8981\u5148\u786e\u8ba4") else verbatim


def hook_input(*, verbatim: str, workspace: str | None = None) -> EvidenceInput:
    return EvidenceInput(
        source_id="cursor",
        conversation_id="conversation-1",
        workspace=workspace,
        message_id=None,
        source_ordinal="hook-visible",
        role="assistant",
        written_by=WrittenBy.AGENT,
        created_at=datetime(2026, 10, 12, 21, 0, tzinfo=UTC),
        verbatim=verbatim,
        origin_meta={EvidenceOrigin.HOOK: OriginMetadata()},
    )


def file_input(
    *,
    verbatim: str,
    workspace: str | None = "/repo",
    message_id: str | None = "cursor-transcript:17",
    source_ordinal: str = "17",
) -> EvidenceInput:
    return EvidenceInput(
        source_id="cursor",
        conversation_id="conversation-1",
        workspace=workspace,
        message_id=message_id,
        source_ordinal=source_ordinal,
        role="assistant",
        written_by=WrittenBy.AGENT,
        created_at=datetime(2026, 10, 12, 21, 0, tzinfo=UTC),
        verbatim=verbatim,
        origin_meta={EvidenceOrigin.FILE: OriginMetadata()},
    )


def test_hook_and_file_scan_share_one_identity_for_one_message(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "one-memory.sqlite3")
    hook = store.put(hook_input(verbatim="Change it to dark green.")).record
    scanned = store.put(file_input(verbatim="Change it to dark green.")).record

    assert scanned.evidence_id == hook.evidence_id
    assert store.count() == 1


def test_identity_is_stable_when_cleaning_strips_thinking(tmp_path) -> None:
    """A rescan after BUG-04 must not mint a new identity for the same message."""
    store = EvidenceStore(tmp_path / "one-memory.sqlite3")
    raw = "Change it to dark green." + THINKING_SUFFIX
    before = store.put(file_input(verbatim=raw)).record
    after = store.put(file_input(verbatim=strip_thinking(raw))).record

    assert after.evidence_id == before.evidence_id
    assert store.count() == 1


def test_both_lane_orders_leave_one_row(tmp_path) -> None:
    """The review asked for both directions; hook-first and file-first must agree."""
    hook_first = EvidenceStore(tmp_path / "hook-first.sqlite3")
    hook_row = hook_first.put(hook_input(verbatim="Change it to dark green.")).record
    assert hook_first.put(file_input(verbatim="Change it to dark green.")).record.evidence_id == (
        hook_row.evidence_id
    )
    assert hook_first.count() == 1

    file_first = EvidenceStore(tmp_path / "file-first.sqlite3")
    file_row = file_first.put(file_input(verbatim="Change it to dark green.")).record
    assert file_first.put(hook_input(verbatim="Change it to dark green.")).record.evidence_id == (
        file_row.evidence_id
    )
    assert file_first.count() == 1


def test_distinct_text_stays_distinct(tmp_path) -> None:
    """Unifying the key must not collapse two genuinely different messages."""
    store = EvidenceStore(tmp_path / "one-memory.sqlite3")
    first = store.put(
        file_input(verbatim="Change it to dark green.", message_id="cursor-transcript:17")
    ).record
    second = store.put(
        file_input(verbatim="Change it to light green.", message_id="cursor-transcript:18")
    ).record

    assert first.evidence_id != second.evidence_id
    assert store.count() == 2


def test_same_text_in_another_conversation_stays_distinct(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "one-memory.sqlite3")
    first = store.put(file_input(verbatim="Change it to dark green.")).record
    second = store.put(
        file_input(verbatim="Change it to dark green.").model_copy(
            update={"conversation_id": "conversation-2"}
        )
    ).record

    assert first.evidence_id != second.evidence_id
    assert store.count() == 2


def test_derived_key_ignores_capture_time(tmp_path) -> None:
    """No shared message id, so both lanes fall through to the derived key.

    A hook stamps the moment it fired while the transcript carries the moment the
    words were written, so the same text seen 90 minutes apart is one message.
    """
    store = EvidenceStore(tmp_path / "one-memory.sqlite3")
    hook = store.put(hook_input(verbatim="Change it to dark green.")).record
    later = store.put(
        file_input(verbatim="Change it to dark green.", message_id=None).model_copy(
            update={"created_at": datetime(2026, 10, 12, 22, 30, tzinfo=UTC)}
        )
    ).record

    assert later.evidence_id == hook.evidence_id
    assert store.count() == 1


def test_derived_key_ignores_workspace(tmp_path) -> None:
    """The hook lane reports no workspace; the file scan knows the repo path."""
    store = EvidenceStore(tmp_path / "one-memory.sqlite3")
    scanned = store.put(
        file_input(verbatim="Change it to dark green.", message_id=None, workspace="/repo")
    ).record
    hooked = store.put(hook_input(verbatim="Change it to dark green.", workspace=None)).record

    assert scanned.evidence_id == hooked.evidence_id
    assert store.count() == 1


def test_derived_key_ignores_source_ordinal(tmp_path) -> None:
    """The hook lane has no transcript position, so it reports a constant one."""
    store = EvidenceStore(tmp_path / "one-memory.sqlite3")
    scanned = store.put(file_input(verbatim="Change it to dark green.", message_id=None)).record
    hooked = store.put(hook_input(verbatim="Change it to dark green.")).record

    assert scanned.evidence_id == hooked.evidence_id
    assert store.count() == 1


def test_identity_is_hashed_from_verbatim_not_raw_verbatim(tmp_path) -> None:
    """Only the file lane carries raw_verbatim, so hashing it keyed one message twice."""
    store = EvidenceStore(tmp_path / "one-memory.sqlite3")
    plain = store.put(file_input(verbatim="Change it to dark green.", message_id=None)).record
    with_raw = store.put(
        file_input(verbatim="Change it to dark green.", message_id=None).model_copy(
            update={"raw_verbatim": "<transcript>Change it to dark green.</transcript>"}
        )
    ).record

    assert with_raw.evidence_id == plain.evidence_id
    assert store.count() == 1
