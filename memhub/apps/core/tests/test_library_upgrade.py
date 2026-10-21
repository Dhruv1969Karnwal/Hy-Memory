"""The pre-upgrade backup is the first step of Core startup: the library is only looked at, \
copied as it is, and only then does anything change it.

The backup is no longer done in passing by whichever governance store happens to open first, so \
whichever store opens first,
the backup must be the library from before this start; the governance store itself also refuses \
to rewrite an old library nobody backed up.
"""

import asyncio
import errno
import os
import stat
from pathlib import Path

import pytest

from one_memory import api
from one_memory.imports.storage import EvidenceStore
from one_memory.memory_governance import MemoryGovernanceStore, MemoryStatus
from tests.test_memory_no_candidate import (
    _app,
    _backups,
    _legacy_library,
    _schema,
    _spy_on_backup,
    _tables,
)

#: The fields in the library file header that describe what kind of library this is (section 1.3 of
# the SQLite file format),
#: without the change counter that every write moves.
HEADER_FIELDS = {
    "page_size": (16, 18),
    "write_version": (18, 19),
    "read_version": (19, 20),
    "auto_vacuum": (52, 56),
    "text_encoding": (56, 60),
    "user_version": (60, 64),
    "incremental_vacuum": (64, 68),
    "application_id": (68, 72),
}


def _header(database: Path) -> dict[str, bytes]:
    head = database.read_bytes()[:100]
    return {name: head[start:end] for name, (start, end) in HEADER_FIELDS.items()}


def _stamp(database: Path) -> None:
    """Stamp user_version / application_id on the library; a backup that is rebuilt rather than \
copied loses both."""
    import sqlite3

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA user_version=7")
        connection.execute("PRAGMA application_id=1330795853")
    finally:
        connection.close()


def _start(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(scenario())


@pytest.mark.parametrize("journal", ["wal", "delete"])
def test_the_backup_is_the_library_file_as_this_start_found_it(tmp_path, journal) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database, _evidence = _legacy_library(data_dir, journal=journal)
    _stamp(database)
    header, schema, before = _header(database), _schema(database), _tables(database)
    # Positive premise: the two journal modes really are written differently in the header, so
    # comparing it means something.
    assert header["write_version"] == (b"\x02" if journal == "wal" else b"\x01")

    _start(tmp_path)

    (backup,) = _backups(data_dir)
    assert _header(backup) == header
    assert _schema(backup) == schema
    assert _tables(backup) == before
    assert _header(database)["write_version"] == b"\x02", "WAL is only switched on after startup"


def test_preparing_a_library_changes_nothing_in_it(tmp_path) -> None:
    from one_memory.library_upgrade import prepare_library

    database, _evidence = _legacy_library(tmp_path, journal="delete")
    content = database.read_bytes()
    listing = sorted(path.name for path in tmp_path.iterdir())

    prepared = prepare_library(database)

    assert prepared.backup is not None and prepared.backup.exists()
    assert database.read_bytes() == content
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([*listing, "backups"])


def test_a_library_without_legacy_rows_is_prepared_without_a_backup(tmp_path) -> None:
    from one_memory.library_upgrade import prepare_library

    database = tmp_path / "one-memory.sqlite3"
    MemoryGovernanceStore(database)

    assert prepare_library(database).backup is None
    assert prepare_library(tmp_path / "absent.sqlite3").backup is None
    assert not (tmp_path / "backups").exists()


def test_the_store_refuses_to_rewrite_a_library_nobody_backed_up(tmp_path) -> None:
    from one_memory.library_upgrade import LibraryUpgradeError

    database, _evidence = _legacy_library(tmp_path, journal="delete")
    content = database.read_bytes()

    with pytest.raises(LibraryUpgradeError) as refused:
        MemoryGovernanceStore(database)

    assert refused.value.code == "library_upgrade_not_prepared"
    assert database.read_bytes() == content
    assert not (tmp_path / "backups").exists()


def test_a_backup_of_another_library_does_not_let_this_one_be_rewritten(tmp_path) -> None:
    from one_memory.library_upgrade import LibraryUpgradeError, prepare_library

    (tmp_path / "here").mkdir()
    (tmp_path / "other").mkdir()
    here, _ = _legacy_library(tmp_path / "here", journal="delete")
    other, _ = _legacy_library(tmp_path / "other", journal="delete")
    content = here.read_bytes()
    # Both are checked out before migration: a migrated library needs no backup, so a proof without
    # a backup could not reach the path.
    for_other, also_for_other = prepare_library(other), prepare_library(other)
    assert also_for_other.backup is not None
    # Positive premise: a proof checked out this way is accepted for the very library it backed up.
    MemoryGovernanceStore(other, prepared=for_other)

    with pytest.raises(LibraryUpgradeError) as refused:
        MemoryGovernanceStore(here, prepared=also_for_other)

    assert refused.value.code == "library_upgrade_not_prepared"
    assert here.read_bytes() == content


def _candidate_status(database: Path) -> str:
    import sqlite3

    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        return connection.execute(
            "SELECT status FROM governed_memories WHERE memory_id='candidate'"
        ).fetchone()[0]
    finally:
        connection.close()


def test_a_proof_nobody_issued_does_not_let_a_library_be_rewritten(tmp_path) -> None:
    """Only prepare_library may issue a proof: assembling one yourself is not accepted, not even \
when it points at a backup that would work."""
    import shutil

    from one_memory.library_upgrade import LibraryUpgradeError, PreparedLibrary

    database, _evidence = _legacy_library(tmp_path, journal="delete")
    content = database.read_bytes()
    real_copy = tmp_path / "copy.sqlite3"
    shutil.copyfile(database, real_copy)

    for backup in (tmp_path / "\u4e0d\u5b58\u5728.sqlite3", real_copy):
        with pytest.raises(LibraryUpgradeError) as refused:
            MemoryGovernanceStore(database, prepared=PreparedLibrary(database.resolve(), backup))
        assert refused.value.code == "library_upgrade_not_prepared"
        assert database.read_bytes() == content
    assert _candidate_status(database) == "candidate"


def test_a_proof_lets_one_store_through_and_no_second(tmp_path) -> None:
    from one_memory.database_access import close_main_databases
    from one_memory.library_upgrade import LibraryUpgradeError, prepare_library

    database, _evidence = _legacy_library(tmp_path, journal="delete")
    content = database.read_bytes()
    prepared = prepare_library(database)
    # Positive premise: first use, migrated as usual.
    MemoryGovernanceStore(database, prepared=prepared)
    assert _candidate_status(database) == "confirmed"
    close_main_databases()
    for sidecar in ("-wal", "-shm"):
        Path(f"{database}{sidecar}").unlink(missing_ok=True)
    database.write_bytes(content)

    with pytest.raises(LibraryUpgradeError) as refused:
        MemoryGovernanceStore(database, prepared=prepared)

    assert refused.value.code == "library_upgrade_not_prepared"
    assert database.read_bytes() == content


def test_legacy_status_migration_synchronizes_the_content_candidate_shadow(tmp_path) -> None:
    from one_memory.library_upgrade import prepare_library

    database, _evidence = _legacy_library(tmp_path, journal="delete")
    prepared = prepare_library(database)
    store = MemoryGovernanceStore(database, prepared=prepared)

    found, _ = store.find_confirmed_matching_terms(user_id="local", terms=("candidate",))
    assert [memory.memory_id for memory in found] == ["candidate"]
    rejected, _ = store.find_confirmed_matching_terms(user_id="local", terms=("rejected",))
    assert rejected == []
    with store._connection() as connection:
        rows = connection.execute(
            "SELECT memory_id FROM governed_memories_terms "
            "WHERE term = 'rejected' ORDER BY memory_id"
        ).fetchall()
        assert [row[0] for row in rows] == []


def _delete(backup: Path) -> None:
    backup.unlink()


def _empty(backup: Path) -> None:
    # A hard kill half way through the backup leaves exactly this kind of empty placeholder.
    backup.write_bytes(b"")


def _garble(backup: Path) -> None:
    # The file header is intact, so a check that only reads the header cannot notice.
    head = backup.read_bytes()
    backup.write_bytes(head[:100] + b"\xa5" * (len(head) - 100))


def _replace(backup: Path) -> None:
    # Swapped for an intact but unrelated library: header and quick_check both pass, yet it cannot
    # restore this library.
    import sqlite3

    backup.unlink()
    connection = sqlite3.connect(backup)
    try:
        connection.execute("CREATE TABLE unrelated (x INTEGER)")
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize(
    "spoil",
    [_delete, _empty, _garble, _replace],
    ids=["deleted", "empty", "garbled", "replaced"],
)
def test_a_backup_that_is_gone_by_the_time_it_is_needed_stops_the_upgrade(tmp_path, spoil) -> None:
    from one_memory.library_upgrade import LibraryUpgradeError, prepare_library

    database, _evidence = _legacy_library(tmp_path, journal="delete")
    content = database.read_bytes()
    prepared = prepare_library(database)
    assert prepared.backup is not None
    spoil(prepared.backup)

    with pytest.raises(LibraryUpgradeError) as refused:
        MemoryGovernanceStore(database, prepared=prepared)

    assert refused.value.code == "library_upgrade_backup_unusable"
    assert str(database) in refused.value.remedy
    assert database.read_bytes() == content


@pytest.mark.parametrize("spoil", [_empty, _garble], ids=["empty", "garbled"])
def test_a_backup_that_was_never_sound_stops_the_upgrade(tmp_path, monkeypatch, spoil) -> None:
    """Broken as it is written: the digest recorded at issue time is the broken file's, so only \
re-verifying the file notices."""
    from one_memory.database_access import MainDatabase
    from one_memory.library_upgrade import LibraryUpgradeError, prepare_library

    real_back_up = MainDatabase.back_up

    def spoiled_back_up(self, target):
        real_back_up(self, target)
        spoil(target)

    database, _evidence = _legacy_library(tmp_path, journal="delete")
    content = database.read_bytes()
    with monkeypatch.context() as patched:
        patched.setattr(MainDatabase, "back_up", spoiled_back_up)
        prepared = prepare_library(database)

    with pytest.raises(LibraryUpgradeError) as refused:
        MemoryGovernanceStore(database, prepared=prepared)

    assert refused.value.code == "library_upgrade_backup_unusable"
    assert database.read_bytes() == content


def test_the_backup_does_not_depend_on_which_store_opens_first(tmp_path, monkeypatch) -> None:
    """Another Core store moved ahead of the governance store, and the backup is still the \
library from before startup."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database, _evidence = _legacy_library(data_dir)
    schema, before = _schema(database), _tables(database)
    governance = api.MemoryGovernanceStore

    def evidence_first(path, **kwargs):
        EvidenceStore(path)
        return governance(path, **kwargs)

    monkeypatch.setattr(api, "MemoryGovernanceStore", evidence_first)
    # Positive premise: EvidenceStore really does add to the old library, otherwise swapping the
    # order proves nothing.
    probe = tmp_path / "probe"
    probe.mkdir()
    probe_database, _ = _legacy_library(probe)
    probe_schema = _schema(probe_database)
    EvidenceStore(probe_database)
    assert _schema(probe_database) != probe_schema

    _start(tmp_path)

    (backup,) = _backups(data_dir)
    assert _schema(backup) == schema
    assert _tables(backup) == before


def test_a_backup_does_not_need_hard_links(tmp_path, monkeypatch) -> None:
    """exFAT and some network drives do not support hard links; measured on exFAT, os.link \
reports ENOTSUP and the old library cannot start."""
    from one_memory.library_upgrade import prepare_library

    database, _evidence = _legacy_library(tmp_path)
    before = _tables(database)

    def no_links(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(os, "link", no_links)

    prepared = prepare_library(database)

    assert prepared.backup is not None
    assert _tables(prepared.backup) == before
    assert stat.S_IMODE(prepared.backup.stat().st_mode) == 0o600
    assert sorted(path.name for path in (tmp_path / "backups").iterdir()) == [prepared.backup.name]


def test_a_failed_backup_stops_the_upgrade_with_a_reason_and_the_next_start_recovers(
    tmp_path, monkeypatch
) -> None:
    from one_memory.library_upgrade import LibraryUpgradeError

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database, _evidence = _legacy_library(data_dir)
    schema, before = _schema(database), _tables(database)

    def full() -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patched:
        _spy_on_backup(patched, full)
        with pytest.raises(LibraryUpgradeError) as stopped:
            _start(tmp_path)

    error = stopped.value
    assert error.code == "library_upgrade_backup_failed"
    assert "No space left on device" in str(error)
    assert str(data_dir / "backups") in error.remedy
    assert list((data_dir / "backups").iterdir()) == []
    assert (_schema(database), _tables(database)) == (schema, before)

    _start(tmp_path)

    (backup,) = _backups(data_dir)
    assert _tables(backup) == before
    migrated = MemoryGovernanceStore(database)
    assert migrated.get("candidate", user_id="local").status is MemoryStatus.CONFIRMED
