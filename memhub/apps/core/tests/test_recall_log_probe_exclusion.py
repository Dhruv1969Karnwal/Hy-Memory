"""A-21: once R-10 introduces recall logging, "probes do not pollute" must turn red by itself \
instead of going quietly dead.

Current state (false green): R-24 accepts "probes never enter user-visible recall logs", and \
that today holds
because core has **no recall log table at all**, not because of filtering. Once R-10 adds the \
log table, `__om_probe__` will start
entering the log while the old test never goes red.

This file does two things:
1. writes this into the R-10 acceptance checklist (see docs/mvp/R-10-acceptance.md);
2. a structural guard: once a recall log schema marker appears in the repo, a
   probe-exclusion declaration must exist alongside it. With no log table today the guard checks \
"the marker is absent"; this is not a forever-green
   placeholder -- manually planting `CREATE TABLE recall_logs` without declaring the exclusion \
turns it red.
"""

from __future__ import annotations

import re
from pathlib import Path

CORE_SRC = Path(__file__).resolve().parents[1] / "src" / "one_memory"

RECALL_LOG_SCHEMA_MARKERS = (
    "recall_logs",
    "recall_events",
    "memory_recall_log",
    "recall_log_entries",
)

PROBE_EXCLUSION_SYMBOL = "PROBE_EXCLUDED_FROM_RECALL_LOG"


def _schema_sql() -> str:
    texts: list[str] = []
    for path in CORE_SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "CREATE TABLE" in text or "create table" in text.lower():
            texts.append(text)
    return "\n".join(texts)


def _recall_source() -> str:
    return (CORE_SRC / "recall.py").read_text(encoding="utf-8")


def assert_recall_logs_exclude_probe(*, schema: str, recall: str) -> None:
    """Shared so the tripwire and its meta-guard cannot drift apart."""
    present = [marker for marker in RECALL_LOG_SCHEMA_MARKERS if marker in schema]
    if not present:
        return
    assert PROBE_EXCLUSION_SYMBOL in recall, (
        f"recall log schema found ({', '.join(present)}) but "
        f"{PROBE_EXCLUSION_SYMBOL} is missing from recall.py — "
        "probe queries would start appearing in user-visible logs"
    )
    assert re.search(rf"{PROBE_EXCLUSION_SYMBOL}\s*=\s*True", recall)
    assert "PROBE_QUERY" in recall
    assert "request.query == PROBE_QUERY" in recall


def test_recall_log_schema_must_come_with_an_explicit_probe_exclusion() -> None:
    assert_recall_logs_exclude_probe(schema=_schema_sql(), recall=_recall_source())


def test_planting_a_recall_log_table_without_exclusion_turns_this_red() -> None:
    """Meta-guard: the tripwire is not a forever-green placeholder."""
    schema = _schema_sql() + "\nCREATE TABLE recall_logs (id TEXT);\n"
    recall = _recall_source()
    try:
        assert_recall_logs_exclude_probe(schema=schema, recall=recall)
    except AssertionError:
        return
    raise AssertionError(
        "tripwire stayed green after planting recall_logs without exclusion — "
        "it has become a forever-green placeholder"
    )
