"""An instrument must never be able to break the thing it measures.

`diagnostic_event` validates its payload: an unlisted `details` key, a missing
`level`, or a detail value over 256 characters all raise. Two of our own nodes
shipped with unlisted keys, and because they sat on a request path and on the
loop heartbeat, they turned every external recall into a 500 and then killed
the process. Both were caught only by running the built runtime, not by tests.

So this checks the payloads at their source, by walking every literal call.
"""

import ast
from pathlib import Path

import pytest

from one_memory.diagnostic_log import _DETAIL_KEYS

SOURCES = (
    "api.py",
    "imports/async_bridge.py",
    "imports/live_sync.py",
    "imports/runner.py",
    "memory_processing.py",
)

ROOT = Path(__file__).resolve().parent.parent / "src" / "one_memory"


def _literal_calls(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "diagnostic_event":
            yield node


@pytest.mark.parametrize("relative", SOURCES)
def test_every_diagnostic_payload_is_valid(relative: str) -> None:
    path = ROOT / relative
    assert path.is_file(), path
    problems: list[str] = []
    for node in _literal_calls(path):
        keys = {keyword.arg for keyword in node.keywords}
        if "level" not in keys:
            problems.append(f"{relative}:{node.lineno}: missing level")
        for keyword in node.keywords:
            if keyword.arg != "details" or not isinstance(keyword.value, ast.Dict):
                continue
            for key in keyword.value.keys:
                if isinstance(key, ast.Constant) and key.value not in _DETAIL_KEYS:
                    problems.append(f"{relative}:{node.lineno}: unlisted details key {key.value!r}")
    assert not problems, "\n".join(problems)
