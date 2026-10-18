"""Python's enums and the hand-written TypeScript union must agree.

The project rule says "TypeScript types are generated or verified from the contract", and before \
this test
neither existed: `packages/contracts/src/index.ts` was copied by hand, and a wrong or missing \
copy would not
warn about anything. Whole-codegen is out of scope for this round, but making the single-source \
claim real
for these enums costs only this one file.
"""

import re
from pathlib import Path

import pytest

from one_memory.memory_governance import MemoryKind, MemoryOrigin, MemoryStatus

CONTRACTS = Path(__file__).resolve().parents[3] / "packages" / "contracts" / "src" / "index.ts"


def _union(name: str) -> tuple[str, ...]:
    source = CONTRACTS.read_text(encoding="utf-8")
    match = re.search(rf"^export type {name} = (.+);$", source, re.MULTILINE)
    assert match is not None, f"{name} is missing from the TypeScript contract"
    return tuple(sorted(re.findall(r'"([^"]+)"', match.group(1))))


@pytest.mark.parametrize(
    ("name", "enum"),
    [
        ("MemoryStatus", MemoryStatus),
        ("MemoryKind", MemoryKind),
        ("MemoryOrigin", MemoryOrigin),
    ],
)
def test_the_typescript_union_says_what_python_says(name, enum) -> None:
    assert _union(name) == tuple(sorted(member.value for member in enum))


def test_governed_memory_carries_the_same_fields_on_both_sides() -> None:
    from one_memory.memory_governance import GovernedMemory

    source = CONTRACTS.read_text(encoding="utf-8")
    match = re.search(
        r"^export interface GovernedMemory \{\n(.*?)^\}$",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    typescript = {
        found.group(1) for found in re.finditer(r"^  (\w+)[?]?:", match.group(1), re.MULTILINE)
    }

    assert typescript == set(GovernedMemory.model_fields)


def test_an_agent_owned_memory_is_usable_and_never_passes_as_the_person(tmp_path) -> None:
    """Facts in assistant output are usable right away, but always labelled as what the \
assistant said.

    Read through recall, origin tells another AI app that this is a model statement, not the \
person's own verdict.
    An existing memory with the same content never has its origin marker rewritten by model \
output either.
    """
    from tests.test_memory_processing import (
        test_agent_output_keeps_the_origin_of_a_same_content_memory,
        test_agent_owned_memories_land_usable_but_marked_as_the_assistant,
    )

    created, existing = tmp_path / "created", tmp_path / "existing"
    created.mkdir()
    existing.mkdir()
    test_agent_owned_memories_land_usable_but_marked_as_the_assistant(created)
    test_agent_output_keeps_the_origin_of_a_same_content_memory(existing, MemoryOrigin.AGENT)
