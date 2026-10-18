"""Probe codes must exist in three places at once: the python enum, the TS contract, the UI \
label table.

Trigger: adding ProbeCode.MODEL_LIST_UNSUPPORTED changed only python;
neither the TS contract's union type nor ModelSettings' probeLabel table was synced.
The TS union type is hand-written, and a missing member does not fail the build;
probeLabel is a Record<ProbeCode, string>, so a lookup miss returns undefined and
that column renders blank -- it looks broken, but no test ever goes red.
This file is exactly that missing guard.
"""

import re
from pathlib import Path

from one_memory.model_probe import ProbeCode

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = REPO_ROOT / "packages" / "contracts" / "src" / "index.ts"
MODEL_SETTINGS = REPO_ROOT / "apps" / "web" / "src" / "ModelSettings.tsx"


def _typescript_union_members() -> set[str]:
    source = CONTRACT.read_text(encoding="utf-8")
    match = re.search(r"export type ProbeCode =(.*?);", source, re.DOTALL)
    assert match, f"found no ProbeCode union type in {CONTRACT}"
    return set(re.findall(r'"([a-z_]+)"', match.group(1)))


def _probe_label_keys() -> set[str]:
    source = MODEL_SETTINGS.read_text(encoding="utf-8")
    match = re.search(
        r"const labels: Record<ProbeCode, string> = \{(.*?)\};",
        source,
        re.DOTALL,
    )
    assert match, f"found no probeLabel label table in {MODEL_SETTINGS}"
    return set(re.findall(r"^\s*([a-z_]+):", match.group(1), re.MULTILINE))


def test_every_probe_code_exists_in_the_typescript_contract() -> None:
    python_codes = {code.value for code in ProbeCode}
    missing = python_codes - _typescript_union_members()
    assert not missing, (
        f"These probe codes exist in python but not in the TS contract: {sorted(missing)}. "
        f"Sync the ProbeCode union type in {CONTRACT.relative_to(REPO_ROOT)}."
    )


def test_every_probe_code_has_a_label_in_the_settings_ui() -> None:
    python_codes = {code.value for code in ProbeCode}
    missing = python_codes - _probe_label_keys()
    assert not missing, (
        f"These probe codes have no matching label in the UI label table: {sorted(missing)}. "
        f"With a label missing, probeLabel returns undefined and the UI renders blank."
        f"Add the probeLabel entry in {MODEL_SETTINGS.relative_to(REPO_ROOT)}."
    )


def test_the_contract_does_not_carry_codes_python_no_longer_emits() -> None:
    """Reverse control: the TS side must not keep codes python has already deleted.

    Without this, the first two checks can be fooled by stuffing TS and the label table with a \
pile of unused codes.
    """
    python_codes = {code.value for code in ProbeCode}
    stale = _typescript_union_members() - python_codes
    assert not stale, f"the TS contract carries codes python no longer emits: {sorted(stale)}"
