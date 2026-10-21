"""Keep the preset profile name identical on both sides of the Core/web boundary.

`seed_preset_profile` claims its one-shot seed with the profile *name* (see
`_claim_seed` and the `model_profile_seeds` table), and `ModelSettings.tsx` has
nothing else to identify the preset by than `profile.name`. The constant therefore
exists twice, in two languages, and the two copies are load-bearing: if they drift,
Core seeds a profile the web layer refuses to recognise as the preset.

This parity was established by hand once, so it is pinned here.
"""

import ast
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE_PRESETS = ROOT / "memhub" / "apps" / "core" / "src" / "one_memory" / "model_presets.py"
WEB_SETTINGS = ROOT / "memhub" / "apps" / "web" / "src" / "ModelSettings.tsx"

CONSTANT = "PRESET_PROFILE_NAME"

pytestmark = pytest.mark.skipif(
    not (CORE_PRESETS.exists() and WEB_SETTINGS.exists()),
    reason="memhub workspaces are not present in this checkout",
)


def _core_constant():
    """The value Core assigns to PRESET_PROFILE_NAME."""
    tree = ast.parse(CORE_PRESETS.read_text(encoding="utf-8"))
    values = [
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == CONSTANT
            for target in node.targets
        )
    ]
    assert len(values) == 1, "expected exactly one %s assignment in %s" % (
        CONSTANT,
        CORE_PRESETS.relative_to(ROOT),
    )
    return values[0]


def _web_constant():
    """The value the web layer assigns to PRESET_PROFILE_NAME."""
    matches = re.findall(
        r"^const\s+%s\s*=\s*\"([^\"]*)\"\s*;" % CONSTANT,
        WEB_SETTINGS.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert len(matches) == 1, "expected exactly one %s declaration in %s" % (
        CONSTANT,
        WEB_SETTINGS.relative_to(ROOT),
    )
    return matches[0]


def test_preset_profile_name_matches_across_core_and_web():
    assert _core_constant() == _web_constant()


def test_preset_profile_name_is_claimed_by_name():
    """The seed claim keys on the name, so the parity above is not cosmetic.

    Were the claim to stop using the profile name, this test should be deleted
    together with the parity test rather than quietly kept green.
    """
    source = CORE_PRESETS.read_text(encoding="utf-8")
    assert "_claim_seed(profiles.database_path, preset.name)" in source


def test_preset_profile_name_carries_no_untranslated_text():
    for label, value in (("core", _core_constant()), ("web", _web_constant())):
        assert value.isascii(), "%s is not ASCII on the %s side: %r" % (CONSTANT, label, value)
