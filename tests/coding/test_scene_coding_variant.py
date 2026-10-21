# -*- coding: utf-8 -*-
"""
scene="coding" -- the software-engineering variant of the coding pipeline.

Relationship to scene="working": the same pipeline (same store / tables / reconciler); only two things differ,
and this file groups its tests by those two:
  1. The extraction spec -- EXTRACT_PROMPT_CODING (TestPromptSelection / TestPromptContent)
  2. The version-evolution chain layer -- chain_store (TestChainObserve / TestChainDecorate)

Plus the routing itself (TestSceneRouting) and the regression gate for "no scene passed means unchanged behavior" (TestWorkingUnchanged).

⚠️ All of these are zero-LLM / zero-embedding tests. The chain layer itself does not call a model; the extraction spec is only checked for "which prompt
is selected", not for what the model does after reading it -- that needs a CodeMemoryBench run.
"""
import os
import tempfile

import pytest

from hy_memory.client import HyMemoryClient
from hy_memory.coding.chain_store import HyCodeMem
from hy_memory.coding.extractor import (
    EXTRACT_PROMPT,
    EXTRACT_PROMPT_CODING,
    CodingMemoryExtractor,
)


class _FakeMsg:
    """Minimal ChatMessage stand-in -- only the fields _chain_text reads."""

    def __init__(self, role, content, tool_name=None):
        self.role = role
        self.content = content
        self.tool_name = tool_name
        self.tool_calls = None

    def is_tool_message(self):
        return self.role == "tool"


class _FakeProvider:
    config = None
    _llm_config = None


def _edit_msgs(said, old, new, path="qutebrowser/config/configtypes.py"):
    """One round of "user asks for a change + agent writes the file".

    Writing the tool message as three `path=/old=/new=` segments is deliberate: when the git lookup comes up empty, the chain layer uses this shape
    to synthesise the diff (_patch_from_edits). Most production scenarios cannot supply a local repo path, so this is the path actually taken.
    """
    return [
        _FakeMsg("user", said),
        _FakeMsg(
            "tool",
            f"path='{path}'\nold='''{old}'''\nnew='''{new}'''",
            tool_name="edit",
        ),
    ]


# ════════════════════════════════════════════════════════════════════
# 1. Routing
# ════════════════════════════════════════════════════════════════════


class TestSceneRouting:
    def test_coding_routes_to_coding_chain(self):
        # "coding" and "working" normalise to the same route value -- every
        # `scene_forced == "coding"` check in add()/search() applies to both at once, so no per-site branches are needed.
        assert HyMemoryClient._normalize_scene("coding") == "coding"
        assert HyMemoryClient._normalize_scene("working") == "coding"

    def test_chat_unaffected(self):
        assert HyMemoryClient._normalize_scene("chat") == "chat"

    def test_none_stays_auto(self):
        assert HyMemoryClient._normalize_scene(None) is None

    def test_case_and_whitespace_insensitive(self):
        assert HyMemoryClient._normalize_scene("  CODING ") == "coding"
        assert HyMemoryClient._scene_variant(" Coding ") == "coding"

    def test_illegal_value_raises(self):
        with pytest.raises(ValueError) as ei:
            HyMemoryClient._normalize_scene("productivity")
        # The error must list all three legal values -- saying only 'chat' or 'working' would make people think
        # that coding does not exist.
        assert "coding" in str(ei.value)

    def test_variant_is_orthogonal_to_route(self):
        # Two variants under the same route value ("coding"); the variant is decided only by _scene_variant.
        assert HyMemoryClient._scene_variant("coding") == "coding"
        assert HyMemoryClient._scene_variant("working") == "working"
        assert HyMemoryClient._scene_variant("chat") == "working"
        assert HyMemoryClient._scene_variant(None) == "working"


# ════════════════════════════════════════════════════════════════════
# 2. Choosing the extraction spec
# ════════════════════════════════════════════════════════════════════


class TestPromptSelection:
    def test_default_is_working(self):
        # ★ Regression gate: with no scene passed it must still be the working one. Already-persisted working
        #   evaluations rely on this to stay untouched.
        e = CodingMemoryExtractor(_FakeProvider())
        assert e.scene == "working"
        assert e._prompt_tmpl is EXTRACT_PROMPT

    def test_coding_picks_coding_prompt(self):
        e = CodingMemoryExtractor(_FakeProvider(), scene="coding")
        assert e.scene == "coding"
        assert e._prompt_tmpl is EXTRACT_PROMPT_CODING

    def test_unknown_scene_falls_back_to_working(self):
        # No exception: extraction is one-shot and this trace never comes back, so dropping the whole segment
        # over a misspelled argument is not worth it (same trade-off as the raw-source fallback on LLM failure in extract()).
        e = CodingMemoryExtractor(_FakeProvider(), scene="bogus")
        assert e.scene == "working"
        assert e._prompt_tmpl is EXTRACT_PROMPT

    def test_both_prompts_share_placeholders(self):
        # Both specs must be fillable by the same .format() arguments -- extract() has only one format call.
        kw = dict(workspace_present="true", branch_present="true",
                  conversation_block="x")
        assert EXTRACT_PROMPT.format(**kw)
        assert EXTRACT_PROMPT_CODING.format(**kw)


class TestPromptContent:
    """Sections unique to each of the two specs. These pin the **content boundaries**, not wording quality."""

    def test_coding_has_hard_constraint_header(self):
        # It overrides, up front, any wording later on that "wants a narrative".
        assert "HARD CONSTRAINT" in EXTRACT_PROMPT_CODING
        assert "HARD CONSTRAINT" not in EXTRACT_PROMPT
        # Three core requirements
        assert "FUTURE AGENT needs in order to act" in EXTRACT_PROMPT_CODING
        assert "never \"the user\"" in EXTRACT_PROMPT_CODING
        assert "the session was completed" in EXTRACT_PROMPT_CODING

    def test_coding_has_code_work_section(self):
        assert "CODE WORK" in EXTRACT_PROMPT_CODING
        assert "CODE WORK" not in EXTRACT_PROMPT
        # Four things a future agent cannot reconstruct from the code alone
        assert "BRANCH / WORKING LOCATION" in EXTRACT_PROMPT_CODING
        assert "WHICH VERSION WON" in EXTRACT_PROMPT_CODING
        assert "EXACT COORDINATES, NOT PASTED CODE" in EXTRACT_PROMPT_CODING
        assert "VERIFIED vs MERELY WRITTEN" in EXTRACT_PROMPT_CODING

    def test_coding_marks_superseded_version_phrases(self):
        # Language markers of the user changing their mind -- the chain layer recognises the same set of phrases.
        assert "scratch that" in EXTRACT_PROMPT_CODING

    def test_working_keeps_the_ask_user_exclusion(self):
        # Unique to working: when stalled on asking the user for parameters, experiences: [] is forced.
        # Deliberately absent on the coding side -- those traces are autonomous agent runs with no such stall.
        assert "ASKING THE USER" in EXTRACT_PROMPT
        assert "ASKING THE USER" not in EXTRACT_PROMPT_CODING

    def test_both_keep_evidence_fidelity(self):
        # This is the shared bottom line of both specs: only record actions actually executed. Losing it in either one is a defect.
        for p in (EXTRACT_PROMPT, EXTRACT_PROMPT_CODING):
            assert "only actions actually executed" in p
            assert "Never promote a partial search to" in p


# ════════════════════════════════════════════════════════════════════
# 3. Version-evolution chain layer -- write
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def chain():
    with tempfile.TemporaryDirectory() as d:
        yield HyCodeMem(os.path.join(d, "chain"))


class TestChainText:
    """client._chain_text: messages -> plain text the chain layer can read."""

    def test_renders_lowercase_role_prefixes(self):
        # The shape must match the lowercase prefixes recognised by the chain layer's _USER_LC / _DIFF_END.
        t = HyMemoryClient._chain_text(
            [_FakeMsg("user", "hi"), _FakeMsg("assistant", "ok"),
             _FakeMsg("tool", "out", tool_name="bash")]
        )
        assert t.startswith("user: hi")
        assert "assistant: ok" in t
        assert "[tool:bash] out" in t

    def test_no_index_prefix(self):
        # ⚠️ Must not reuse extractor._format_conversation -- that one prefixes every line with an `[i] ` index
        # (for the LLM to read), while the chain layer's regexes anchor at line starts; any prefix breaks every match.
        t = HyMemoryClient._chain_text([_FakeMsg("user", "hi")])
        assert not t.startswith("[0]")

    def test_tolerates_empty_and_missing_content(self):
        t = HyMemoryClient._chain_text(
            [_FakeMsg("user", None), _FakeMsg("assistant", "")]
        )
        assert isinstance(t, str)

    def test_empty_input(self):
        assert HyMemoryClient._chain_text([]) == ""
        assert HyMemoryClient._chain_text(None) == ""


class TestChainObserve:
    def test_extracts_patch_from_dialog_without_git(self):
        # Storage must work without a local repo -- when the git lookup misses, synthesise from the old/new in the conversation.
        # An agent does not necessarily commit right after editing files, so commit diffs alone cannot be relied on.
        with tempfile.TemporaryDirectory() as d:
            H = HyCodeMem(os.path.join(d, "c"))   # repo_root=None
            n = H.observe(
                HyMemoryClient._chain_text(
                    _edit_msgs("On branch feat/q4, set mult.",
                               "mult = 255.0\n", "mult = 360.0\n")),
                meta={"branch": "feat/q4"},
            )
            assert n >= 1
            assert H.stats()["threads"] == 1

    def test_multiple_versions_land_on_one_thread(self, chain):
        # ★ The reason this layer exists: when one piece of code is edited several versions, each version's original words + the real patch are kept.
        for said, old, new in [
            ("On branch feat/q4, first pass: 255.0 everywhere.",
             "mult = 255.0\n", "mult = 360.0\n"),
            ("Hold on, scratch that — special-case only hue.",
             "mult = 360.0\n", "mult = 360.0 if kind == 'h' else 255.0\n"),
        ]:
            chain.observe(HyMemoryClient._chain_text(_edit_msgs(said, old, new)),
                          meta={"branch": "feat/q4"})
        st = chain.stats()
        assert st["threads"] == 1, "multiple versions of the same symbol must merge into one chain, or version numbering recounts from 1"
        assert st["rounds"] == 2
        assert st["multi"] == 1

    def test_branch_from_meta_beats_guessing(self, chain):
        chain.observe(
            HyMemoryClient._chain_text(
                _edit_msgs("no branch token here", "a = 1\n", "a = 2\n")),
            meta={"branch": "fix/explicit"},
        )
        assert chain.stats()["branches"] == 1

    def test_no_code_change_stores_nothing(self, chain):
        # Pure conversation with no identifiable change -> 0. This is a normal case, not an error.
        assert chain.observe(
            HyMemoryClient._chain_text([_FakeMsg("user", "Good morning")])) == 0
        assert chain.stats()["rounds"] == 0

    def test_empty_text_is_noop(self, chain):
        assert chain.observe("") == 0
        assert chain.observe(None) == 0


# ════════════════════════════════════════════════════════════════════
# 4. Version-evolution chain layer -- read
# ════════════════════════════════════════════════════════════════════


def _seed_two_versions(H, branch="feat/q4",
                       path="qutebrowser/config/configtypes.py"):
    for said, old, new in [
        ("On branch %s, first pass: 255.0 everywhere." % branch,
         "mult = 255.0\n", "mult = 360.0\n"),
        ("Hold on, scratch that — special-case only hue.",
         "mult = 360.0\n", "mult = 360.0 if kind == 'h' else 255.0\n"),
    ]:
        H.observe(HyMemoryClient._chain_text(_edit_msgs(said, old, new, path=path)),
                  meta={"branch": branch})


_FACT = {
    "statement": ("On branch feat/q4, in qutebrowser/config/configtypes.py "
                  "QtColor._parse_value: settled on the hue multiplier."),
    "branch": "feat/q4",
    "files": ["qutebrowser/config/configtypes.py"],
}


class TestChainDecorate:
    def test_block_carries_branch_and_all_versions(self, chain):
        _seed_two_versions(chain)
        blk, used = chain.decorate([dict(_FACT)], instruction="which version won?")
        assert used == [0]
        # The branch name must be in the block -- code an agent changed on the wrong branch would be silently dropped.
        assert "feat/q4" in blk
        # Both versions stay, not just the last one: the discarded version's code must stay recognisable, or it gets "fixed" back.
        assert "v1" in blk and "v2" in blk
        assert "255.0" in blk and "360.0" in blk

    def test_fact_without_branch_falls_back_to_files(self, chain):
        # A fact-layer fact may lack a branch field. If the branch cannot be recognised, fall back to files;
        # giving up outright would leave the diff block permanently empty.
        _seed_two_versions(chain)
        fa = {"statement": "the hue multiplier decision",
              "files": ["qutebrowser/config/configtypes.py"]}
        blk, used = chain.decorate([fa], instruction="hue multiplier")
        assert used == [0]
        assert blk

    def test_unrelated_fact_gets_no_block(self, chain):
        _seed_two_versions(chain)
        blk, used = chain.decorate(
            [{"statement": "something else entirely", "files": ["other/file.py"]}],
            instruction="x")
        assert used == []
        assert blk == ""

    def test_empty_facts(self, chain):
        assert chain.decorate([], instruction="x") == ("", [])

    def test_block_off_switch_disables_layer(self, chain):
        # Ablation switch: degrade coding to a "pure vector two-layer" (= working's retrieval shape) for
        # single-variable A/B. CodeMemoryBench ablation relies on it: removing this layer gives 77/216 -> 45/216.
        _seed_two_versions(chain)
        os.environ["HCM_BLOCK_OFF"] = "1"
        try:
            assert chain.decorate([dict(_FACT)], instruction="x") == ("", [])
        finally:
            os.environ.pop("HCM_BLOCK_OFF", None)


class TestDecorateGlue:
    """client._decorate_with_chain -- the fail-safe of the glue layer."""

    def test_returns_empty_when_chain_unavailable(self):
        import types
        ns = types.SimpleNamespace(
            _chain_store=None, _chain_store_failed=True,
            _get_chain_store=lambda owner="": None,
        )
        assert HyMemoryClient._decorate_with_chain(ns, [dict(_FACT)]) == ("", [])

    def test_store_exception_degrades_not_raises(self):
        # This layer is incremental; on failure it must fall back to the pure vector two-layer instead of failing the whole search.
        import types

        class _Boom:
            def decorate(self, *a, **k):
                raise RuntimeError("boom")

        ns = types.SimpleNamespace(_get_chain_store=lambda owner="": _Boom())
        assert HyMemoryClient._decorate_with_chain(ns, [dict(_FACT)]) == ("", [])


class TestObserveGlue:
    def test_returns_zero_when_chain_unavailable(self):
        import types
        ns = types.SimpleNamespace(_get_chain_store=lambda owner="": None)
        assert HyMemoryClient._observe_chain(ns, [_FakeMsg("user", "hi")]) == 0

    def test_store_exception_degrades_not_raises(self):
        import types

        class _Boom:
            def observe(self, *a, **k):
                raise RuntimeError("boom")

        ns = types.SimpleNamespace(_get_chain_store=lambda owner="": _Boom())
        n = HyMemoryClient._observe_chain(
            ns, _edit_msgs("x", "a = 1\n", "a = 2\n"))
        assert n == 0


# ════════════════════════════════════════════════════════════════════
# 5. working is unaffected
# ════════════════════════════════════════════════════════════════════


class TestWorkingUnchanged:
    """★ The premise of the whole change: the working path is not touched by a single byte."""

    def test_working_prompt_byte_length_pinned(self):
        # Pin the length, not the content: content has its own assertions (30+ in tests/coding/test_extractor.py),
        # and what this guards against is "slipping into the working spec while editing coding".
        #
        # ⚠️ This number must come from an actual len(EXTRACT_PROMPT) measurement; do not derive it from the length of
        #    diff / regex-extracted fragments -- the regex body omits trailing newlines and other boundary characters, so filling it
        #    that way is off by a few bytes, and this assertion would go red even when working was never touched (my first version tripped here).
        assert len(EXTRACT_PROMPT) == 18697, (
            "the working extraction spec was modified. It carries its own measured findings (the three-way"
            "comparison of statement lengths, see its body); changing it requires re-running that A/B first."
        )

    def test_coding_prompt_is_separate_object(self):
        assert EXTRACT_PROMPT is not EXTRACT_PROMPT_CODING

    def test_working_variant_does_not_touch_chain(self):
        # With variant="working" the write side must not call the chain layer -- otherwise a hy_code_chain
        # directory would show up in the working store, and behavior would differ from before the change.
        assert HyMemoryClient._scene_variant("working") == "working"
        assert HyMemoryClient._scene_variant(None) == "working"


# ════════════════════════════════════════════════════════════════════
# 6. Separate stores for the two variants
# ════════════════════════════════════════════════════════════════════


class TestStoreSeparation:
    """★ working and coding read and write **different stores**.

    Why separate stores rather than a variant column with filtering: retrieval **first** takes top-K from the vector collection,
    **then** filters with sqlite. Once both variants' facts share one collection, the vector step is already mixed --
    no later SQL condition can save it, because the other variant already occupies the top-K slots.
    """

    def test_sqlite_path_diverges(self):
        from hy_memory.coding.store import _variant_db_path
        assert _variant_db_path("/d/coding_memory.db", "_cv") == "/d/coding_memory_cv.db"

    def test_working_paths_byte_identical(self):
        # ★ Regression gate: working's paths and collection names must be byte-identical to before variants existed, or already
        #   persisted stores become unreadable (which shows up as "the store is empty").
        from hy_memory.coding.store import (
            _variant_db_path, _derive_fact_collection_name,
            _derive_coding_collection_name,
        )
        assert _variant_db_path("/d/coding_memory.db", "") == "/d/coding_memory.db"
        assert _derive_fact_collection_name("am", "") == "am_working_facts"
        assert _derive_coding_collection_name("am", "") == "am_coding_keys"

    def test_collections_diverge(self):
        from hy_memory.coding.store import (
            _derive_fact_collection_name, _derive_coding_collection_name,
        )
        assert _derive_fact_collection_name("am", "_cv") == "am_working_facts_cv"
        assert _derive_coding_collection_name("am", "_cv") == "am_coding_keys_cv"


# ════════════════════════════════════════════════════════════════════
# 7. Chain-layer data-domain isolation
# ════════════════════════════════════════════════════════════════════


class TestChainOwnerIsolation:
    """★ The thread key must include the owner.

    Without it the thread key is only (branch, file, pkey) -- user A's feat/q4 and user B's
    feat/q4 would merge into one chain, and A could read B's diffs in the injected block. That is a data leak.
    """

    _FACT = [{"statement": "On branch feat/q4, what changed in cfg.py",
              "branch": "feat/q4", "files": ["cfg.py"]}]

    @staticmethod
    def _edit(said, old, new, path="cfg.py"):
        return [
            _FakeMsg("user", said),
            _FakeMsg("tool",
                     f"path='{path}'\nold='''{old}'''\nnew='''{new}'''",
                     tool_name="edit"),
        ]

    def test_owner_key_shape(self):
        assert HyMemoryClient._chain_owner("u1", "repo1") == "u1|repo1"
        # both empty -> "", keeping compatibility with old single-user single-repo data
        assert HyMemoryClient._chain_owner(None, None) == ""
        assert HyMemoryClient._chain_owner("", "") == ""

    def test_two_users_same_branch_do_not_leak(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ch")
            HA = HyCodeMem(p, owner="userA|repo1")
            HA.observe(HyMemoryClient._chain_text(
                self._edit("On branch feat/q4, edited by A", "x = 1\n", "x = 2\n")),
                meta={"branch": "feat/q4"})
            # same store file, same branch, same file name
            HB = HyCodeMem(p, owner="userB|repo1")
            HB.observe(HyMemoryClient._chain_text(
                self._edit("On branch feat/q4, edited by B", "y = 1\n", "y = 2\n")),
                meta={"branch": "feat/q4"})

            blkA, _ = HA.decorate([dict(self._FACT[0])], instruction="q")
            blkB, _ = HB.decorate([dict(self._FACT[0])], instruction="q")
            assert "x = 2" in blkA and "y = 2" not in blkA, "A read B's diff"
            assert "y = 2" in blkB and "x = 2" not in blkB, "B read A's diff"

    def test_empty_owner_is_not_a_public_pool(self):
        # An empty owner must only match an empty owner, otherwise old empty-owner data would wrongly read all new writes.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ch")
            H1 = HyCodeMem(p, owner="u1|r1")
            H1.observe(HyMemoryClient._chain_text(
                self._edit("On branch feat/q4, edited by u1", "x = 1\n", "x = 2\n")),
                meta={"branch": "feat/q4"})
            H0 = HyCodeMem(p)      # owner=''
            blk, used = H0.decorate([dict(self._FACT[0])], instruction="q")
            assert blk == "" and used == []

    def test_stats_scoped_to_owner_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ch")
            for o, var in (("u1|r1", "a"), ("u2|r1", "b")):
                H = HyCodeMem(p, owner=o)
                H.observe(HyMemoryClient._chain_text(
                    self._edit(f"On branch feat/q4, {o}", f"{var} = 1\n", f"{var} = 2\n")),
                    meta={"branch": "feat/q4"})
            H = HyCodeMem(p, owner="u1|r1")
            # By default only count one's own -- the ingestion self-check relies on this; reporting the whole store would overstate it
            assert H.stats()["threads"] == 1
            assert H.stats(all_owners=True)["threads"] == 2


# ════════════════════════════════════════════════════════════════════
# 8. Failure gate for path 3
# ════════════════════════════════════════════════════════════════════


class TestEditFailureGate:
    """When synthesising diffs from conversation, failed edits must not enter the chain.

    The chain is the only record of "which version won" -- mixing in a version that never took effect could make an agent write it back.
    """

    _EDIT = ('path = "cfg.py"\n'
             'old = """mult = 255.0"""\n'
             'new = """mult = 360.0"""')

    def _observe(self, tmpdir, tail, tag):
        H = HyCodeMem(os.path.join(tmpdir, tag))
        t = HyMemoryClient._chain_text([
            _FakeMsg("user", "On branch feat/q4, change mult"),
            _FakeMsg("tool", self._EDIT + "\n" + tail, tool_name="bash"),
        ])
        return H.observe(t, meta={"branch": "feat/q4"})

    @pytest.mark.parametrize("tail,tag", [
        ("Traceback (most recent call last):\n  File \"<stdin>\"\nAssertionError", "as"),
        ("old not in s", "ni"),
        ("OBSERVATION: <returncode>2</returncode>", "rc2"),
        ("SyntaxError: unterminated string literal", "se"),
    ])
    def test_failed_edits_blocked(self, tail, tag):
        with tempfile.TemporaryDirectory() as d:
            assert self._observe(d, tail, tag) == 0

    @pytest.mark.parametrize("tail,tag", [
        ("OBSERVATION: <returncode>0</returncode>", "rc0"),
        ("REPLACED=1", "rp"),
        ("", "none"),          # no result marker at all -> do not block (prefer keeping, never miss a real change)
    ])
    def test_successful_and_unmarked_edits_kept(self, tail, tag):
        with tempfile.TemporaryDirectory() as d:
            assert self._observe(d, tail, tag) == 1

    def test_retry_after_error_is_kept(self):
        """★ A success sign after a failure word -> the retry succeeded; must not block.

        Measured: of 910 path-3 rounds across nine repos, 9 hit a failure word, and **6 of them** were immediately followed by
        returncode>0 / REPLACED=1. Blocking on failure words alone would wrongly drop those 6 real changes.
        """
        with tempfile.TemporaryDirectory() as d:
            assert self._observe(d, "AssertionError\nTry again\nREPLACED=1", "rt") == 1

    def test_failure_patterns_are_case_sensitive(self):
        """★ The matcher must not use re.I -- it would collide with source identifiers.

        My first version scanned `FAILED|...` with re.I and reported 50 failed rounds across nine repos; all 50 were actually
        matches on ansible source function names like `v2_runner_on_failed`, while those rounds'
        <returncode>0</returncode> clearly indicated success.
        """
        from hy_memory.coding.chain_store import _EDIT_FAIL
        assert not _EDIT_FAIL.search("def v2_runner_on_failed(self, result):")
        assert not _EDIT_FAIL.search("    self._handle_failed_task()")
        # Real exception class names must be caught
        assert _EDIT_FAIL.search("\nAssertionError\n")
