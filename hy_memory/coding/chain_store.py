#!/usr/bin/env python3
"""
hy_code_mem -- a general coding memory plugin. It attaches "how many times this piece of code was changed and what each change did" to the vector memory layer.

════════════════════════════════════════════════════════════════════════════
It is a decorator, not a retriever
════════════════════════════════════════════════════════════════════════════
The vector layer (hy_mem / any RAG) decides "which facts to return" -- top-k, thresholds and ranking are all its call.
This module only decides "what follows each fact": for the piece of code that fact describes, all
**changed versions** on this branch (each version's user wording + each version's real diff) are hung under it in order.

No similarity, no top-k, no version selection, no guessing which version is the final answer. The judgement is left to the instruction itself.

════════════════════════════════════════════════════════════════════════════
Why the diff must be stored instead of writing the fact in more detail
════════════════════════════════════════════════════════════════════════════
The vector layer's extraction prompt **explicitly demands cutting code** (hy_memory/coding/extractor.py):
    "A record that omits the branch but quotes 30 lines of diff is a bad trade."
    "EXACT COORDINATES, NOT PASTED CODE" / hard cap ~200 CJK characters
So a fact usually does not record the abandoned version and rarely carries a code snippet.

This trade-off is **right** in the general case -- within a 200-character budget the single token `feat/x` is worth far more than 30 lines
of diff, and code can "be read back". But the premise it assumes has a hole: **the code of the reverted version cannot be read back** --
it is not in the working tree, not in HEAD, only in some intermediate commit.

So this module does not compete for those 200 characters: the extraction prompt keeps every word, and the real patch of every version is stored here with zero LLM.
What it adds is not "extraction quality" but the class of information an LLM summary inevitably loses.

════════════════════════════════════════════════════════════════════════════
Two tables
════════════════════════════════════════════════════════════════════════════
    thread(id, owner, branch, file, symbol, lo, hi, n_rounds, updated_at)
        one thread = the history of one block in one function of one file on one branch being changed repeatedly
        owner = isolation key ("<user_id>|<workspace_id>"). **Must be part of the key** -- without it,
        user A's feat/q4 and B's feat/q4 merge into one chain, and A can read B's diff.
        An empty string = unspecified and matches only empty strings (not a shared pool), so single-user legacy stores behave unchanged.
    round(id, thread_id, seq, said, patch, adds, created_at)
        one round = one change. seq = which version
    repo_symbol(name, path, line, kind)
        Whole-repo AST index. Used only for "which file holds the function the instruction names"; it plays no part in chain splitting

`seq` is **how many times observe() has been called**, a pure counter -- no comparison between diffs and no inference about which supersedes which,
so there is no way to guess wrong. Version 1 is simply the first call, a physical fact.

The thread key degrades over three levels. All three tiers below stay **inside one owner**:
    ① hunk header carries a func/def/class name → (owner, branch, file, symbol, line range)
    ② context text only, no function name       → (owner, branch, file, line range)
    ③ bare hunk, line numbers only              → (owner, branch, file)
For arrow functions / object methods git's default hunk header cannot recognize the function name, so they land on ②. ② clusters by line range instead of dropping straight to file granularity -- otherwise every change in the same file
would squeeze into one long chain of a dozen versions, with no way to tell which version corresponds to which change.

repo_symbol is **not** used to look a symbol up by line number to patch ②: that would systematically shift by one symbol,
because line numbers in a patch are post-change while the index was built pre-change. The hunk header is the correct source.
"""
import os
import re
import json
import time
import sqlite3
import subprocess
import collections
import difflib

# Marker on the first line of the injected block -- the host uses it to recognize "this is a multi-line structural block, do not put it in a `- {r}` bullet slot"
BLOCK_MARK = "[Work context]"

# Budget
MAX_CODE_LINES = int(os.environ.get("HCM_CODE_LINES", "6"))    # how many + lines each version shows
MAX_CODE_CHARS = int(os.environ.get("HCM_CODE_CHARS", "480"))   # character cap for the code of one version
# Character cap for the user wording of one version. The openings of the wording across a chain often match,
# and the words that really distinguish versions are usually later, so cutting too short leaves the model unable to tell versions apart.
# ⚠️ Must be tuned together with MAX_BLOCK_CHARS: lengthening the wording while keeping the block cap makes more blocks hit the cap and
# instead triggers wording compression.
MAX_SAID_CHARS = int(os.environ.get("HCM_SAID_CHARS", "800"))
MAX_BLOCK_CHARS = int(os.environ.get("HCM_BLOCK_CHARS", "14000"))  # whole-block cap
# The whole-block cap and BODY_MAX_CHARS must be tuned together; enlarging BLOCK alone is useless: an oversized body pushes the diff lines out.
# The diff is the only clue to "which version changed what" and complements the body ("what the complete shape is"); both are required.
# It should not keep growing either: injecting too much overloads the model.
LINE_MERGE = int(os.environ.get("HCM_LINE_MERGE", "20"))        # how close two line ranges must be to count as the same chain
# How many versions of one chain get code shown in full. Beyond that only the index plus a wording summary stays (the index is never dropped; ordinal references count with it)
SHOW_ALL = int(os.environ.get("HCM_SHOW_ALL", "8"))
# How many chains one fact may carry at most (cut after sorting by token overlap with the instruction).
# Too many chains hit the whole-block cap and compress the per-version wording past recognition; the wording is the core information of this mechanism,
# so it is better to attach one chain less than to shorten it.
# 0 = no cut.
TOPN_THREADS = int(os.environ.get("HCM_TOPN_THREADS", "3"))
# How many **facts** at most get chains. The other facts get none and stay in the bullet list as usual (no information lost, just no diff).
# Block size is roughly linear in the number of facts carrying chains; attaching too many exceeds the whole-block cap and causes overload.
# The chain for the target point only needs attaching once; the extra facts usually carry chains for **other points in the same file** -- that is the noise.
# And since _rank_threads already sorts by instruction tokens, the first fact has the highest chance of hitting the target chain.
TOPN_FACTS = int(os.environ.get("HCM_TOPN_FACTS", "1"))
# Character cap for the "complete code of the last version" at the end of a chain. Injecting too much overloads, so one chain gets a ceiling.
BODY_MAX_CHARS = int(os.environ.get("HCM_BODY_CHARS", "900"))

_HUNK = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@[ \t]*(.*)$', re.M)
_SYM = re.compile(r'(?:func|def|class|function)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)')
# Recognizes the agent's "replace a whole block" pattern in dialogue (used by _patch_from_edits). This is the common
# edit shape in empty-commit rounds: a piece of Python that first reads with open(path), then s.replace(old, new), and finally writes back.
# All three of path/old/new must be captured or the match is skipped -- extracting something wrong is worse than extracting nothing.
# Synthetic diff marker: placed in the context slot of the @@ line (_SYM finds symbols with a regex, extra words do not matter),
# observe uses it to recognize "these line numbers are fake" → merge by symbol instead of by line range.
# Boilerplate lines: identical in every version, zero information about "which version this is" (used by _pick_lines).
# Function/class headers, control-flow keywords, bare brackets, import/package. Assignment and return are **not** included --
# those are exactly where the real differences between versions lie.
_BOILER = re.compile(
    r'^(?:\}|\{|\)|\(|else\b|\}\s*else|end\b'
    r'|(?:func|def|class|function|type|interface|impl)\b'
    r'|(?:for|while|if|switch|case|try|except|catch|with|elif|do)\b'
    r'|(?:import|from|package|use|include|#include)\b'
    r'|(?:continue|break|pass)\s*$'
    r')')
_SYNTH_MARK = "[synth-from-dialog]"
# ── Failure gate for synthetic edits ───────────────────────────────────────────────────────
# _patch_from_edits used to treat a regex hit on path/old/new as "a change happened" without checking
# **whether it succeeded**. Two kinds of misjudgement:
#   · the agent only **proposed** a change (still asking "should I run it") and the user answered "that plan is wrong, do not change it" → stored anyway
#   · the edit **failed** (AssertionError: old not in s) → the failed version also entered the chain
# The consequences are asymmetric: the chain is the only record of "which version won", so mixing in a version that never took effect may lead the agent to write it.
# This contradicts the principle of _patch_from_edits itself ("extracting something wrong is worse than extracting nothing") -- closed here.
#
# The criterion must be case-sensitive, otherwise source identifiers such as `failed` would be misread as edit failures.
_EDIT_FAIL = re.compile(
    r'(?m)^\s*(?:AssertionError|IndentationError|SyntaxError'
    r'|FileNotFoundError|KeyError|ValueError)\b'
    r'|Traceback \(most recent call last\)'
    r'|\bold not in\b|substring not found'
    r'|returncode>[1-9]')
# Success signals must be checked too: an edit may error first and then succeed on retry.
_EDIT_OK = re.compile(r'returncode>0<|REPLACED\s*=\s*1')
# Only a bounded window after the edit action is scanned, so the next round's output is not misattributed to this round.
_EDIT_TAIL_WIN = int(os.environ.get("HCM_EDIT_TAIL_WIN", "600"))
_EDIT_PY = re.compile(
    r'path\s*=\s*["\'](?P<path>[^"\']+)["\'].*?'
    r'old\s*=\s*(?P<q1>"""|\'\'\')(?P<old>.*?)(?P=q1).*?'
    r'new\s*=\s*(?P<q2>"""|\'\'\')(?P<new>.*?)(?P=q2)',
    re.S)
_BR = re.compile(r'\b((?:feat|fix|feature|chore|refactor)/[A-Za-z0-9._\-]+)')
_USER = re.compile(r'(?ms)^USER:[ \t]*(.*?)(?=^(?:ASSISTANT|OBSERVATION|SUBMITTED):|\Z)')
# Also accepts lowercase `user:` / `assistant:` and the `[tool:...]` / `[result:...]` tool markers.
_USER_LC = re.compile(
    r'(?ms)^user:[ \t]*(.*?)(?=^(?:assistant|system|observation):|^\[(?:tool|result)[:\s]|\Z)')
# Fixed prefixes the host adds around the instruction, not what the user is saying
_SHELL = [
    re.compile(r'^\s*(?:user|human)\s*:\s*', re.I),
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  owner TEXT NOT NULL DEFAULT '',
  branch TEXT NOT NULL, file TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT '',
  pkey TEXT NOT NULL DEFAULT '',
  lo INTEGER NOT NULL DEFAULT 0, hi INTEGER NOT NULL DEFAULT 0,
  n_rounds INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0,
  body TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS ix_thread_bf ON thread(owner, branch, file, pkey);
CREATE TABLE IF NOT EXISTS round(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id INTEGER NOT NULL, seq INTEGER NOT NULL,
  said TEXT NOT NULL DEFAULT '', patch TEXT NOT NULL DEFAULT '',
  adds TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_round_t ON round(thread_id, seq);
CREATE TABLE IF NOT EXISTS repo_symbol(
  name TEXT NOT NULL, path TEXT NOT NULL, line INTEGER NOT NULL DEFAULT 0,
  kind TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS ix_sym_name ON repo_symbol(name);
"""


def _clean(s):
    return " ".join((s or "").split())


def _strip_shell(s):
    """Strip the fixed prefixes the host wraps around the instruction, keeping only what the user actually said."""
    out = s or ""
    for p in _SHELL:
        out = p.sub("", out)
    return _clean(out)


class HyCodeMem:
    """One instance per workspace (usually = one repo). observe() writes, recall() reads."""

    def __init__(self, store_dir, repo_root=None, owner=""):
        os.makedirs(store_dir, exist_ok=True)
        self.dir = store_dir
        self.repo = repo_root
        # owner = isolation key. The business key of thread has no user dimension; without owner the same
        # branch/file/pkey from different users/workspaces merges into one thread, leaking data.
        # An empty string means "unspecified", not "shared pool": an empty owner matches only empty owners (see the WHERE in _own),
        # so legacy empty-owner data is not polluted by new writes that carry an owner.
        self.owner = str(owner or "")
        self.db = sqlite3.connect(os.path.join(store_dir, "hy_code.db"),
                                  timeout=30, check_same_thread=False)
        self.db.executescript(_SCHEMA)
        # ── Legacy store migration: add thread.pkey ───────────────────────────────────────────
        # CREATE TABLE IF NOT EXISTS changes nothing about an **existing** table, so stores loaded by
        # an earlier version (key = branch,file,symbol) have no pkey column once opened, while the
        # SELECT in _upsert_thread carries `AND pkey=?` → `OperationalError: no such column: pkey`.
        # ⚠️ This crash **only appears on real diffs**: a fake diff blocked by _get_patch returns 0 right away and
        # never reaches that SELECT at all.
        # After migration the legacy rows have pkey='', so they still merge into one chain (matching the behaviour when they were loaded);
        # getting the split chains requires re-importing; migration only guarantees the old store stays readable.
        _cols = {r[1] for r in self.db.execute("PRAGMA table_info(thread)")}
        if "pkey" not in _cols:
            self.db.execute("ALTER TABLE thread ADD COLUMN pkey TEXT NOT NULL DEFAULT ''")
        # body: the complete code context of the chain's last version. Empty in legacy stores -- the block then falls back to
        # the old "diff only" behaviour instead of crashing; getting the body requires a reload.
        if "body" not in _cols:
            self.db.execute("ALTER TABLE thread ADD COLUMN body TEXT NOT NULL DEFAULT ''")
        # owner: generic isolation key. '' in legacy stores, matching callers that pass no owner.
        # Same as above: without a reload there is no isolation; migration only guarantees the old store opens and is not merged into by other people's data.
        if "owner" not in _cols:
            self.db.execute("ALTER TABLE thread ADD COLUMN owner TEXT NOT NULL DEFAULT ''")
        # Index rebuilding comes after every ALTER -- it references the owner/pkey columns and raises
        # `no such column` while those columns are still missing. This CREATE INDEX is idempotent on legacy stores (IF NOT EXISTS), but an index under the old name
        # may cover only (branch,file,pkey), so the same-named one is dropped first and then rebuilt.
        self.db.execute("DROP INDEX IF EXISTS ix_thread_bf")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_thread_bf "
                        "ON thread(owner, branch, file, pkey)")
        self.db.commit()
        self._commit_cache = None

    # ────────────────────────────── Writes ──────────────────────────────

    def observe(self, mem_text, meta=None):
        """Called once after each round in which the agent changed something. Zero LLM, zero embedding.

        meta may carry branch / file / commit / kind / key / seg -- all optional shortcuts:
        give branch and there is no guessing, give commit and there is no reverse lookup. It runs with none of them too.
        """
        meta = meta or {}
        text = mem_text or ""
        if not text.strip():
            return 0
        if not self._pass_gate(text, meta):
            return 0

        branch = meta.get("branch") or self._guess_branch(text)
        patch = self._get_patch(text, meta)
        if not patch:
            return 0
        said = self._get_said(text)

        n = 0
        for file, symbol, lo, hi, hunk in self._split_hunks(patch):
            if meta.get("file") and file != meta["file"] and not meta.get("_multi"):
                pass  # meta.file is only a hint, not a filter -- one round really may change several files
            # Line numbers of a synthetic diff (_patch_from_edits) are fake (the real ones are unknowable, so 1 is always written).
            # Passing them straight to _upsert_thread would start a second chain for the same symbol because "the ranges do not overlap" --
            # synthetic line numbers are untrustworthy, so merging is by symbol. pkey must also be part of the thread key,
            # otherwise several independent decision points in one file/symbol merge into one chain and the version ordinals lose their meaning.
            tid = self._upsert_thread(branch, file, symbol, lo, hi,
                                      by_symbol=_SYNTH_MARK in hunk,
                                      pkey=meta.get("pkey") or "")
            self._add_round(tid, said, hunk, at=self._round_time(meta))
            # body: the complete shape of that code after this round (including unchanged lines). Overwritten every round →
            # what remains after loading is the full text of the **last version**, exactly the version _body_at needs.
            # When it cannot be fetched nothing changes (old body kept / stays empty); this round is never dropped because of it.
            if self.BODY_CTX > 0 and _SYNTH_MARK not in hunk:
                # Synthetic diff line numbers are fake (always 1); using them for context would fetch the file head -- skip.
                _c = meta.get("commit") or self._commit_for(
                    str(meta.get("key") or meta.get("pkey") or ""), meta.get("seg"))
                _b = self._body_at(_c, file, lo, hi)
                if _b:
                    self.db.execute("UPDATE thread SET body=? WHERE id=?", (_b, tid))
            n += 1
        if n:
            self.db.commit()
        return n

    def _pass_gate(self, text, meta):
        """Noise rounds and grading rounds are not accepted.

        Grading rounds (the final-draft segment) must be blocked -- that is the round being tested on, and storing it leaks the answer to ourselves.
        """
        kind = (meta.get("kind") or "").lower()
        if kind in ("noise", "eval", "final", "s2"):
            return False
        head = text[:24]
        return not (head.startswith("[\u566a\u97f3]") or head.startswith("[\u5b9a\u7a3f"))

    def _guess_branch(self, text):
        # Trailing punctuation is not part of a branch name. The character class includes `.` (branch names really can carry dots, e.g. feat/v1.2),
        # so "On branch feat/No16." would swallow the period, and the lookup would always come back empty.
        # Otherwise trailing punctuation enters the query key and the exact branch lookup misses.
        f = [x.rstrip(".,;:!?") for x in _BR.findall(text)]
        f = [x for x in f if "/" in x and not x.endswith("/")]
        return max(set(f), key=f.count) if f else ""

    def _get_said(self, text):
        """Take only the user's own words. The assistant's restatement is the model's own invention; the original wording is the fact.

        Both cases must be tried; accepting only one fails the match and falls back to text[:400],
        mistaking the ingest template header for the user's wording.
        """
        m = _USER.search(text) or _USER_LC.search(text)
        return _strip_shell(m.group(1) if m else text[:400])

    # ---- patch: two paths ----

    # Marker that the diff ends here -- in host dialogues the next agent action often follows the diff immediately.
    # assistant/tool/result dialogue text may follow the diff directly; those tails would:
    #   · make _hunks treat dialogue lines as hunk content, so line numbers and symbol attribution go wrong with them
    #   · waste patch storage and _core_diff computation for nothing
    # So the text is cut at the first dialogue marker.
    # Both cases of the role markers and tool markers must be recognized, otherwise the whole following dialogue is stored as the patch.
    _DIFF_END = re.compile(
        r'^(?:ASSISTANT:|USER:|OBSERVATION:|SUBMITTED\s*$|```mswea|COMPLETE_TASK_AND_SUBMIT'
        r'|assistant:|user:|system:|\[tool:|\[result:)',
        re.M)

    def _get_patch(self, text, meta):
        """① The text already carries a diff → use it (main path in general use; the diff is at hand right after the change)
           ② none → reverse lookup from git (main path when loading trajectories after the fact)
           ③ git has nothing either → synthesize from the **edit actions** in the dialogue (see _patch_from_edits)
        """
        i = text.find("diff --git")
        if i >= 0:
            body = text[i:]
            m = self._DIFF_END.search(body)
            return body[:m.start()] if m else body
        p = self._patch_from_git(meta)
        if p:
            return p
        return self._patch_from_edits(text)

    def _patch_from_edits(self, text):
        """Synthesize a diff from the agent's "write file" actions in the dialogue.

        Why this path is needed: the git reverse lookup comes up empty. Many rounds have empty commits -- the agent
        clearly wrote files, but the change never reached that round's commit (a later `git checkout` discards uncommitted work, for example).
        Those rounds all yield observe→0, so thread/round cannot be built, versions go missing from the chain, and the injected block
        may be left with only the wrong old version.

        What can be recovered: the common shape in empty-commit rounds is a Python replacement such as `old='''…''' new='''…'''`
        or a variant of the same family carrying `REPLACED=`; both old and new code sit in the dialogue, so the information
        is actually richer than a git diff (it even says "what was changed into what"). The rest -- sed -i / cat heredoc / purely verbal
        confirmations -- cannot be extracted, and nothing is guessed: extracting something wrong is worse than extracting nothing.

        The synthetic diff only needs to survive _split_hunks: line numbers are 1 (the real ones are unknowable, and threads are
        split by symbol+range, so several versions of one symbol still merge into one chain); the symbol name goes into the context slot after @@
        so that _SYM can capture it.
        """
        out = []
        for m in _EDIT_PY.finditer(text or ""):
            path, old, new = m.group("path"), m.group("old"), m.group("new")
            if not path or old is None or new is None or old == new:
                continue
            if self._edit_failed(text, m.end()):
                continue
            sym = ""
            sm = _SYM.search(new) or _SYM.search(old)
            if sm:
                sym = sm.group(0)          # the whole "func Xxx" fragment, so _SYM captures it again in the @@ context
            else:
                # old/new fragments are often the **middle** of a function, so no header is found → locate the enclosing function in the whole text
                seg = text[:m.start()]
                cands = list(_SYM.finditer(seg))
                sym = cands[-1].group(0) if cands else ""
            ol, nl = old.splitlines(), new.splitlines()
            body = "".join(f"-{x}\n" for x in ol) + "".join(f"+{x}\n" for x in nl)
            out.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
                       f"@@ -1,{max(1,len(ol))} +1,{max(1,len(nl))} @@ {sym} {_SYNTH_MARK}\n{body}")
        return "".join(out)

    @staticmethod
    def _edit_failed(text, pos):
        """Did this edit action fail, or is it only a proposal that was never executed?

        pos = end position of the edit-action regex match; the following _EDIT_TAIL_WIN characters are scanned for the outcome.

        Verdict: it counts as a failure only when failure signs appear **and no success sign follows them**.
        Both must be checked: an edit may fail first and then succeed on retry. Looking only at failure words would block real changes.

        A chain polluted by a version that never took effect misleads future agents, so missing an entry beats accepting a wrong one.
        """
        tail = text[pos:pos + _EDIT_TAIL_WIN]
        fm = _EDIT_FAIL.search(tail)
        if not fm:
            return False
        # A success sign follows the failure word → the retry succeeded, do not block
        return not _EDIT_OK.search(tail[fm.end():])

    def _git(self, *args):
        if not self.repo:
            return ""
        try:
            r = subprocess.run(("git",) + args, cwd=self.repo, capture_output=True,
                               text=True, timeout=30, errors="replace")
            return r.stdout if r.returncode == 0 else ""
        except Exception:
            return ""

    def _patch_from_git(self, meta):
        """Reverse-lookup that round's commit by commit message.

        The host commits once after each round, and the message carries that round's identifier (key/seg). A given commit is used directly.
        """
        c = meta.get("commit")
        if not c:
            key, seg = meta.get("key") or meta.get("pkey"), meta.get("seg")
            if not key:
                return ""
            c = self._commit_for(str(key), seg)
        return self._git("show", "--format=", "--unified=3", c) if c else ""

    def _commit_for(self, key, seg):
        """Commits whose message contains key(+seg), taking the seg-th one in chronological order.

        The git log is read into memory once and indexed -- running `git log --grep` separately per round would mean
        hundreds of process starts at 300+ rounds.
        """
        if self._commit_cache is None:
            self._commit_cache = collections.defaultdict(list)
            out = self._git("log", "--reverse", "--format=%H%x01%s", "--all")
            for ln in out.splitlines():
                if "\x01" not in ln:
                    continue
                h, subj = ln.split("\x01", 1)
                self._commit_cache[subj.strip()].append(h)
                # also index by tokens appearing in the message (the subject format may carry a prefix)
                for tok in re.findall(r'[\w.\-/#]+', subj):
                    if len(tok) > 3:
                        self._commit_cache[tok].append(h)
        cand = self._commit_cache.get(f"{key}#{seg}") or self._commit_cache.get(key) or []
        if not cand:
            for k, v in self._commit_cache.items():
                if key in k:
                    cand = v
                    break
        if not cand:
            return ""
        i = int(seg) if isinstance(seg, int) or (isinstance(seg, str) and seg.isdigit()) else 0
        return cand[i] if i < len(cand) else cand[-1]

    # ---- split hunks, decide the thread key ----

    def _split_hunks(self, patch):
        """Split by hunk; each hunk yields (file, symbol, lo, hi, the patch text of that hunk)."""
        out = []
        for blk in re.split(r'(?m)^diff --git ', patch or ""):
            if not blk.strip():
                continue
            m = re.search(r'^a/(\S+) b/(\S+)', blk)
            file = m.group(2) if m else ""
            if not file:
                m2 = re.search(r'(?m)^\+\+\+ b/(\S+)', blk)
                file = m2.group(1) if m2 else ""
            if not file:
                continue
            hs = list(_HUNK.finditer(blk))
            for i, h in enumerate(hs):
                lo = int(h.group(1))
                hi = lo + int(h.group(2) or 1)
                ctx = h.group(3)
                sm = _SYM.search(ctx)
                symbol = sm.group(1) if sm else ""
                end = hs[i + 1].start() if i + 1 < len(hs) else len(blk)
                out.append((file, symbol, lo, hi, blk[h.start():end]))
        return out

    def _upsert_thread(self, branch, file, symbol, lo, hi, by_symbol=False, pkey=""):
        """Reuse the same thread when there is one, otherwise create a new one.

        Within one symbol, chains are split by line range: overlapping ranges, or ranges adjacent within ±LINE_MERGE lines, count as one thing.
        Changing a function signature (the header) and changing parsing logic in the middle are two things and should not share one chain.

        by_symbol=True: line numbers are untrustworthy (synthetic diff from _patch_from_edits) → match by symbol only
        and merge straight into that symbol's existing chain. Mixing "signature change" and "middle change" into one chain is preferable to version indices breaking into two runs.

        ⚠️ **Merging is also required when the symbol changed but the line ranges overlap**.
        Matching only on `symbol=?` means a different symbol always creates a new chain -- yet when **a function is added**
        the symbol changes mid-evolution: in version 1 the new function is not in the repo_symbol index yet, so _symbol_of
        can only attribute it to the enclosing function; from version 2 on it attributes to itself. Overlapping ranges would still give two chains, and _pick_versions
        only picks versions inside a single thread, so the chain the agent sees lacks the real version 1.
        Therefore: look up by symbol exactly first (preferred, preserving the original semantics); if nothing is found, search **the same file and branch**
        by line range; on a hit, merge and narrow the symbol to the more specific one (non-empty preferred).
        """
        # ★ pkey must be in the WHERE. See the pkey comment in observe: without it, N points in one file
        # are strung into one chain and version indices run across points → ordinal references are always wrong.
        cur = self.db.execute(
            "SELECT id, lo, hi FROM thread"
            " WHERE owner=? AND branch=? AND file=? AND symbol=? AND pkey=?",
            (self.owner, branch, file, symbol, pkey)).fetchall()
        if by_symbol and cur:
            tid, line_start, line_end = cur[0]
            self.db.execute("UPDATE thread SET updated_at=? WHERE id=?", (time.time(), tid))
            return tid
        for tid, line_start, line_end in cur:
            if lo <= line_end + LINE_MERGE and hi >= line_start - LINE_MERGE:
                self.db.execute("UPDATE thread SET lo=?, hi=?, updated_at=? WHERE id=?",
                                (min(line_start, lo), max(line_end, hi), time.time(), tid))
                return tid
        # Cross-symbol merging by line range (see the comment above). Only done when **line numbers are trustworthy**:
        # by_symbol=True means the diff is synthetic and its line numbers fake; using fake numbers to hit other symbols would merge blindly.
        if not by_symbol:
            for tid, line_start, line_end, sym2 in self.db.execute(
                    "SELECT id, lo, hi, symbol FROM thread"
                    " WHERE owner=? AND branch=? AND file=? AND symbol!=? AND pkey=?",
                    (self.owner, branch, file, symbol, pkey)).fetchall():
                if lo <= line_end + LINE_MERGE and hi >= line_start - LINE_MERGE:
                    # Narrow the symbol to the **more specific** one, judged by the narrower line range.
                    # When a function is added, the version-1 diff lands inside the enclosing function's range; from version 2 on it lands on itself.
                    # The name must be the narrower one -- the chain label `└ <symbol>` in the injected block is the agent's
                    # only clue to "which function this history belongs to"; taking the enclosing function is the same as not labelling at all.
                    keep = symbol if (hi - lo) < (line_end - line_start) else (sym2 or symbol)
                    self.db.execute(
                        "UPDATE thread SET lo=?, hi=?, symbol=?, updated_at=? WHERE id=?",
                        (min(line_start, lo), max(line_end, hi), keep, time.time(), tid))
                    return tid
        cx = self.db.execute(
            "INSERT INTO thread(owner,branch,file,symbol,pkey,lo,hi,n_rounds,updated_at)"
            " VALUES(?,?,?,?,?,?,?,0,?)",
            (self.owner, branch, file, symbol, pkey, lo, hi, time.time()))
        return cx.lastrowid

    # How many context lines of the complete code of the chain's last version to take. 0 = turn the feature off (fall back to diff only).
    # ⚠️ Taken by **the line range of the diff**, not by finding function boundaries from the symbol name -- the latter has incomplete coverage
    # (Go methods, JS arrow functions, field declarations/constant blocks cannot be carved out), while line ranges work for every language.
    BODY_CTX = int(os.environ.get("HCM_BODY_CTX", "18"))

    def _body_at(self, commit, file, lo, hi):
        """The complete code of lines [lo-CTX, hi+CTX] in file at commit (including unchanged lines).

        ── Why it is required ─────────────────────────────────────────────────────────
        The store keeps only a diff per version, and a diff has **only the changed lines**. Most of the "complete function afterwards" consists of lines
        that were never changed and therefore never entered any version's diff; they cannot be derived by accumulating diffs either.

        Giving only the full text of the last version beats giving full text for every version -- full text per version overloads the model.
        The model can reconstruct any version itself from "last version full text + per-version diffs", so there is no need to spread them all out.
        """
        if not commit or not file or self.BODY_CTX <= 0:
            return ""
        src = self._git("show", f"{commit}:{file}")
        if not src:
            return ""
        lines = src.splitlines()
        a = max(0, (lo or 1) - 1 - self.BODY_CTX)
        b = min(len(lines), (hi or lo or 1) + self.BODY_CTX)
        if b <= a:
            return ""
        return "\n".join(lines[a:b])

    def _add_round(self, tid, said, patch, at=None):
        """seq = number of rounds the thread already has + 1. A pure counter, no diff comparison whatsoever.

        at = the time this round **actually happened** (not the load time). When loading after the fact, time.time() values crowd into a few seconds,
        and the injected block becomes a row of "+0s" with no information at all.

        ⚠️ **Multiple hunks of the same round must be merged into one version, not recorded as one version each.**
        One dialogue round often changes several places of a function at once (signature + middle + return); _hunks cuts out N hunks, and
        previously each hunk called _add_round once, so the same said was recorded as N versions.
        Consequence: the meaning of "version N" is polluted -- the injected block shows a row of versions with different content but identical wording,
        which both eats budget and leaves the agent unable to count which version "the first one" actually was.
        Approach: when the same thread + same said already exists, **append the new hunk into that version's patch** without adding a seq.
        """
        row = self.db.execute(
            "SELECT id, seq, patch, adds FROM round WHERE thread_id=? AND said=? ORDER BY seq DESC LIMIT 1",
            (tid, said)).fetchone()
        new_adds = [ln[1:] for ln in patch.split("\n")
                    if ln.startswith("+") and not ln.startswith("+++") and ln[1:].strip()]
        if row:
            rid, seq, old_patch, old_adds_j = row
            merged_patch = (old_patch or "") + ("\n" if old_patch else "") + patch
            old_adds = json.loads(old_adds_j or "[]")
            adds = list(dict.fromkeys([a.strip() for a in old_adds]
                                      + [a.strip() for a in new_adds]))
            self.db.execute("UPDATE round SET patch=?, adds=? WHERE id=?",
                            (merged_patch, json.dumps(adds, ensure_ascii=False), rid))
            return
        seq = self.db.execute("SELECT COUNT(*) FROM round WHERE thread_id=?",
                              (tid,)).fetchone()[0] + 1
        # Deduplicate but keep order -- repeated return/assignment lines within one round waste budget
        adds = list(dict.fromkeys(a.strip() for a in new_adds))
        self.db.execute(
            "INSERT INTO round(thread_id,seq,said,patch,adds,created_at) VALUES(?,?,?,?,?,?)",
            (tid, seq, said, patch, json.dumps(adds, ensure_ascii=False), at or time.time()))
        self.db.execute("UPDATE thread SET n_rounds=?, updated_at=? WHERE id=?",
                        (seq, time.time(), tid))

    def _round_time(self, meta):
        """The time this round actually happened. The author date of that round's git commit comes first;
        falling back to the current time when unavailable (in real-time use the two are the same anyway)."""
        c = meta.get("commit")
        if not c:
            key, seg = meta.get("key") or meta.get("pkey"), meta.get("seg")
            c = self._commit_for(str(key), seg) if key else None
        if c:
            out = self._git("show", "-s", "--format=%at", c).strip().splitlines()
            if out:
                try:
                    return float(out[0])
                except ValueError:
                    pass
        return time.time()

    # ---- Whole-repo symbol index (optional) ----

    def index_repo(self, root=None, exts=(".py", ".go", ".ts", ".js", ".java", ".rb")):
        """Build the whole-repo AST index. Used only for "which file holds the function the instruction names"; it plays no part in chain splitting."""
        root = root or self.repo
        if not root:
            return 0
        pat = re.compile(
            r'(?m)^[ \t]*(?:export\s+)?(?:async\s+)?'
            r'(?:func|def|class|function|type|interface)\s+'
            r'(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)')
        rows, n = [], 0
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if not d.startswith(".")
                      and d not in ("node_modules", "vendor", "dist", "build", "__pycache__")]
            for fn in fns:
                if not fn.endswith(exts):
                    continue
                p = os.path.join(dp, fn)
                try:
                    with open(p, "r", errors="replace") as f:
                        src = f.read()
                except Exception:
                    continue
                rel = os.path.relpath(p, root)
                for m in pat.finditer(src):
                    rows.append((m.group(1), rel, src.count("\n", 0, m.start()) + 1, ""))
                    n += 1
        if rows:
            self.db.execute("DELETE FROM repo_symbol")
            self.db.executemany("INSERT INTO repo_symbol VALUES(?,?,?,?)", rows)
            self.db.commit()
        return n

    # ────────────────────────────── Reads ──────────────────────────────

    def decorate(self, facts, instruction="", scope=None):
        """Core interface: attach rounds to the facts the vector layer selected.

        facts: [{"statement": str, "branch": str, "files": [str], "status": str}, ...]
                order = the order the vector layer gave, not reordered (its top-k is its own business).
        scope: optional **same-repo filter**. When given, only threads whose branch/file path belongs to this scope are attached.
        Returns (block, used_idx):
                block    = the multi-line structural block to place **before** the bullet list ("" when no round can be attached)
                used_idx  = indices of the facts already hung into the block; the host should remove these from the bullet list
                            so the same fact does not appear twice.

        scope prevents cross-repo pollution in a shared index. Semantically similar facts from different repos may enter
        the top-k together, but their diffs cannot be useful for the current workspace, so the same-repo filter must run before attaching chains.
        """
        facts = facts or []
        # Switch: HCM_BLOCK_OFF=1 → no block at all, and no fact is removed from the bullets either.
        if os.environ.get("HCM_BLOCK_OFF") == "1":
            return "", []
        picked, used = [], []
        seen_tid = set()
        for i, fa in enumerate(facts):
            # Only the first TOPN_FACTS facts that **actually got a chain** get a block; the rest get none and stay in the
            # bullet list as usual (no information lost, just no diff). Block size is roughly linear in the number of facts carrying chains,
            # so the count must be limited to keep the context length under control.
            if TOPN_FACTS > 0 and len(picked) >= TOPN_FACTS:
                break
            th = self._threads_for(fa, scope=scope)
            if scope:
                th = [t for t in th if self._in_scope(t, scope)]
            th = [t for t in th if t[0] not in seen_tid]
            th = self._rank_threads(th, instruction)
            if not th:
                continue
            seen_tid.update(t[0] for t in th)
            picked.append((fa, th))
            used.append(i)
        if not picked:
            return "", []
        return self._render(picked, instruction), used

    def _rank_threads(self, th, instruction):
        """Sort threads by token overlap with the instruction and keep only the first TOPN_THREADS.

        One branch may contain several decision points. Ranking uses only the file names, symbol names and
        code tokens already present in the instruction; never a hidden key the caller did not supply. HCM_TOPN_THREADS=0 disables the cut.
        """
        n = TOPN_THREADS
        if n <= 0 or len(th) <= n:
            return th
        want = self._tokens(instruction)
        if not want:
            return th[:n]
        scored = []
        for t in th:
            body = " ".join(
                f"{sd} {' '.join(json.loads(aj or '[]'))}"
                for sd, aj in self.db.execute(
                    "SELECT said,adds FROM round WHERE thread_id=?", (t[0],)))
            # Scoring counts the number of overlaps only; ties keep the original order (updated_at) -- stable sort
            scored.append((len(want & self._tokens(body)), t))
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:n]]

    @staticmethod
    def _in_scope(row, scope):
        """Whether a thread row belongs to scope. row = (id, file, symbol, lo, hi, n_rounds).

        scope is usually the repo name. A hit on the first or on any other path segment counts, to stay compatible with
        layouts such as `project/...` and `lib/project/...`.
        ⚠️ No fuzzy matching: when scope is not recognized the row is **let through** (returns True), so this repo's own rows are not filtered out --
        keeping noise is acceptable, losing the only useful row is not.
        """
        f = (row[1] or "").replace("\\", "/")
        if not f or not scope:
            return True
        segs = set(f.split("/"))
        s = scope.lower()
        return (s in {x.lower() for x in segs}
                or f.lower().startswith(s + "/")
                or f"/{s}/" in f.lower())

    def _threads_for(self, fa, scope=None):
        """One fact → the threads it talks about. Intersection of three conditions, all exact lookups, no similarity.

        scope: used as the **last fallback** only when neither branch nor files can be recognized (requires HCM_SCOPE_FALLBACK=1).
               See the comment in the else branch below.
        """
        # The fact returned by the vector layer does not necessarily carry a branch field, so the statement text is scanned too;
        # when nothing is recognized, nothing is guessed.
        br = fa.get("branch") or self._guess_branch(fa.get("statement") or "")
        files = fa.get("files") or []
        st = fa.get("statement") or ""
        base = {os.path.basename(f) for f in files if f}
        rows = []
        if br:
            rows = self.db.execute(
                "SELECT id,file,symbol,lo,hi,n_rounds FROM thread"
                " WHERE owner=? AND branch=? ORDER BY updated_at",
                (self.owner, br)).fetchall()
        if not rows and base:
            # Fall back to files when there is no branch or the branch lookup is empty. branch and files must not be made into
            # mutually exclusive branches, because the fact layer and the chain layer may each be partially missing data.
            _q = ",".join("?" * len(base))
            rows = self.db.execute(
                f"SELECT id,file,symbol,lo,hi,n_rounds FROM thread"
                f" WHERE owner=? AND (file IN ({_q}) OR "
                + " OR ".join(["file LIKE ?"] * len(base)) +
                ") ORDER BY updated_at",
                (self.owner,) + tuple(base) + tuple(f"%/{b}" for b in base)).fetchall()
        if not rows and scope and os.environ.get("HCM_SCOPE_FALLBACK") == "1":
            # Last fallback: take all chains of this repo by scope. Off by default because it widens the candidate set; before enabling it
            # confirm that the data source cannot contain the current task's answer or another self-linking leak.
            rows = [r for r in self.db.execute(
                        "SELECT id,file,symbol,lo,hi,n_rounds FROM thread"
                        " WHERE owner=? ORDER BY updated_at", (self.owner,))
                    if self._in_scope(r, scope)]
        if not rows:
            return []
        # file: hits on the fact's files (full path or basename); when the fact gives no files there is no file-level filter
        cand = [r for r in rows if (not base) or r[1] in files
                or os.path.basename(r[1]) in base] or rows
        # symbol: the statement names this symbol → **rank it first**, instead of filtering the rest out.
        # ⚠️ No hard filter: the fact's statement is a natural-language summary, **whether it mentions the symbol name is uncertain**,
        # using it as a hard condition would leave points whose chain is in the store and whose file matches unable to attach to their own chain.
        named = [r for r in cand if r[2] and re.search(r'\b%s\b' % re.escape(r[2]), st)]
        if named:
            _rest = [r for r in cand if r not in named]
            return named + _rest
        return cand

    def _render(self, picked, instruction):
        """Render the injected block.

        The budget is **split evenly** across facts, not first-come-first-served, so a long first chain cannot crowd out the later facts.

        Long chains (more than SHOW_ALL versions) are not fully expanded: the relevant versions are picked by "token overlap of that version's code/wording with the instruction"
        and shown **in full**, while the other versions keep only a one-line said summary. Every version keeps its index --
        because references such as "use the version I changed the second time" must be countable; a gap in the numbering breaks them.
        """
        want = self._tokens(instruction)
        files = list(dict.fromkeys(t[1] for _, th in picked for t in th))
        # branch cannot come from the fact alone: the fact layer may have no branch while the thread itself does. The block header must
        # recover branch from the attached threads, so the caller does not lose what it needs to switch branches.
        _br = picked[0][0].get("branch") or ""
        if not _br:
            for _fa, _th in picked:
                _b = self._branch_of_threads(_th)
                if _b:
                    _br = _b
                    break
        _head = f"{BLOCK_MARK} " + (f"branch {_br} · " if _br else "") + " / ".join(files)[:160]
        L = [_head]
        share = max(600, (MAX_BLOCK_CHARS - sum(len(x) for x in L)) // max(1, len(picked)))

        for fa, th in picked:
            seg = ["\n- " + _clean(fa.get("statement") or "")]
            for tid, file, symbol, lo, hi, n in th:
                where = symbol if symbol else f"{os.path.basename(file)}:L{lo}-{hi}"
                seg.append(f"    └ {where}  (changed {n} times on this branch)")
                rs = self.db.execute(
                    "SELECT seq,said,patch,created_at,adds FROM round WHERE thread_id=? ORDER BY seq",
                    (tid,)).fetchall()
                t0 = rs[0][3] if rs else 0
                full = self._pick_versions([(r[0], r[1], r[2]) for r in rs], want)
                # Replacement pairs + revocation relations, computed for the whole chain at once (see _rev_lines)
                _rv = self._rev_lines([(r[0], r[1], r[2]) for r in rs])
                for seq, said, patch, ct, adds_j in rs:
                    mark = "   ← current code" if seq == n and n > 1 else ""
                    # Relative time: states how far "this version is from version 1". A bare unix timestamp would need the model to convert it,
                    # while a relative quantity it reads directly -- 3 minutes apart and 2 hours apart say different things (the latter means something else happened in between).
                    # The order itself is expressed by `version N` (seq is the observe call counter, a physical fact); the time is supplementary.
                    dt = self._reltime(ct - t0) if seq > 1 else ""
                    # Revocation annotation: "this version reverted version N". Final-draft instructions often refer by ordinal
                    # ("back to the very first version" / "not the first one, the one from the second change"); with this line the agent
                    # no longer has to compare the versions itself to work out which one negated which.
                    _rl, _rev = _rv.get(seq, ([], []))
                    rvk = ("   ← reverted v" + ", v".join(str(s) for s in _rev)) if _rev else ""
                    if seq not in full:
                        # Drop the code but keep the index and a one-line summary -- the index must never be missing
                        seg.append(f'       v{seq}{dt} · "{said[:60]}"{rvk}{mark}')
                        continue
                    seg.append(f'       v{seq}{dt} · "{said[:MAX_SAID_CHARS]}"{rvk}{mark}')                    # ⚠️ Trade-offs between the three code line formats:
                    #   ① only `+ added lines` (adds)   -- shortest, but loses "what was changed into what".
                    #      If all three versions show `+ return X`, for instance, the agent assumes three coexisting lines,
                    #      when they are really three replacements of the same line, so it misidentifies "the first version".
                    #   ② `-/+` two-line contrast (HCM_DIFF_VIEW=1) -- full semantics but twice as long → under an evenly split budget it
                    #      easily pushes other facts out of the block, so it is kept only as a debug switch.
                    #   ③ `~ old → new` on one line (current default) -- length ≈①, semantics ≈②.
                    # HCM_ADDS_ONLY=1 falls back to ①.
                    if os.environ.get("HCM_DIFF_VIEW") == "1":
                        lines = self._pick_lines(self._core_diff(patch), want)
                    elif os.environ.get("HCM_ADDS_ONLY") == "1":
                        try:
                            lines = ["+ " + x for x in json.loads(adds_j or "[]")][:MAX_CODE_LINES]
                        except Exception:
                            lines = []
                    else:
                        lines = self._pick_lines(_rl, want) if _rl else []
                    used_c = 0
                    for a in lines:
                        if used_c + len(a) > MAX_CODE_CHARS:
                            break
                        seg.append(f"              {a}")
                        used_c += len(a)
                # ── End of chain: complete code of the last version ────────────────────────────────
                # Each version shows only diffs, and a diff contains no "lines that were never changed" (see _body_at). Giving the full text of the last
                # version lets the model compute: full text − per-version diffs = any version.
                _bd = self.db.execute("SELECT body FROM thread WHERE id=?", (tid,)).fetchone()
                _bd = (_bd[0] if _bd else "") or ""
                if _bd:
                    if len(_bd) > BODY_MAX_CHARS:
                        _bd = _bd[:BODY_MAX_CHARS] + "\n...(truncated)"
                    seg.append(f"       ── Final shape of this code after version ({n}) landed "
                               f"(the diffs of the earlier versions are traced back against it) ──")
                    for _l in _bd.splitlines():
                        seg.append(f"         {_l}")
            # ── Cut order when over budget: compress the wording first, drop code lines second ──────────────────────
            # The reverse must not be used (popping every code line first and compressing the wording afterwards): what a final draft needs is the **code**,
            # while the wording is only the clue to "which version this is". Cutting code first = cutting the answer and keeping the clue.
            # The wording still distinguishes versions once compressed to a certain length (it already carries file name + symbol name as discriminators), so it goes first,
            # and only when it is compressed to the floor and still over budget do code lines start being dropped.
            # ⚠️ Compression floor is 240 and no lower: squeezing further basically voids the wording, and the wording is the core of this mechanism
            # (the fact states only the final state, said carries what each version thought at the time). Better to attach one chain less than to compress the wording
            # past recognition.
            _SAID_TIERS = (240,)
            for _cut in _SAID_TIERS:
                if sum(len(x) for x in seg) <= share:
                    break
                for _i, _x in enumerate(seg):
                    _mm = re.match(r'^(\s+v\d+[^·]*· ")(.*)("[^"]*)$', _x)
                    if _mm and len(_mm.group(2)) > _cut:
                        seg[_i] = _mm.group(1) + _mm.group(2)[:_cut] + "…" + _mm.group(3)
            # Wording compressed to the floor and still over budget → only then drop code lines from the tail (version index lines are never dropped)
            while sum(len(x) for x in seg) > share and any(
                    # `~` is the replacement-pair prefix from _rev_lines (see its comment); it belongs to the same "code line" class as +/-
                    x.startswith("              ") and x.strip()[:1] in "+-~" for x in seg):
                for i in range(len(seg) - 1, -1, -1):
                    if seg[i].startswith("              ") and seg[i].strip()[:1] in "+-~":
                        seg.pop(i)
                        break
            # The body is touched last (it is the "complete code" section, indented 9) -- lines are dropped from the tail, keeping the start
            # (the function signature is at the top and is more useful than the last lines); when only the title line is left, the title goes too.
            while sum(len(x) for x in seg) > share and any(
                    x.startswith("         ") and not x.startswith("          ") for x in seg):
                for i in range(len(seg) - 1, -1, -1):
                    if seg[i].startswith("         ") and not seg[i].startswith("          "):
                        seg.pop(i)
                        break
            if not any(x.startswith("         ") and not x.startswith("          ") for x in seg):
                seg = [x for x in seg if "Final shape of this code" not in x]
            L.extend(seg)
        # Fallback: block still over budget → keep compressing the wording summaries (indices and code lines untouched)
        # Fallback: block still over budget → compress to 240 and stop (same floor as _SAID_TIERS, not down to unrecognizable).
        # This used to be (80,40,24,14) -- that squeezed 381 versions down to 80 characters; see the comment above.
        for cut in (240,):
            if sum(len(x) for x in L) + len(L) <= MAX_BLOCK_CHARS:
                break
            for i, x in enumerate(L):
                mm = re.match(r'^(\s+v\d+[^·]*· ")(.*)("[^"]*)$', x)
                if mm and len(mm.group(2)) > cut:
                    L[i] = mm.group(1) + mm.group(2)[:cut] + "…" + mm.group(3)
        # ── The branch hint goes at the **end of the block**, not at the head ────────────────────────────────────────
        # Keeping the branch hint at the block end puts it near the following instruction; a hint at the start of a long block is more easily overridden by later candidates.
        if _br:
            L.append(f"\n⚠ All the changes above are on branch `{_br}`. **Run `git checkout {_br}` first** before doing anything --"
                     f"the other `feat/...` branches mentioned elsewhere below are unrelated to this task; do not switch to them.")
        return "\n".join(L)

    def _branch_of_threads(self, th):
        """Take the branch from the attached threads -- by **majority vote**, not the first entry.

        The attached chains may come from several tasks on the same file, and the first in order need not represent the current branch. Majority voting is
        steadier; when no standard branch can be recognized an empty string is returned, so the block header degrades naturally to no branch hint.
        """
        if not th:
            return ""
        ids = [str(t[0]) for t in th]
        q = ",".join("?" * len(ids))
        cnt = collections.Counter()
        for (b,) in self.db.execute(
                f"SELECT branch FROM thread WHERE id IN ({q})", tuple(ids)):
            if b and _BR.search(b):
                cnt[b] += 1
        if not cnt:
            return ""
        # On a tie, take the first of the candidates (it has the highest token overlap with the instruction)
        top = cnt.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            for t in th:
                row = self.db.execute("SELECT branch FROM thread WHERE id=?", (t[0],)).fetchone()
                if row and row[0] in cnt:
                    return row[0]
        return top[0][0]

    @staticmethod
    def _tokens(s):
        return {t.lower() for t in re.findall(r'[A-Za-z_$][\w$]{2,}', s or "")}

    def _pick_versions(self, rs, want):
        """Which versions of a long chain are shown in full (with code).

        Up to SHOW_ALL versions are all expanded. Beyond that they are scored by token overlap with the instruction and the top SHOW_ALL are taken,
        while **version 1 and the last version are always kept** -- version 1 is what references like "the very first version" point at,
        and the last version is the current code; neither end may be dropped.
        """
        n = len(rs)
        if n <= SHOW_ALL:
            return {r[0] for r in rs}
        scored = []
        for seq, said, adds_j in rs:
            try:
                body = said + " " + " ".join(json.loads(adds_j or "[]"))
            except Exception:
                body = said
            scored.append((len(want & self._tokens(body)), seq))
        scored.sort(key=lambda x: (-x[0], x[1]))
        keep = {rs[0][0], rs[-1][0]}
        for _, seq in scored:
            if len(keep) >= SHOW_ALL:
                break
            keep.add(seq)
        return keep

    def _rev_lines(self, rs):
        """Render the versions of one chain as "replacement pairs + revocation relations". → {seq: (lines, revokes)}

        lines   : the code lines this version shows, shaped like `~ old → new` (a replacement at one spot) or `+ new` (pure addition)
        revokes : which versions this one reverts (list of seq)

        ── Why `~ old → new` instead of two `-/+` lines ────────────────────────────
        Two `-/+` lines are longer than a single replacement line and would crowd other facts out of the evenly split budget.
        But giving only `+` loses the semantics: when three versions all render as `+ return X`, the agent cannot tell whether these are "three
        versions of the same line" or "three lines coexisting".
        `~ old → new` expresses the replacement on one line: length close to a single `+` line, semantics equivalent to `-/+`.

        The revocation relation uses `Δt_del ∩ Δi_add ≠ ∅`: when the lines deleted now intersect lines added earlier, that version is marked
        as revoked. This relation is decided on the ingest side, with no extra LLM reasoning.
        """
        seq_add = {}      # seq -> lines added by that version (stripped)
        out = {}
        for seq, said, patch in rs:
            core = self._core_diff(patch or "")
            dels = [line[1:].strip() for line in core if line[:1] == "-" and line[1:].strip()]
            adds = [line[1:].strip() for line in core if line[:1] == "+" and line[1:].strip()]
            # Revocation: lines this version deleted that earlier versions had added (computed before registering, so it cannot hit itself)
            rev = sorted(s for s, a in seq_add.items() if s < seq and (set(dels) & a))
            seq_add[seq] = set(adds)
            # Pairing: the equal-length part of deletions and additions is paired positionally into replacement pairs; the surplus stays separate
            k = min(len(dels), len(adds))
            lines = [f"~ {dels[i]} → {adds[i]}" for i in range(k)]
            lines += [f"- {x}" for x in dels[k:]]
            lines += [f"+ {x}" for x in adds[k:]]
            out[seq] = (lines, rev)
        return out

    def _core_diff(self, patch):
        """Extract the **lines that actually changed** (with -/+ prefixes) from a version's patch, instead of only the added lines.

        Why diffs are shown instead of only adds: with `+` lines only, the model learns "what this version is"
        but not "**what it was changed from**" -- and versions are distinguished precisely by the differences between them.
        With adds you only see `+ Split(packVer, "_")`,
        while with the diff you see the pair `- Split(packVer, "-")` / `+ Split(packVer, "_")`, which discriminates completely differently.

        Why compression is needed: agents sometimes use a "whole-function replacement" style, so the patch deletes the entire function and adds it back
        while only a few spots really change. A whole block does not fit the budget and the key lines get cut by the window, which is worse than giving nothing.
        difflib is run once more over the -/+ sides so only the genuinely discriminating information remains.
        """
        if not patch:
            return []
        old = [line[1:] for line in patch.splitlines()
               if line.startswith("-") and not line.startswith("---")]
        new = [line[1:] for line in patch.splitlines()
               if line.startswith("+") and not line.startswith("+++")]
        if not old:                      # pure addition (new function), no old side to compare against
            return ["+ " + x.strip() for x in new]
        core = [line for line in difflib.unified_diff(old, new, lineterm="", n=0)
                if line[:1] in "+-" and line[:3] not in ("---", "+++")]
        if not core:                     # both sides identical (pure formatting/position change)
            return []
        return [line[:1] + " " + line[1:].strip() for line in core]

    @staticmethod
    def _reltime(sec):
        """How long after version 1. What the model gets is a directly readable quantity, not a unix timestamp."""
        sec = max(0, int(sec))
        if sec < 90:
            return f"(+{sec}s)"
        if sec < 5400:
            return f"(+{sec // 60}min)"
        if sec < 172800:
            return f"(+{sec // 3600}h)"
        return f"(+{sec // 86400}d)"

    def _pick_lines(self, lines, want):
        """Which lines of one version are shown. The input is the -/+ lines from _core_diff (no longer adds). **Not simply the first N lines.**

        Taking the first N lines would systematically kill the discriminating information: in a whole-function replacement the leading adds are usually
        function headers and for/if boilerplate, while the lines identifying "which version this is" fall exactly outside the window -- some versions'
        discriminating lines get shown and others get cut, which is worse than giving nothing.

        Ranking rules (stable; equal tiers keep the original code order):
          ① Lines overlapping instruction tokens -- but **do not count on it**: _tokens only catches English identifiers, while instructions are often
             colloquial ("after splitting, stitch the earlier segments back together"), so the intersection is usually empty; it is only a bonus.
          ② Discriminating lines: substantive statements such as assignments, returns and calls.
          ③ Boilerplate lines: function headers, for/if/else, bare brackets -- identical in every version and zero information about "which version".
        """
        if len(lines) <= MAX_CODE_LINES:
            return lines
        def rank(a):
            s = a.strip().lstrip("+-~").strip()  # Strip the diff/replacement prefix before judging whether it is boilerplate
            if want & self._tokens(a):
                return 0
            if _BOILER.match(s):
                return 2
            return 1
        idx = sorted(range(len(lines)), key=lambda i: (rank(lines[i]), i))[:MAX_CODE_LINES]
        return [lines[i] for i in sorted(idx)]      # Display back in the original code order


    # ---- Convenience interfaces ----

    def recall(self, instruction, facts=None):
        """Use this when only the block is wanted and deduplication does not matter. With facts=None it degrades to detecting the branch from the instruction."""
        if facts is not None:
            return self.decorate(facts, instruction)[0]
        br = self._guess_branch(instruction)
        if not br:
            return ""
        return self.decorate([{"statement": instruction, "branch": br, "files": []}],
                             instruction)[0]

    def stats(self, all_owners=False):
        """Store size. By default only **this owner** is counted, matching the scope decorate sees.

        all_owners=True counts the whole store -- for operations/troubleshooting. Routine reporting stays with the current owner by default,
        so data from other isolation domains is not counted into the current import size.
        """
        def q(sql, params=()):
            return self.db.execute(sql, params).fetchone()[0]

        if all_owners:
            return dict(threads=q("SELECT COUNT(*) FROM thread"),
                        rounds=q("SELECT COUNT(*) FROM round"),
                        symbols=q("SELECT COUNT(*) FROM repo_symbol"),
                        branches=q("SELECT COUNT(DISTINCT branch) FROM thread"),
                        multi=q("SELECT COUNT(*) FROM thread WHERE n_rounds>1"))
        o = (self.owner,)
        return dict(
            threads=q("SELECT COUNT(*) FROM thread WHERE owner=?", o),
            # The round table has no owner column (it hangs off thread), so a JOIN is used
            rounds=q("SELECT COUNT(*) FROM round r JOIN thread t ON r.thread_id=t.id"
                     " WHERE t.owner=?", o),
            symbols=q("SELECT COUNT(*) FROM repo_symbol"),
            branches=q("SELECT COUNT(DISTINCT branch) FROM thread WHERE owner=?", o),
            multi=q("SELECT COUNT(*) FROM thread WHERE owner=? AND n_rounds>1", o),
        )

    def close(self):
        try:
            self.db.commit()
            self.db.close()
        except Exception:
            pass
