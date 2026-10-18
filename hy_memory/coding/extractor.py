# -*- coding: utf-8 -*-
"""
Coding Memory - Extractor

One LLM call produces the two-layer drafts for the same trace:

- Experience layer (extracted only when valuable): task / search_keys / solution /
  boundary_envs / confidence. The value bar is strict; trivial ops like rename a file should not become
  memories; only valuable problems likely to recur in the future are extracted.
- Fact layer (the model may return 0 or 1): statement / files. statement is the summary of this session --
  what the user wanted / how it went roughly / the outcome. Whether a verbatim-text fallback is stored when the model produces nothing
  is controlled by `MEMORY_CODING_VERBATIM_FALLBACK_ENABLED`; off by default.

Reuses LLMConfig.model (the same LLM config shared with the chat extractor).
"""

import json
import logging
import time
from typing import List, Optional, Dict, Any, Tuple

from ..agent.llm_provider import LLMProvider
from ..pipelines.base import ChatMessage
from ..utils.lang_detect import is_chinese
from .preproc import (
    extract_files,
    keep_last_k_user_turns,
    prepare_extractor_messages,
)
from .judge import _store_coding_log
from .types import CodingMemoryDraft, WorkingFactDraft



logger = logging.getLogger(__name__)


def _resolve_configured_temperature(llm_provider) -> Optional[float]:
    """Resolve the temperature from MemoryConfig.llm; None = omit the parameter from the request."""
    config = getattr(llm_provider, "config", None)
    llm = getattr(config, "llm", None) if config is not None else None
    if llm is not None:
        if getattr(llm, "omit_temperature", False):
            return None
        temp = getattr(llm, "temperature", None)
        if temp is not None:
            return float(temp)
    inner = getattr(llm_provider, "_llm_config", None)
    temp = getattr(inner, "temperature", None) if inner is not None else None
    return float(temp) if temp is not None else None


# subject is a hard key compared by equality; a whole sentence means extraction went off track -- treat it as "no subject extracted".
_MAX_SUBJECT_CHARS = 80
# details stores raw text; it must fit a whole plan table / config block, but one fact must not eat the entire prompt.
_MAX_DETAILS_CHARS = 4000
# The fallback statement is the retrieval vector; overlength gets diluted by irrelevant content. One or two paragraphs of the user's own words suffice.
# The same reasoning bounds the normal path's length too: the prompt sets a soft statement cap around 1200 characters,
# reached by "writing givens as data instead of dialogue sentences", not by cutting coverage -- details is not in the retrieval vector,
# and statement is the main readable field once retrieval hits.
_VERBATIM_STATEMENT_CHARS = 600


# ================================================================
# Prompt
# ================================================================

EXTRACT_PROMPT = """\
You extract durable WORKING memories from a task-execution session segment.
A "working" session is any hands-on task where the user operates a computer /
system through tools — this covers BOTH software/code work (writing, debugging,
refactoring, building, deploying, API/library usage) AND system & operations
work (running shell/PowerShell commands, disk/storage analysis or cleanup, file
& folder operations, installing/configuring software, environment/device setup,
hardware/disk health inspection, backup, scheduling/automation, data processing).

You produce TWO layers from the SAME segment:

  EXPERIENCES (procedural, "how to do it") — reusable method, extracted only
  when it clears the VALUE BAR below. A (task, search_keys, solution,
  boundary_envs) bundle.

  FACT (episodic, "what happened this time") — EXACTLY ONE record of THIS
  session: what the user wanted, how it was gone about, how it came out. No
  value bar, but there is never more than one.

The two layers answer different questions and are stored separately. A method
that generalizes goes in EXPERIENCES; the story of this particular session —
only true for this system / this project / right now — goes in the FACT.

═══════════════════════════════════════════════════════════════════
VALUE BAR (EXPERIENCES ONLY) — DO NOT cross this lightly.
═══════════════════════════════════════════════════════════════════
A memory MUST satisfy ALL of:
  (a) Solves a problem likely to recur for the same user
  (b) Solution is non-trivial: combines multiple pieces of info / steps, OR
      carries a decision rationale, OR captures a hard-won workaround, OR is a
      reusable procedure/recipe for an operational task (e.g. how to analyze &
      clean up disk usage, how to batch-organize files, how to inspect hardware
      health, how to set up a scheduled job)
  (c) A future agent / user would prefer looking it up over re-discovering it

IMPORTANT — analysis / diagnostic / investigation tasks COUNT:
A task does NOT have to make a change or "fix" something to be worth remembering.
Read-only / analytical / troubleshooting work is valuable when its PROCEDURE or
CONCLUSION is reusable. Explicitly extract:
  - Analysis / diagnostic recipes: the reusable procedure + what it revealed
    (e.g. "how to analyze which folders eat C-drive space" — the scan commands,
    the gotchas hit, and the finding). "It only produced a report, no change"
    is NOT a reason to skip — the reusable METHOD is the memory.
  - Investigation outcomes, INCLUDING negative / not-found results
    (e.g. "searched folder X across C: and D: with rg / fs_search / SQL index —
    not found anywhere"). A recorded dead-end saves the next agent from repeating
    the same fruitless search, so it clears the bar.
Do NOT reject a memory merely because "the assistant could regenerate the
commands" or "nothing was actually modified". If the procedure or the conclusion
would save future effort, keep it.

DO NOT extract:
  - Truly trivial single-step ops with no reusable procedure ("rename one file",
    "git status", "run ls once")
  - One-off issues already resolved with no transferable lesson
  - Information that is already in the repo/system (don't duplicate package.json)
  - Per-PR / per-session throwaway tasks unless they encode a reusable pattern
  - The act of ASKING THE USER for missing information. Requesting a parameter,
    listing what is unknown, or eliciting required fields is conversational
    behaviour, not a method: every agent already knows how to ask a question, so
    there is nothing to look up later. Never emit an experience whose task or
    search_keys are shaped like "how to ask for X", "gather the missing Y",
    "eliciting required fields", "what to clarify before starting". Note this
    stays out of EXPERIENCES only — that the session ended waiting on the user
    is real state and belongs in the FACT.

If the segment made no substantive progress because it stalled waiting on the
user — the assistant only asked for parameters, or was blocked before doing any
real work — emit "experiences": [] and write the FACT alone. Nothing was
learned yet, so there is no method to store. This matters more than it looks:
an experience distilled from a stall reads as a legitimate skill on retrieval,
and a later segment of the same task will match it and ask the user again
instead of proceeding, so the stall reproduces itself for the rest of the
session. Judge substance by what the tools actually did, not by whether the
turn produced fluent text.

Heuristic: if the user (or a future agent doing the same kind of task) would
want this written down instead of figuring it out again, keep it. A multi-step
operational procedure (even a system/ops or read-only analysis/search one, not
just code) usually qualifies. Only drop it if it is a single command any user
would find via `--help` in 30 seconds AND carries no decision/gotcha worth
remembering.

═══════════════════════════════════════════════════════════════════
EXPERIENCE FIELDS
═══════════════════════════════════════════════════════════════════

task          — Query-shaped name; phrase as how a user would later ASK.
                "publish a release to the xx platform" not
                "publish_release_internal"
                Self-contained (no "this", "the file", "we").

search_keys   — 0 to 5 alternative phrasings capturing important aspects of
                solution that `task` alone wouldn't match. Phrase as questions
                or topics future users would ASK ABOUT. Distinct from task.
                Omit if task already covers everything.

solution      — Complete, self-contained content the user actually needs:
                commands, credential locations (NOT credential VALUES),
                file paths, ordered steps, gotchas, decision rationale.
                Aggregate multi-turn end-to-end work into ONE solution if it
                forms a coherent task; split into multiple memories only when
                the user did unrelated tasks.

boundary_envs — Concrete pinning that solution depends on, multi-line "key: value":
                  runtime / SDK / library + version
                  API endpoint or method signature
                  external service / middleware version
                  env var NAME (not value)
                  config file path
                  host / cluster / region identifier
                If solution is environment-agnostic, leave as "" (empty).
                Purpose: when the underlying SDK/middleware upgrades, future
                consumers can inspect this to decide if solution still applies.

confidence  — 0.0 to 1.0
  1.0   user explicitly stated this as a rule / instruction
  0.7   inferred with good evidence from tool actions
  0.5   weakly inferred; one observation; uncertain

═══════════════════════════════════════════════════════════════════
FACT FIELD — EXACTLY ONE per segment, no value bar
═══════════════════════════════════════════════════════════════════

You write ONE fact for the whole segment — a recoverable state record for this
session, not a polished retrospective and never a list of separate findings. A
future agent must be able to resume from it without mistaking a suggestion,
inference, or partial search for something verified.

subject    — WHO or WHAT this session is about: the single entity the work was
             performed on or for. One short noun phrase, no sentence, no verb.
             Pick the most specific entity that the session is *about*, not the
             tool it used:
               a person's name when the session serves one named person
                 ("Dylan", "Matthew")
               the file / module / symbol when the work targets code
                 ("auth/middleware.ts", "parse_plan_text")
               the service / host / device / drive for ops work
                 ("D:\\WXWork", "redis-cache-prod", "ci.yml")
             This is a HARD KEY, not prose: two sessions about different people
             or different files MUST get different subjects, and two sessions
             about the same one MUST get the same string, spelled identically.
             Prefer the exact identifier as it appears in the trace over a
             description of it. If the segment genuinely has no single subject,
             output "".

             Why it matters: subject is the gate that decides whether a later
             session is allowed to overwrite this one as "the same task,
             continued". Get it wrong and one person's state silently replaces
             another's.

statement  — A compact record of THIS session. Two parts, and the second is the
             one that gets lost.
             Preserve task-scoped givens and concrete constraints instead of
             replacing them with a generic evaluation of the outcome. State
             each rule once: repeated instruction text competes with the data
             the memory must retain.

             PART 1 — what happened, about 300 latin characters (~150 CJK).
             Start with the specific goal and object ("reclaim disk space taken
             by D:\\WXWork", "locate the new_file_search folder"), not a generic
             "fix a bug" / "find a file". Then roughly what was inspected, tried,
             or changed. Then how it ended, with the semantics of exactly one of:
             completed, partially completed, failed, blocked, or
             still awaiting verification. Include negative results and dead ends
             without overstating them.

             PART 2 — the TASK-SCOPED GIVENS: everything the WHOLE task runs on,
             even when this segment did not establish it and even when it feels
             too elementary to state. Bindings (what each name/symbol/label
             stands for), fixed values the task supplies rather than derives,
             conventions adopted (indexing base, units, timezone, defaults), and
             this segment's own result WITH its value ("x = 7", not "the value
             was computed").

             Write PART 2 as data, one item per line, values as given:
               retry budget = 5; timeout 30s; index starts at 0
               step 3 input: n=64, tolerance 1e-6
               slot B: type=private, cost 688.7-721.3 per unit
             A line should be almost entirely values. Keep the trace's notation —
             symbols, units, numbers, spellings — and drop its sentences: strip
             the phrases that announce a given rather than being one ("I would
             prefer", "it should be somewhere between … and …"). They repeat once
             per item, so a 15-item list pays for them 15 times.
             Carry them on EVERY fact; never write "as above" or "same as the
             previous session" — retrieval returns a handful of facts by
             similarity and the referent may not be among them.

             Aim under 1200 characters total. If PART 2 does not fit, fix the
             notation, never the coverage: statement is the retrieval vector AND
             the only field a later segment can read (details is neither embedded
             nor injected), so anything cut from it is gone. Compress the wording
             around a given; never the given.

             The test: if the next segment cannot see the original material, can
             it continue from this record alone? Only a yes counts. A missing
             binding does not fail loudly — the name still looks meaningful, so
             it gets read with its general-usage meaning and the answer that
             follows is fluent and about a different question.

             LANGUAGE — same language as the session trace: statement is the
             retrieval vector and queries arrive in the user's working language,
             so another language pays a cross-lingual penalty on every lookup.

             EVIDENCE FIDELITY — record only actions actually executed and
             evidence explicitly present in the trace. Preserve material
             commands, flags, arguments, errors, and numbers verbatim.
             NEVER complete a command with customary flags, and never turn
             advice or plans into actions:
               "ran Get-ChildItem -Force" → "ran Get-ChildItem"
                 (the trace did not contain `-Force`; do not invent it)
             Scope every negative finding to what was actually searched, and
             enumerate EVERY explicit coverage limit that could invalidate it,
             especially permission failures and depth limits: keep the roots,
             filters, and depth that were in effect ("not found under C:\\ and
             D:\\ at depth 3; some directories were not covered due to Access
             denied"), never widen it to "does not exist". Do not omit one limit
             to make the record prettier.
             Never promote a partial search to global absence.
             A blocked attempt keeps its blocker.
             Preserve the distinction among attempted, observed, verified,
             inferred, and unresolved — express it naturally ("tried … but was
             blocked by Access denied", "suspected …, not yet verified"), not as
             labels. Without tool output or explicit confirmation, NEVER claim a
             fix succeeded, a cause was established, or a path does not exist.

             Do NOT record greetings or pure chatter — that is the only case
             getting "fact": null. A segment that merely stated requirements, or
             stalled before doing work, still established something.

details    — The session's concrete OUTPUT, copied over verbatim. `statement` is
             a summary and a summary necessarily drops the individual values; a
             later lookup almost always wants one specific value, not the gist.
             So whatever addressable result this session produced, reproduce it
             here in its original form and structure:
               a plan / schedule / itinerary → keep every line, every day, every
                 slot, and every price exactly as written
               a config or code block → keep the block
               a command's material output → keep the lines that carry the
                 findings (sizes, counts, versions, error text)
               a structured result (table, list of records) → keep the rows
             Preserve the original layout — line breaks, `Day 1:` / key: value
             prefixes, units, currency symbols, ids. Do not re-summarize, do not
             reorder, do not round numbers, do not translate.

             Copy ONLY what is present in the trace. Never fill a gap: if a slot
             was left unfilled, leave it unfilled or say so — an invented value
             is worse than a missing one, because it will be retrieved and used.
             Cap around 4000 characters; if the output is longer, keep the parts
             carrying concrete values and drop decorative prose, noting what was
             dropped.

             "" when the session produced no such artifact (pure investigation
             with no findings yet, or a stall waiting on the user).

             This field is NOT part of the retrieval vector — `statement` and
             `subject` do the finding, `details` supplies the value once found.
             So do not write retrieval bait here, and do not duplicate the
             summary; write the raw thing.

files      — 0..N absolute or repo-relative FILE OR DIRECTORY paths actually
             operated on or investigated during this session. Keep investigated
             roots such as `C:\\` and `D:\\`. Only include paths that were
             actually accessed/searched, not paths merely suggested or planned;
             [] if none apply.
═══════════════════════════════════════════════════════════════════
HOST CONTEXT (given by SDK)
═══════════════════════════════════════════════════════════════════
- workspace_id present: {workspace_present}
- branch present: {branch_present}

When no workspace_id is present, the solution cannot rely on unstated
project-local context — spell out any repo / path / service it needs inside
`solution` or `boundary_envs` so the memory stays usable on its own.

═══════════════════════════════════════════════════════════════════
CONVERSATION
═══════════════════════════════════════════════════════════════════
{conversation_block}

═══════════════════════════════════════════════════════════════════
OUTPUT (strict JSON only, no markdown)
═══════════════════════════════════════════════════════════════════
{{
  "experiences": [
    {{
      "task": "...",
      "search_keys": ["...", "..."],
      "solution": "...",
      "boundary_envs": "...",
      "confidence": 0.9
    }}
  ],
  "fact": {{
    "subject": "...",
    "statement": "...",
    "files": ["..."],
    "details": "..."
  }}
}}

DEFAULT TO EXTRACTING. When the segment did real work, you MUST output at least
one experience — even if the task looks simple or was a one-shot success, still
capture the reusable essence of what was done (the method / command / how-to).
When the session is thin, the single most valuable experience is enough.

The ONE exception is the stall case above: if the segment did no real work
because it was blocked asking the user for missing information, output
"experiences": [] and write only the fact. Do not manufacture an experience to
satisfy this instruction — a fabricated one ("how to ask for the departure
city") is worse than none, because it is later retrieved and followed. Empty is
allowed ONLY for that reason, never as a shortcut when work was actually done.

"fact" is a SINGLE object, never a list — one record per session.

ALWAYS write the fact. The bar for the fact is not "was this valuable" but "did
anything at all happen", and a segment reaching you has already cleared that.
This includes segments that look like setup rather than work: a single
user message stating requirements, parameters, or a starting state, with no
tools run, is exactly the material a later session cannot reconstruct on its own
— write the requirements, parameters, and starting state together with the situation into `statement` (compact notation, don't copy the dialogue verbatim),
and don't push them to `details` (`details` is not in the retrieval vector, and the read side no longer injects it). Use
"fact": null ONLY for greetings and pure chatter with no content whatsoever.
When in doubt, write the fact: an unneeded record costs one row, while a dropped
one costs every later session that needed it and has no way to know it existed.
"""


# ================================================================
# Prompt -- scene="coding" (software engineering scene)
# ================================================================
# Relation to EXTRACT_PROMPT(scene="working") above: same extractor, same placeholder set
# ({workspace_present}/{branch_present}/{conversation_block}), same output schema;
# only the spec differs. Both files coexist instead of one edited, because the two scenes impose different
# statement requirements, and sharing one file would let one scene's spec dilute the other's.
#
# The three blocks unique to this file (absent on the working side):
#   1. HARD CONSTRAINT header -- placed up front to override any later wording that "wants a narrative": the grammatical subject must be
#      CODE or DECISION, not "the user"; never close with "the session was completed";
#      keep the concrete symptom (error string / failing command / wrong output): the surface future searches match on.
#   2. The whole CODE WORK section -- four things a future agent cannot re-derive from code: the branch name (verbatim),
#      which version won (settled on A; interim B dropped), exact coordinates (repo-relative path + symbol,
#      not pasted code), and verified vs merely written. It asks not for length but for spending the same budget on
#      higher-value tokens.
#   3. The three-part statement -- OPEN with the symptom (how the user would report it) -> THEN coordinates + decision ->
#      CLOSE with the evidence state. Hard ceiling ~400 latin characters.
#
# What this file drops (present on the working side, deliberately absent here):
#   · The "ask the user" exclusion -- forcing experiences: [] when stalled asking the user for parameters. Coding-scene traces
#     come from autonomous agent execution, where no such stall exists.
#   · The TASK-SCOPED GIVENS per-line notation in statement PART 2 -- on the coding side the givens are
#     code coordinates, already covered by the CODE WORK section.
#   · The artifact-copy spec for details -- coding-side details are neither embedded nor injected.
#
# When editing the prompt, run regressions on representative coding traces to confirm coordinates, decisions, and verification states are still kept.

EXTRACT_PROMPT_CODING = """\
╔═══════════════════════════════════════════════════════════════════════════╗
║ HARD CONSTRAINT — read this before anything else, it overrides any         ║
║ phrasing later in this prompt that sounds like it wants a narrative.       ║
║                                                                           ║
║ You are NOT writing a report of what a person did. You are writing what a  ║
║ FUTURE AGENT needs in order to act. That agent already has its own task;   ║
║ it does not care what this user wanted. It needs to know: what was decided ║
║ here, at which code coordinates, what was tried and rejected, and whether  ║
║ it was verified.                                                          ║
║                                                                           ║
║ Concretely, in every `statement` and every `solution`:                     ║
║   · The grammatical subject is the CODE or THE DECISION — never "the user" ║
║     and never "the agent". Do not write "The user attempted to…",          ║
║     "The user wanted…", "The agent then checked…".                        ║
║   · Never close with a bookkeeping phrase like "the session was completed", ║
║     "the error was noted", "no solution was found yet". Close with the     ║
║     decision and its verification status. If nothing was decided, close    ║
║     with what was RULED OUT and the next unexplored lead — that is the     ║
║     reusable part.                                                        ║
║   · Keep the concrete symptom (error string, failing command, wrong        ║
║     output) — that is what a future search will match on.                 ║
║                                                                           ║
║ bad:  "The user attempted to write a QTable to RST with header_rows and    ║
║        hit a TypeError. The session was completed with the error noted."   ║
║ good: "ascii.rst's RST writer rejects header_rows: `tbl.write(..., \        ║
║        format='ascii.rst', header_rows=['name','unit'])` raises TypeError   ║
║        because RST.__init__ does not forward it, unlike ascii.fixed_width   ║
║        which accepts it. No fix written — the writer needs the same         ║
║        header_rows plumbing as fixed_width."                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

You extract durable WORKING memories from a task-execution session segment.
A "working" session is any hands-on task where the user operates a computer /
system through tools — this covers BOTH software/code work (writing, debugging,
refactoring, building, deploying, API/library usage) AND system & operations
work (running shell/PowerShell commands, disk/storage analysis or cleanup, file
& folder operations, installing/configuring software, environment/device setup,
hardware/disk health inspection, backup, scheduling/automation, data processing).

You produce TWO layers from the SAME segment:

  EXPERIENCES (procedural, "how to do it") — reusable method, extracted only
  when it clears the VALUE BAR below. A (task, search_keys, solution,
  boundary_envs) bundle.

  FACT (episodic, "what was established this time") — EXACTLY ONE record of THIS
  session. It is NOT a narrative of what the user did. It records, in this order:
  the task and its concrete symptom (this is what makes it findable later), the
  code coordinates, WHAT WAS SETTLED ON and what was rejected, and how far it was
  verified. No value bar, but there is never more than one.

The two layers answer different questions and are stored separately. A method
that generalizes goes in EXPERIENCES; the story of this particular session —
only true for this system / this project / right now — goes in the FACT.

═══════════════════════════════════════════════════════════════════
VALUE BAR (EXPERIENCES ONLY) — DO NOT cross this lightly.
═══════════════════════════════════════════════════════════════════
A memory MUST satisfy ALL of:
  (a) Solves a problem likely to recur for the same user
  (b) Solution is non-trivial: combines multiple pieces of info / steps, OR
      carries a decision rationale, OR captures a hard-won workaround, OR is a
      reusable procedure/recipe for an operational task (e.g. how to analyze &
      clean up disk usage, how to batch-organize files, how to inspect hardware
      health, how to set up a scheduled job)
  (c) A future agent / user would prefer looking it up over re-discovering it

IMPORTANT — analysis / diagnostic / investigation tasks COUNT:
A task does NOT have to make a change or "fix" something to be worth remembering.
Read-only / analytical / troubleshooting work is valuable when its PROCEDURE or
CONCLUSION is reusable. Explicitly extract:
  - Analysis / diagnostic recipes: the reusable procedure + what it revealed
    (e.g. "how to analyze which folders eat C-drive space" — the scan commands,
    the gotchas hit, and the finding). "It only produced a report, no change"
    is NOT a reason to skip — the reusable METHOD is the memory.
  - Investigation outcomes, INCLUDING negative / not-found results
    (e.g. "searched folder X across C: and D: with rg / fs_search / SQL index —
    not found anywhere"). A recorded dead-end saves the next agent from repeating
    the same fruitless search, so it clears the bar.
Do NOT reject a memory merely because "the assistant could regenerate the
commands" or "nothing was actually modified". If the procedure or the conclusion
would save future effort, keep it.

DO NOT extract:
  - Truly trivial single-step ops with no reusable procedure ("rename one file",
    "git status", "run ls once")
  - One-off issues already resolved with no transferable lesson
  - Information that is already in the repo/system (don't duplicate package.json)
  - Per-PR / per-session throwaway tasks unless they encode a reusable pattern

Heuristic: if the user (or a future agent doing the same kind of task) would
want this written down instead of figuring it out again, keep it. A multi-step
operational procedure (even a system/ops or read-only analysis/search one, not
just code) usually qualifies. Only drop it if it is a single command any user
would find via `--help` in 30 seconds AND carries no decision/gotcha worth
remembering.

═══════════════════════════════════════════════════════════════════
CODE WORK — what a future agent cannot re-derive, so it MUST be captured
═══════════════════════════════════════════════════════════════════
When the segment is software work on a repository, the reusable method is only
half of it. The other half is WHERE the work lives and WHICH VERSION won. Those
cannot be recovered by reading the code later — the code shows the current state,
not which branch it sits on, nor what was already rejected.

This section does NOT ask for a longer record. It asks you to SPEND THE SAME
BUDGET on higher-value tokens: drop restated goals, narration of what the
assistant "then decided to", and pasted code you are only echoing; spend those
characters on the four items below instead. A record that omits the branch but
quotes 30 lines of diff is a bad trade.

1. BRANCH / WORKING LOCATION — verbatim, in the fact `statement`. If the segment
   mentions or checks out `feat/x` / `fix/y`, that token MUST appear. An agent
   that resumes on the wrong branch produces changes that are silently thrown
   away, so this one token outranks every other detail.

2. WHICH VERSION WON — when the user changed their mind, name the winner and mark
   the loser in one clause: "settled on A; the interim B was dropped". Not both in
   full — one clause. Keeping only the latest is wrong: an agent that meets the
   rejected code must recognize it, or it will "fix" things back to the discarded
   variant. Phrases like `scratch that / go back to / actually, let's / no, not that`
   mark this: what follows is of record, what precedes is superseded.

3. EXACT COORDINATES, NOT PASTED CODE — repo-relative path
   (`qutebrowser/utils/debug.py`, never bare `debug.py`) plus the function /
   class touched. If a signature changed, give the new signature. If a constant,
   regex or literal was decided, quote THAT verbatim — but quote only it, not the
   surrounding block. One accurate line beats a paragraph of approximation.

4. VERIFIED vs MERELY WRITTEN — four words is enough ("tests run and passing" /
   "confirmed by git diff only, tests not run"). "the edit is in" and "the edit is in
   and tested" lead a future agent to do very different things next.

Anti-pattern — summarizing the goal and dropping the coordinates:
  bad:  "refactored the colour parsing logic to be cleaner."
  good: "on feat/q4, in qutebrowser/config/configtypes.py QtColor._parse_value:
         settled on `mult = 360.0 if kind == 'h' else 255.0`; the interim version that
         used 255.0 for everything was dropped. Tests not run."
  (the good one is SHORTER than most bad ones — it just spends every character
   on something that cannot be re-derived)

═══════════════════════════════════════════════════════════════════
EXPERIENCE FIELDS
═══════════════════════════════════════════════════════════════════

task          — Query-shaped name; phrase as how a user would later ASK.
                "publish a release to the xx platform", not "publish_release_internal"
                Self-contained (no "this", "the file", "we").

search_keys   — 0 to 5 alternative phrasings capturing important aspects of
                solution that `task` alone wouldn't match. Phrase as questions
                or topics future users would ASK ABOUT. Distinct from task.
                Omit if task already covers everything.

solution      — Complete, self-contained content the user actually needs:
                commands, credential locations (NOT credential VALUES),
                file paths, ordered steps, gotchas, decision rationale.
                Aggregate multi-turn end-to-end work into ONE solution if it
                forms a coherent task; split into multiple memories only when
                the user did unrelated tasks.

boundary_envs — Concrete pinning that solution depends on, multi-line "key: value":
                  runtime / SDK / library + version
                  API endpoint or method signature
                  external service / middleware version
                  env var NAME (not value)
                  config file path
                  host / cluster / region identifier
                If solution is environment-agnostic, leave as "" (empty).
                Purpose: when the underlying SDK/middleware upgrades, future
                consumers can inspect this to decide if solution still applies.

confidence  — 0.0 to 1.0
  1.0   user explicitly stated this as a rule / instruction
  0.7   inferred with good evidence from tool actions
  0.5   weakly inferred; one observation; uncertain

═══════════════════════════════════════════════════════════════════
FACT FIELD — EXACTLY ONE per segment, no value bar
═══════════════════════════════════════════════════════════════════

You write ONE fact for the whole segment — a recoverable state record for this
session, not a polished retrospective and never a list of separate findings. A
future agent must be able to resume from it without mistaking a suggestion,
inference, or partial search for something verified.

statement  — A compact summary of THIS session. It has two jobs at once, and it
             must do BOTH: it is the retrieval vector (a future agent finds it by
             searching with a fresh issue report, phrased the way a user phrases a
             problem), and it is the payload that agent then acts on. So:

             OPEN with the task and the symptom, in the words a user would use.
                This is what makes it findable — keep the surface form close to how
                the problem would be reported. Name the specific goal and object
                ("reclaim disk space under D:\\WXWork", "adjust the hue multiplier
                in QtColor._parse_value on feat/q4", "locate the new_file_search
                folder"), never a generic "fix a bug" / "look for a file". Include
                the concrete symptom: the error string, the failing command, the
                wrong output.

             THEN give the coordinates and the decision.
                repo-relative path, symbol, and branch for code work. State what was
                SETTLED ON, and in the same clause mark what was dropped
                ("settled on X; the interim Y was dropped"). This is the part that
                cannot be re-derived by reading the code — the code only shows the
                winner, never that Y was tried and rejected. An agent that meets the
                rejected variant later must be able to recognize it, or it will
                "fix" the code back to the discarded version.

             CLOSE with the evidence status.
                Exactly one of: completed, partially completed, failed, blocked, or
                still awaiting verification — plus how it was checked ("tests run and
                passing" / "confirmed by git diff only, tests not run"). Include
                negative results and dead ends without overstating them.

             ⚠️ Never stop at the narrative. "The user tried X and hit error Y; no
             solution found yet" is a wasted record: it retrieves fine but gives a
             future agent nothing to act on. If the session genuinely reached no
             decision, say what was ruled out and what the next unexplored lead is —
             that is the reusable part.

             The subject of the sentences should be the code and the decision, not
             "the user". 2 to 4 sentences, flowing prose, not labelled sections.
             Hard ceiling ~400 latin characters (~200 CJK characters).

             EVIDENCE FIDELITY — mandatory:
             - Record only actions actually executed and evidence explicitly
               present in the trace. Preserve material commands, flags,
               arguments, errors, and numbers verbatim. NEVER complete a
               command with customary flags or turn advice/plans into actions.
             - Preserve the distinction among attempted, observed, verified,
               inferred, and unresolved. Do not add those labels to the prose;
               express their meaning naturally, e.g. "tried … but was blocked by
               Access denied" or "suspect … , not yet verified".
             - Without tool output or explicit confirmation, NEVER claim a fix
               succeeded, a cause was established, a file/folder does not
               exist, or tests passed.
             - Scope every negative finding to what was actually searched and
               enumerate EVERY explicit coverage limit that could invalidate
               it, especially permission failures and maximum search depth:
               "not found within the searched parts of C:\\ and D:\\; some
               directories were skipped on Access denied, and the search was also
               depth-limited". Do not omit one limit to make the
               summary prettier. Never promote a partial search to global
               absence.
             - State causality as certain only when the trace directly verifies
               it. Otherwise retain the hedge ("suspected" / "inferred" / "not yet
               verified"); do not rewrite an
               incorrect test assumption as the system's root cause.

             CRITICAL — this text IS the retrieval vector, and it is the only
             thing a future query gets matched against. A generic summary is
             worthless, because EVERY session looks like a generic summary: it
             would match every query and answer none of them. So you MUST name
             the concrete anchors that make THIS session findable — real file
             paths, function / symbol names, branch names, commands, error
             strings, versions, config keys and values, host / service names.
             If someone later searches for any one of those specifics, this
             summary has to be the thing that comes back.
               good: "fixing a failing `npm run build` in CI. Read
                      .github/workflows/ci.yml and package.json; traced it to webpack
                      5.74 hitting ERR_OSSL_EVP_UNSUPPORTED under node 18. Bumping the
                      workflow's node to 20 made the build pass; node 18 still
                      reproduces locally."
               good: "on branch feat/q0, in qutebrowser/utils/debug.py signal_name:
                      PyQt<5.11 only exposes the name via repr(sig), and repr comes in
                      both a bracketed and a parenthesised form. Settled on two patterns
                      tried in order with fullmatch, breaking on the first hit; the
                      single-regex version was dropped. Confirmed by git diff, tests not
                      run."
               bad:  "user wanted a build bug fixed, the agent looked at some config
                      files, changed some code, and fixed it."
                      (no path, no command, no error string, no version — nothing
                       to retrieve on; every session reads exactly like this)
               bad:  "fixed the signal-name parsing compatibility issue on a feature
                      branch."
                      (which branch? which file? which function? which of the two
                       competing versions won? nothing here survives being
                       retrieved a week later)
             Fidelity examples (bad → good):
               "ran Get-ChildItem -Force" → "ran Get-ChildItem"
                 (the trace did not contain `-Force`; do not invent it)
               "confirmed new_file_search does not exist" →
                 "new_file_search was not found within the searched parts of C:\\ and
                  D:\\; some directories were skipped on Access denied or by the depth
                  limit, so the investigation is only partly done"
               "identical payloads cause a spurious lock error" →
                 "the verification script assumed the last writer is always a fixed
                  rank; that assumption does not hold, so whether there is a lock bug
                  is still unverified"

             Do NOT record greetings or pure chatter. If the segment established
             nothing at all, output "fact": null.

files      — 0..N absolute or repo-relative FILE OR DIRECTORY paths actually
             operated on or investigated during this session. Keep investigated
             roots such as `C:\\` and `D:\\`. Only include paths that were
             actually accessed/searched, not paths merely suggested or planned;
             [] if none apply.

═══════════════════════════════════════════════════════════════════
HOST CONTEXT (given by SDK)
═══════════════════════════════════════════════════════════════════
- workspace_id present: {workspace_present}
- branch present: {branch_present}

When no workspace_id is present, the solution cannot rely on unstated
project-local context — spell out any repo / path / service it needs inside
`solution` or `boundary_envs` so the memory stays usable on its own.

═══════════════════════════════════════════════════════════════════
CONVERSATION
═══════════════════════════════════════════════════════════════════
{conversation_block}

═══════════════════════════════════════════════════════════════════
OUTPUT (strict JSON only, no markdown)
═══════════════════════════════════════════════════════════════════
{{
  "experiences": [
    {{
      "task": "...",
      "search_keys": ["...", "..."],
      "solution": "...",
      "boundary_envs": "...",
      "confidence": 0.9
    }}
  ],
  "fact": {{
    "statement": "...",
    "files": ["..."]
  }}
}}

MANDATORY: You MUST output at least ONE experience. Do NOT return an empty list.
Even if the task looks simple or is a one-shot success, still capture the
reusable essence of what was done (the method / command / how-to). Only the
single most valuable experience is required when the session is thin — but never
output an empty "experiences" array.

"fact" is a SINGLE object, never a list — one record per session. Use
"fact": null only when the segment is pure chatter that established nothing.
"""


# ================================================================
# CodingMemoryExtractor
# ================================================================

class CodingMemoryExtractor:
    """
    LLM-driven two-layer extractor.

    Input: the full New Messages for this call + the most recent K user turns from history; the LLM view of both parts is length-limited
    Output: (List[CodingMemoryDraft], List[WorkingFactDraft])

    The fact layer yields at most one record per trace segment, so the returned fact list length is always 0 or 1; a list rather than
    Optional keeps writer / reconciler signatures unchanged.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_tokens: int = 6000,
        temperature: Optional[float] = None,
        cache: Any = None,
        scene: str = "working",
        verbatim_fallback_enabled: bool = False,
    ):
        # scene decides which extraction spec to use. Direct construction defaults to working; the client's
        # scene=None auto-routing runs the three-way judge first, then builds the matching variant.
        #   "working" → EXTRACT_PROMPT        (office / general productivity)
        #   "coding"  → EXTRACT_PROMPT_CODING (software engineering)
        # Invalid values don't raise; they fall back to working: extraction is one-shot, and letting the whole
        # write fail over one misspelled argument is not worth it.
        self.scene = "coding" if str(scene or "").strip().lower() == "coding" else "working"
        self._prompt_tmpl = (
            EXTRACT_PROMPT_CODING if self.scene == "coding" else EXTRACT_PROMPT
        )
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        # Off by default: the verbatim fallback may write assistant/tool traces straight into the statement,
        # which search consumers would then display as the memory body. CodingConfig opts in explicitly when the old behavior is needed.
        self.verbatim_fallback_enabled = bool(verbatim_fallback_enabled)
        # When cache is present, persist a CODING_EXTRACT pipeline log (best-effort)
        self.cache = cache
        # None → use LLMConfig.temperature (avoids hardcoded values tripping kimi/deepseek temperature constraints)
        if temperature is None:
            # Passing None through to the provider omits the parameter; the previous 0.1 fallback bypassed user config and
            # returned a hard 400 on gateways that only accept specific temperatures (Opus 5 accepts only 1 or omission)
            temperature = _resolve_configured_temperature(llm_provider)
        self.temperature = temperature

    @staticmethod
    def _language_text(messages: List[ChatMessage]) -> str:
        """Use new user text for routing; tool output must not dominate it."""
        user_parts = [
            str(message.content or "").strip()
            for message in messages
            if message.role == "user" and str(message.content or "").strip()
        ]
        if user_parts:
            return "\n".join(user_parts)

        non_tool_parts = [
            str(message.content or "").strip()
            for message in messages
            if message.role != "tool" and str(message.content or "").strip()
        ]
        if non_tool_parts:
            return "\n".join(non_tool_parts)
        return "\n".join(
            str(message.content or "").strip()
            for message in messages
            if str(message.content or "").strip()
        )

    def _select_prompt_template(self, messages: List[ChatMessage]) -> Tuple[str, str]:
        """Return (template, language) for this write's new messages."""
        if not is_chinese(self._language_text(messages)):
            return self._prompt_tmpl, "en"

        from .prompts_zh import EXTRACT_PROMPT_CODING_ZH, EXTRACT_PROMPT_ZH

        prompt = EXTRACT_PROMPT_CODING_ZH if self.scene == "coding" else EXTRACT_PROMPT_ZH
        return prompt, "zh"

    async def extract(
        self,
        messages: List[ChatMessage],
        *,
        history_messages: Optional[List[ChatMessage]] = None,
        tool_result_max_bytes: int = 2048,
        user_id: str,
        agent_id: str = "default_agent",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: str = "",
    ) -> Tuple[List[CodingMemoryDraft], List[WorkingFactDraft]]:
        """
        Args:
            messages: the complete new messages of this write; per-message length limits only, no turn trimming
            history_messages: history from previous writes; only the last 5 user turns as coreference context
            user_id / agent_id / workspace_id / branch / session_id: injected by the SDK

        Returns:
            (experience drafts, fact drafts). Either layer may be empty. Only when
            verbatim_fallback_enabled is explicitly on does _verbatim_fact run for a missing fact.
        """
        if not messages:
            return [], []

        # New Messages is the extraction subject of this write: whether 1 user-turn or 10 is passed in,
        # it must enter the prompt complete; this only caps the size of each tool / assistant message.
        new_messages = prepare_extractor_messages(
            messages, tool_max_bytes=tool_result_max_bytes,
        )

        # Last 5 User Turns is history from previous writes, for coreference resolution and context only. Never use
        # last-K to trim the current input, or write(10 turns) would silently drop the first 5 turns.
        history_window = keep_last_k_user_turns(
            prepare_extractor_messages(
                history_messages or [], tool_max_bytes=tool_result_max_bytes,
            ),
            k=5,
        )

        history_block = self._format_conversation(history_window).strip()
        new_block = self._format_conversation(new_messages).strip()
        prompt_tmpl, prompt_language = self._select_prompt_template(new_messages)
        if prompt_language == "zh":
            conversation_block = (
                "LAST 5 USER TURNS (context only; do not extract them again):\n"
                + (history_block or "(none)")
                + "\n\nNEW MESSAGES THIS TIME (extract only from this complete write):\n"
                + (new_block or "(none)")
            )
        else:
            conversation_block = (
                "LAST 5 USER TURNS (CONTEXT ONLY; do not extract them again):\n"
                + (history_block or "(none)")
                + "\n\nNEW MESSAGES (extract only from this complete write):\n"
                + (new_block or "(none)")
            )

        prompt = prompt_tmpl.format(
            workspace_present=str(bool(workspace_id)).lower(),
            branch_present=str(bool(branch)).lower(),
            conversation_block=conversation_block,
        )

        _t0 = time.perf_counter()
        try:
            resp = await self.llm_provider.complete(
                prompt=prompt,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.warning(f"[coding-extract] LLM call failed: {e}")
            # No fallback for the experience layer. The fact-layer verbatim fallback is off by default too, to stop assistant/tool traces from being
            # shown directly as the statement; the old behavior survives only when config explicitly enables it.
            verbatim = (
                self._verbatim_fact(messages)
                if self.verbatim_fallback_enabled else None
            )
            fact_source = (
                "verbatim_fallback" if verbatim
                else "verbatim_fallback_disabled"
                if not self.verbatim_fallback_enabled
                else "none"
            )
            await _store_coding_log(
                self.cache, request_id=request_id, user_id=user_id, agent_id=agent_id,
                session_id=session_id or "", step="CODING_EXTRACT", prompt=prompt,
                response=f"[LLM error] {type(e).__name__}: {str(e)[:300]}",
                parsed=json.dumps(
                    {
                        "drafts": 0,
                        "llm_error": True,
                        "facts": [verbatim.to_dict()] if verbatim else [],
                        "fact_count": 1 if verbatim else 0,
                        "fact_source": fact_source,
                    },
                    ensure_ascii=False,
                ),
                elapsed_ms=round((time.perf_counter() - _t0) * 1000, 2),
            )
            return [], ([verbatim] if verbatim else [])
        _elapsed = round((time.perf_counter() - _t0) * 1000, 2)

        raw_experiences, raw_facts = self._parse_output(resp.content)

        # Experience files use rule extraction (files touched across the whole trace); fact files come from the
        # LLM, which picks what was truly touched/inspected this time, matching the session summary better than the rule-extracted superset
        files = extract_files(messages)

        drafts: List[CodingMemoryDraft] = []
        for raw in raw_experiences:
            d = self._normalize_draft(
                raw,
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                branch=branch,
                session_id=session_id,
                files=files,
            )
            if d is not None:
                drafts.append(d)

        fact_drafts: List[WorkingFactDraft] = []
        for raw in raw_facts:
            f = self._normalize_fact_draft(raw)
            if f is not None:
                fact_drafts.append(f)

        # For details, don't trust the LLM's transcription; take it directly from the uncompressed messages.
        #
        # The conversation in the extraction prompt is windowed and truncated (see preproc); the middle of a long assistant
        # text is already gone when the prompt is built. Asking the model to "copy verbatim" a source it cannot see
        # can only produce a head and tail with ellipsis markers.
        #
        # details exists to keep the addressable artifacts a summary drops (plan tables, config blocks). It is not in the
        # retrieval vector (see store._upsert_fact_vec) and its length does not affect recall, so there is no reason to route it through
        # another lossy channel -- take the raw text, bounded only by details' own limit.
        #
        # Note: bypassing one lossy stage does not fix it. The input-side truncation thresholds affect statement too;
        # see preproc.DEFAULT_ASSISTANT_HEAD_CHARS.
        #
        # The LLM's details stay as a fallback: when messages are empty it is the only source.
        verbatim_details = self._verbatim_details(messages)
        if verbatim_details:
            fact_drafts = [
                WorkingFactDraft(
                    statement=f.statement,
                    files=f.files,
                    subject=f.subject,
                    details=verbatim_details,
                )
                for f in fact_drafts
            ]

        fact_source = "llm"
        if not fact_drafts:
            if self.verbatim_fallback_enabled:
                verbatim = self._verbatim_fact(messages)
                if verbatim is not None:
                    fact_drafts.append(verbatim)
                    fact_source = "verbatim_fallback"
                else:
                    fact_source = "none"
            else:
                fact_source = "verbatim_fallback_disabled"

        logger.info(
            f"[coding-extract] produced {len(drafts)} experience draft(s), "
            f"{'1 session fact' if fact_drafts else 'no session fact'}"
            f"{' (verbatim fallback: LLM returned none)' if fact_source == 'verbatim_fallback' else ''}"
            f"{' (verbatim fallback disabled)' if fact_source == 'verbatim_fallback_disabled' else ''}"
            f" (workspace={workspace_id!r}, branch={branch!r})"
        )
        # CODING_EXTRACT: log one record no matter how many drafts were produced (including 0), for debugging "nothing extracted".
        # With the two-layer architecture, parsed records both experience and fact drafts, so each layer's output stays traceable.
        await _store_coding_log(
            self.cache, request_id=request_id, user_id=user_id, agent_id=agent_id,
            session_id=session_id or "", step="CODING_EXTRACT", prompt=prompt,
            response=resp.content or "",
            parsed=json.dumps(
                {
                    "experiences": [d.to_dict() for d in drafts],
                    "experience_count": len(drafts),
                    "facts": [f.to_dict() for f in fact_drafts],
                    "fact_count": len(fact_drafts),
                    "fact_source": fact_source,
                },
                ensure_ascii=False, default=str,
            ),
            elapsed_ms=_elapsed, prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens, total_tokens=resp.tokens_used,
        )
        return drafts, fact_drafts

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    @classmethod
    def _verbatim_details(cls, messages: List[ChatMessage]) -> str:
        """The raw text of the whole trace, the authoritative source for details.

        It reads messages un-compressed by keep_last_k_user_turns: details exists precisely to
        keep the addressable artifacts a summary loses (the "day N, meal M" cells), and compression is the step that loses them.
        It is not embedded into the retrieval vector (store._upsert_fact_vec embeds only subject + statement), so its
        length affects storage only, never recall quality; the sole cap is _MAX_DETAILS_CHARS.

        Truncate from the tail: the head starts the structured artifact (Day 1, first config line) and carries the highest density.
        """
        if not messages:
            return ""
        body = cls._format_conversation(messages).strip()
        if not body:
            return ""
        if len(body) <= _MAX_DETAILS_CHARS:
            return body
        return body[:_MAX_DETAILS_CHARS] + "\n…(truncated)"

    @classmethod
    def _verbatim_fact(
        cls, messages: List[ChatMessage]
    ) -> Optional[WorkingFactDraft]:
        """Fallback when the LLM produced no fact: store the raw trace as one fact.

        It returns None only when there is literally not a single character (empty messages, all-empty content). This is the last gate;
        the test must be "is there any content", never "is the content good enough" -- goodness was already judged by the LLM call
        above, which misjudged once; making the fallback echo the same judgment adds nothing.

        statement feeds retrieval, so it takes the user's own words (they are the phrasing future queries will use;
        assistant restatements and tool output are not); details holds the full raw trace for reading after a hit.
        subject stays empty -- identifying the subject is a semantic judgment rules cannot make; an empty string sends the reconciler back to
        the pre-change pure-semantic overwrite decision, rather than gating on a misguessed anchor.
        """
        if not messages:
            return None

        user_texts: List[str] = []
        for m in messages:
            if m.role == "user" and not m.is_tool_message():
                text = (m.content or "").strip()
                if text:
                    user_texts.append(text)

        details = cls._verbatim_details(messages)
        if not details:
            return None

        # Prefer the user's own words; with not one sentence (pure tool replay), fall back to the start of the trace
        head = "\n".join(user_texts).strip() or details
        statement = head[:_VERBATIM_STATEMENT_CHARS]
        if len(head) > _VERBATIM_STATEMENT_CHARS:
            statement += "…"

        return WorkingFactDraft(statement=statement, files=[], subject="", details=details)

    @staticmethod
    def _format_conversation(messages: List[ChatMessage]) -> str:
        """Render the whole conversation for the LLM to read."""
        rows: List[str] = []
        for i, m in enumerate(messages):
            if m.role == "user" and not m.is_tool_message():
                rows.append(f"[{i}] user: {m.content}")
            elif m.role == "assistant":
                line = f"[{i}] assistant: {m.content}"
                if m.tool_calls:
                    tc_summary = "; ".join(
                        f"{tc.name}({_compact_args(tc.arguments)})"
                        for tc in m.tool_calls
                    )
                    line += f"  [tools: {tc_summary}]"
                rows.append(line)
            elif m.is_tool_message():
                tn = m.tool_name or "?"
                rows.append(f"[{i}] tool({tn}): {m.content}")
            elif m.role == "system":
                rows.append(f"[{i}] system: {m.content}")
            else:
                rows.append(f"[{i}] {m.role}: {m.content}")
        return "\n".join(rows)

    @classmethod
    def _parse_output(
        cls, text: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Leniently parse the LLM output JSON into the raw (experiences, facts) lists."""
        if not text:
            return [], []
        s = text.strip()
        # Strip the markdown fence
        if s.startswith("```"):
            lines = s.split("\n")
            body, in_fence = [], False
            for line in lines:
                if line.startswith("```"):
                    in_fence = not in_fence
                    continue
                body.append(line)
            s = "\n".join(body).strip() or s

        obj = None
        # Try plain JSON first
        try:
            obj = json.loads(s)
        except Exception:
            # Find the first {...} block
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end > start:
                try:
                    obj = json.loads(s[start: end + 1])
                except Exception:
                    obj = None

        if not isinstance(obj, dict):
            logger.warning(f"[coding-extract] failed to parse output: {text!r}")
            return [], []

        # "memories" is the pre-two-layer key; cached old prompts / old models may still return it
        experiences = cls._as_dict_list(obj.get("experiences"))
        if not experiences:
            experiences = cls._as_dict_list(obj.get("memories"))
        return experiences, cls._as_fact_list(obj)

    @classmethod
    def _as_fact_list(cls, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract the segment's single fact, returning a list of length 0 or 1.

        The current contract is one "fact" object. The "facts" array is the shape from the multi-fact-per-segment version;
        old models and the old prompts cached in them still reply that way, so it stays accepted, but only the first entry is kept -- the extras
        should never have existed, and dropping them is truer to the intent than letting each become its own session record.
        """
        fact = obj.get("fact")
        if isinstance(fact, dict):
            return [fact]

        legacy = cls._as_dict_list(obj.get("facts"))
        if len(legacy) > 1:
            logger.warning(
                f"[coding-extract] legacy 'facts' array with {len(legacy)} entries; "
                f"keeping the first, dropping {len(legacy) - 1}"
            )
        return legacy[:1]

    @staticmethod
    def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [v for v in value if isinstance(v, dict)]

    @staticmethod
    def _normalize_draft(
        raw: Dict[str, Any],
        *,
        user_id: str,
        agent_id: str,
        workspace_id: Optional[str],
        branch: Optional[str],
        session_id: Optional[str],
        files: List[str],
    ) -> Optional[CodingMemoryDraft]:
        """
        Normalize one memory dict from the LLM output into a CodingMemoryDraft.

        Returning None marks the entry invalid (missing task or solution).
        """
        task = (raw.get("task") or "").strip()
        solution = (raw.get("solution") or "").strip()
        if not task or not solution:
            logger.debug(f"[coding-extract] drop draft missing task/solution: {raw}")
            return None

        # Normalize search_keys
        sk_raw = raw.get("search_keys") or []
        if not isinstance(sk_raw, list):
            sk_raw = []
        search_keys = [str(s).strip() for s in sk_raw if str(s).strip()]
        # Deduplicate + drop keys identical to task
        seen = set()
        deduped = []
        for k in search_keys:
            if k == task:
                continue
            if k in seen:
                continue
            seen.add(k)
            deduped.append(k)
        search_keys = deduped[:5]  # Soft cap of 5

        # Normalize confidence
        try:
            confidence = float(raw.get("confidence", 0.7))
        except Exception:
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        boundary_envs = (raw.get("boundary_envs") or "").strip()

        return CodingMemoryDraft(
            task=task,
            search_keys=search_keys,
            solution=solution,
            boundary_envs=boundary_envs,
            confidence=confidence,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            branch=branch,
            session_id=session_id,
            files=list(files),
            extraction_source="auto_extract",
        )

    @staticmethod
    def _normalize_fact_draft(raw: Dict[str, Any]) -> Optional[WorkingFactDraft]:
        """Normalize one fact from the LLM output. Returns None when statement is empty."""
        statement = (raw.get("statement") or "").strip()
        if not statement:
            logger.debug(f"[coding-extract] drop fact missing statement: {raw}")
            return None

        files_raw = raw.get("files") or []
        if not isinstance(files_raw, list):
            files_raw = []
        seen, files = set(), []
        for f in files_raw:
            f = str(f).strip()
            if f and f not in seen:
                seen.add(f)
                files.append(f)

        subject = " ".join(str(raw.get("subject") or "").split())
        # subject is a hard key, not prose; a whole sentence means the segment went off track, so prefer treating it as "no subject"
        # (fall back to pure-semantic overwrite decisions) over comparing a sentence for equality.
        if len(subject) > _MAX_SUBJECT_CHARS:
            logger.debug(
                f"[coding-extract] subject too long, treated as absent: {subject!r}"
            )
            subject = ""

        details = str(raw.get("details") or "").strip()
        if len(details) > _MAX_DETAILS_CHARS:
            # Truncate from the tail; the head starts the structured artifact (Day 1, first config line), highest density
            details = details[:_MAX_DETAILS_CHARS] + "\n…(truncated)"

        return WorkingFactDraft(
            statement=statement, files=files, subject=subject, details=details
        )


# ================================================================
# Helpers
# ================================================================

def _compact_args(args: Dict[str, Any], max_chars: int = 600) -> str:
    """Render tool args on one line while keeping an evidence window wide enough for key command arguments.

    A cap remains so whole scripts / big payloads cannot blow up the prompt; but 80 characters would truncate
    common PowerShell commands before parameters like ``-Depth`` / ``-ErrorAction``, breaking the fact's evidence-fidelity contract.
    """
    if not args:
        return ""
    try:
        s = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(args)
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s
