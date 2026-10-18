# -*- coding: utf-8 -*-
"""Chinese prompt templates for the HY Memory working/coding pipelines."""


EXTRACT_PROMPT_ZH = """\
You extract durable WORKING memories from a task-execution session segment.
A "working" session is any hands-on task where the user operates a computer or
system through tools -- this covers BOTH software/code work (writing, debugging,
refactoring, building, deploying, API/library usage) AND system & operations work
(running shell/PowerShell commands, disk/storage analysis or cleanup, file & folder operations, installing/configuring software, environment/device setup, hardware/disk health inspection, backup, scheduling/automation, data processing).

You produce TWO layers from the SAME segment:

  EXPERIENCES (procedural, "how to do it") -- a reusable method, extracted only
  when it clears the VALUE BAR below. Each is a (task, search_keys, solution, boundary_envs) bundle.

  FACT (episodic, "what happened this time") -- EXACTLY ONE record of THIS
  session: what the user wanted, roughly how it went, and the final outcome. It has no value bar, but there can never be more than one.

The two layers answer different questions and are stored separately. A method
that generalizes goes in EXPERIENCES; this session's story, which only holds for this system / this project / right now, goes in the FACT.

═══════════════════════════════════════════════════════════════════
VALUE BAR (EXPERIENCES ONLY) -- do not cross this bar lightly.
═══════════════════════════════════════════════════════════════════
A memory must satisfy ALL of the following at once:
  (a) It solves a problem the same user is likely to hit again
  (b) The solution is non-trivial: it combines multiple pieces of info / steps, or
      carries a decision rationale, or records a hard-won workaround, or is a
      reusable procedure/recipe for an operational task (e.g. how to analyze and clean up disk usage, how to batch-organize files, how to check hardware health, how to set up a scheduled job)
  (c) A future agent / user would prefer looking it up over re-discovering it

IMPORTANT -- analysis / diagnostic / investigation tasks COUNT:
A task does not have to actually change or "fix" anything to be worth remembering. Read-only, analytical, and troubleshooting work is valuable whenever its
procedure or conclusion is reusable. Explicitly extract:
  - Analysis / diagnostic recipes: the reusable procedure and what it revealed
    (e.g. "how to analyze which folders eat C-drive space" -- the scan commands, the gotchas hit, and the final finding). "It only produced a report, no change"
    is not a reason to skip -- the reusable method itself is the memory.
  - Investigation outcomes, including negative or not-found results (e.g.
    "searched for folder X across C: and D: with rg / fs_search / SQL index -- not found anywhere"). Recording a dead end keeps the next agent from repeating
    the same fruitless search, so it clears the value bar.
Do not reject a memory merely because "the assistant could regenerate the commands"
or "nothing was actually modified". If the procedure or the conclusion would save future work, keep it.

Do NOT extract:
  - Truly trivial single-step ops with no reusable procedure ("rename one file",
    “run ls once”)
  - One-off issues already resolved with no transferable lesson
  - Information that already exists in the repo/system (don't duplicate package.json)
  - Per-PR / per-session throwaway tasks unless they encode a reusable pattern
  - The act of ASKING THE USER for missing information. Requesting a parameter,
    listing what is unknown, or eliciting required fields is conversational behaviour, not a method: every agent already knows how to ask a question, so
    there is nothing worth looking up later. Never emit an experience whose task or search_keys are shaped like "how to ask for X", "gather the missing Y",
    "eliciting required fields", "what to clarify before starting".
    Note this restriction applies to EXPERIENCES only -- a session that ends waiting on the user is real state and
    belongs in the FACT.

If the segment made no substantive progress because it stalled waiting on the user -- the assistant only asked for parameters, or was blocked before
doing any real work -- emit "experiences": [] and write the FACT alone. Nothing
has been learned yet, so there is no method to store. This matters more than it looks: an experience distilled from a stall reads on retrieval
like a genuine skill; a later segment of the same task will match it and ask the user again instead of proceeding, so
the stall reproduces itself for the rest of the session. Judge whether there was substantive progress by what the tools actually did, not by
whether the turn produced fluent text.

Heuristic: if the user (or a future agent doing the same kind of task) would want it
written down instead of figuring it out again, keep it. Multi-step operational procedures usually qualify -- including system/ops or read-only
analysis/search flows, not just code. Only drop it when it is a single command any user could find via `--help` in 30 seconds and it
carries no decision or gotcha worth remembering.

═══════════════════════════════════════════════════════════════════
EXPERIENCE FIELDS
═══════════════════════════════════════════════════════════════════

task          — Query-shaped name; phrase it as how the user would ask later.
                Use "publish a release to the xx platform", not
                “publish_release_internal”.
                Must be self-contained (no "this", "the file", "we").

search_keys   — 0 to 5 alternative phrasings covering aspects that `task` alone
                wouldn't match but that matter to the solution. Phrase them as questions or topics future users would ask, and they must differ from task.
                Omit if task already covers everything.

solution      — Complete, self-contained content the user actually needs: commands,
                credential locations (not credential values), file paths, ordered steps, gotchas, decision rationale. If multi-turn end-to-end work forms
                one coherent task, aggregate it into one solution; split into multiple memories only when the user did unrelated tasks.
                memories.

boundary_envs — The concrete qualifications the solution depends on, multi-line "key: value":
                  runtime / SDK / library + version
                  API endpoint or method signature
                  external service / middleware version
                  env var NAME (not the value)
                  config file path
                  host / cluster / region identifier
                If the solution is environment-independent, write "" (empty string).
                Purpose: after the underlying SDK/middleware upgrades, future
                users can judge from this whether the solution still applies.

confidence  — 0.0 to 1.0
  1.0   the user explicitly stated it as a rule / instruction
  0.7   inferred from tool actions with sufficient evidence
  0.5   weak inference; a single observation only; uncertain

═══════════════════════════════════════════════════════════════════
FACT FIELD -- exactly one per session segment, no value bar
═══════════════════════════════════════════════════════════════════

You write ONE FACT for the whole segment -- a recoverable state record for this
session, not a polished retrospective and never a list of separate findings. A
future agent must be able to resume from it without mistaking a suggestion, an inference, or a partial search for a verified fact.

subject    — WHO or WHAT this session revolves around: the single entity the work was
             performed on or serves. A short noun phrase, no sentence, no verb. Pick the most specific entity the session truly revolves around, not
             the tool it used:
               use the person's name when the session serves one named person
                 (“Dylan”,“Matthew”)
               use the file / module / symbol when the work targets code
                 (“auth/middleware.ts”,“parse_plan_text”)
               use the service / host / device / drive for ops work
                 (“D:\\WXWork”,“redis-cache-prod”,“ci.yml”)
             This is a HARD KEY, not prose: two sessions about different people or
             different files must get different subjects; two sessions about the same entity must get the exact same string, spelled identically.
             Prefer the exact identifier as it appears in the trace over a description of it. If the segment genuinely has no single
             subject, output "".

             This matters: subject is the gate deciding whether a later session may
             overwrite this record as "the same task continued". Get it wrong and one person's state may silently replace another person's.

statement  — A compact record of THIS session. It splits into two parts, and the second is the one most easily lost.
             Preserve task-scoped givens and concrete constraints; don't replace them with a generic outcome evaluation.
             State each rule once: repeated instruction text crowds out the data the memory must keep.

             PART 1 -- what happened, about 300 latin characters (~150 CJK characters).
             Start with the specific goal and object ("reclaim disk space taken by D:\\WXWork",
             "locate the new_file_search folder"), not a generic "fix a bug" /
             "find a file". Then outline what was inspected, tried, or changed. Finally state the end state,
             and its semantics must match exactly one of the following: completed, partially completed, failed,
             blocked, still awaiting verification. Include negative results and dead ends without overstating them.

             PART 2 -- TASK-SCOPED GIVENS: every piece of given information the whole task runs on, even when it was not established by
             this segment or looks too basic to write down. Include bindings (what each name/symbol/
             label stands for), fixed values the task supplies rather than derives, conventions adopted (indexing
             base, units, timezone, defaults), and this segment's own results and values (write "x = 7",
             not just "the value was computed").

             Write PART 2 as data, one item per line, values kept as given:
               retry budget = 5; timeout 30s; index starts at 0
               step 3 input: n=64, tolerance 1e-6
               slot B: type=private, cost 688.7-721.3 per unit
             A line should be almost entirely values. Keep the trace's notation -- symbols, units,
             numbers, spellings -- and drop its sentence shell: strip the phrases that merely announce
             "this is a given" rather than the givens themselves ("I would prefer", "it should be somewhere
             between ... and ..."). These phrases repeat once per item; a 15-item list pays
             the space cost 15 times.
             Every FACT carries this information; never write "as above" or "same as the
             previous session" -- retrieval returns only a handful of facts by similarity, and the referent may not be
             among them.

             Keep the total under 1200 characters when possible. If PART 2 doesn't fit, optimize the notation,
             never the coverage: statement is the retrieval vector and also the only field a later segment
             can read (details is neither embedded nor injected), so anything deleted from it is
             permanently lost. Compress the wording around a given; never compress the given itself.

             The test: if the next segment cannot see the original material, can it continue the task from this record alone? Only a yes
             qualifies. A missing binding does not fail loudly -- the name still looks meaningful, so it
             gets read with its general-usage meaning, and the follow-up answer can be perfectly fluent while answering a different question.

             LANGUAGE -- use the same language as the session trace: statement is the retrieval
             vector and queries arrive in the user's working language; switching to another language makes every lookup pay a
             cross-lingual matching penalty.

             EVIDENCE FIDELITY -- record only actions actually executed and evidence explicitly present in the trace.
             Keep material commands, flags, arguments, errors, and numbers as-is.
             Never complete command flags by convention, and never write advice or plans as actions that were executed:
               “ran Get-ChildItem -Force” → “ran Get-ChildItem”
                 (the trace did not contain `-Force`; do not invent it)
             Scope every negative finding to what was actually searched, and enumerate one by one all explicit coverage limits that could invalidate it,
             especially permission failures and depth limits: keep the
             roots, filters, depth (“not found under C:\\ and D:\\ at depth 3;
             some directories were not covered due to Access denied"), never widen it to
             "does not exist". Don't omit any limit just to make the record prettier.
             Never promote a partial search to global absence.
             A blocked attempt must keep its blocker.
             Preserve the distinction among attempted, observed, verified, inferred, and unresolved;
             express it in natural language ("tried ... but was blocked by Access denied",
             "suspected ..., not yet verified"), not as labels. Without tool output or
             explicit confirmation, never claim a fix succeeded, a cause was established, or a path does not exist.

             Do not record greetings or pure chatter -- only in that case output "fact": null. A session that merely stated
             requirements, or stalled before any work began, still established information.

details    — The session's concrete OUTPUT, copied over verbatim. `statement` is a summary, and a summary necessarily
             drops the individual values; a later lookup almost always wants one specific value, not the gist. So whenever the
             session produced an addressable result, reproduce it here in its original form and structure:
               plan / schedule / itinerary → keep every line, every day, every slot, every price
                 exactly as in the original
               config or code block → keep the whole block
               a command's key output → keep the lines that carry the findings (sizes, counts, versions,
                 error text)
               structured result (table, records list) → keep the rows
             Preserve the original layout -- line breaks, `Day 1:` / key: value prefixes, units,
             currency symbols, ids. Do not re-summarize, do not reorder, do not round numbers, do not translate.

             Copy only what is actually present in the trace. Never fill a gap: if a slot was left unfilled,
             leave it unfilled or say so explicitly -- an invented value is worse than a missing one, because it will be retrieved and used.
             Cap at about 4000 characters; if the output is longer, keep the parts carrying concrete values and drop decorative
             prose, and note what was dropped.

             If this session produced no such artifact (a pure investigation with no
             findings yet, or a stall waiting on the user), write "".

             This field is NOT part of the retrieval vector -- `statement` and `subject` do the
             finding; `details` supplies the concrete values once found. So don't stack retrieval bait here, and don't
             repeat the summary; write only the raw content.

files      — 0..N absolute or repo-relative FILE OR DIRECTORY paths actually
             operated on or investigated during this session. Keep investigated roots such as `C:\\` and `D:\\`. Only include paths that were
             actually accessed/searched, not paths merely suggested or planned; write [] when none apply.
═══════════════════════════════════════════════════════════════════
HOST CONTEXT (provided by the SDK)
═══════════════════════════════════════════════════════════════════
- workspace_id present: {workspace_present}
- branch present: {branch_present}

With no workspace_id, the solution cannot rely on undeclared project-local context -- it must spell out the repo / path / service it needs in `solution`
or `boundary_envs` so the memory stays usable on its own.

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

Extraction is the default. When the segment genuinely completed real work, you MUST output at least one experience -- even if
the task looks simple or succeeded in one shot, record the reusable essence of the work done
(method / command / how-to). When the session is thin, keeping the single most valuable experience is enough.

The only exception is the stall case above: if the segment did no real work because it was blocked asking the user for missing information, output
"experiences": [] and write only the fact. Do not manufacture an experience to satisfy this instruction -- a fabricated
"how to ask for the departure city" is worse than no memory at all, because it will later be retrieved and followed. Only this
reason allows an empty array; when work was actually done, never treat an empty array as a shortcut.

"fact" is a SINGLE object, never a list -- one record per session.

Always write the fact. The bar for the fact is not "was this valuable" but "did anything at all happen"; a segment
reaching you has already cleared that bar. This includes segments that look like setup rather than work: a single user
message stating requirements, parameters, or a starting state, even with no tools run, is exactly the material a later session cannot rebuild on its own -- write the requirements, parameters, and starting state together with the situation into
`statement` (compact notation, don't copy the dialogue verbatim); don't push them to `details` (`details` is not in the retrieval vector, and the read side no longer
injects it). Use "fact": null only for greetings and pure chatter with no content whatsoever. When in doubt, write the fact:
a surplus record costs one extra row, while a dropped record harms every later session
that needed it and has no way of knowing it ever existed.
"""


EXTRACT_PROMPT_CODING_ZH = """\
╔═══════════════════════════════════════════════════════════════════════════╗
║ HARD CONSTRAINT -- read this section first; it overrides any wording later in this prompt that sounds like it wants a narrative.   ║
║                                                                           ║
║ You are not writing a report of what some person did. You are writing the information a FUTURE AGENT needs in order to act.        ║
║ That agent already has its own task; it does not care what this user wanted. It needs to know: what was decided here,              ║
║ at which code coordinates, what was tried and rejected, and whether it was verified.                                          ║
║                                                                           ║
║ Specifically, every `statement` and every `solution` must obey the following:                                                  ║
║   · The grammatical subject must be the CODE or THE DECISION -- never "the user", and never   ║
║     "the agent". Do not write "The user attempted to...", "The user wanted...",   ║
║     “The agent then checked…”.                                           ║
║   · Never close with bookkeeping phrases like "the session was completed", "the error was noted",         ║
║     "no solution was found yet". Close with the decision and its verification    ║
║     status. If no decision was made, close with what was ruled out and the next unexplored lead -- that is the ║
║     reusable part.                                                        ║
║   · Keep the concrete symptom (error string, failing command, wrong output) -- future searches ║
║     match on exactly this content.                                                ║
║                                                                           ║
║ bad:  “The user attempted to write a QTable to RST with header_rows and   ║
║        hit a TypeError. The session was completed with the error noted.”  ║
║ good: “ascii.rst's RST writer rejects header_rows:                        ║
║        `tbl.write(..., format='ascii.rst', header_rows=['name','unit'])`  ║
║        raises TypeError because RST.__init__ does not forward it, unlike  ║
║        ascii.fixed_width which accepts it. No fix written — the writer    ║
║        needs the same header_rows plumbing as fixed_width.”               ║
╚═══════════════════════════════════════════════════════════════════════════╝

You extract durable WORKING memories from a task-execution session segment.
A "working" session is any hands-on task where the user operates a computer or
system through tools -- this covers BOTH software/code work (writing, debugging,
refactoring, building, deploying, API/library usage) AND system & operations work
(running shell/PowerShell commands, disk/storage analysis or cleanup, file & folder operations, installing/configuring software, environment/device setup, hardware/disk health inspection, backup, scheduling/automation, data processing).

You produce TWO layers from the SAME segment:

  EXPERIENCES (procedural, "how to do it") -- a reusable method, extracted only
  when it clears the VALUE BAR below. Each is a (task, search_keys, solution, boundary_envs) bundle.

  FACT (episodic, "what was established this time") -- EXACTLY ONE record of THIS session. It is not a narrative of what the user did,
  but records, in this order: the task and its concrete symptom (this is what makes it findable later), the code coordinates,
  what was finally adopted and what was rejected, and how far verification got. It has no value bar, but there can never be more
  than one.

The two layers answer different questions and are stored separately. A method
that generalizes goes in EXPERIENCES; this session's story, which only holds for this system / this project / right now, goes in the FACT.

═══════════════════════════════════════════════════════════════════
VALUE BAR (EXPERIENCES ONLY) -- do not cross this bar lightly.
═══════════════════════════════════════════════════════════════════
A memory must satisfy ALL of the following at once:
  (a) It solves a problem the same user is likely to hit again
  (b) The solution is non-trivial: it combines multiple pieces of info / steps, or
      carries a decision rationale, or records a hard-won workaround, or is a
      reusable procedure/recipe for an operational task (e.g. how to analyze and clean up disk usage, how to batch-organize files, how to check hardware health, how to set up a scheduled job)
  (c) A future agent / user would prefer looking it up over re-discovering it

IMPORTANT -- analysis / diagnostic / investigation tasks COUNT:
A task does not have to actually change or "fix" anything to be worth remembering. Read-only, analytical, and troubleshooting work is valuable whenever its
procedure or conclusion is reusable. Explicitly extract:
  - Analysis / diagnostic recipes: the reusable procedure and what it revealed
    (e.g. "how to analyze which folders eat C-drive space" -- the scan commands, the gotchas hit, and the final finding). "It only produced a report, no change"
    is not a reason to skip -- the reusable method itself is the memory.
  - Investigation outcomes, including negative or not-found results (e.g.
    "searched for folder X across C: and D: with rg / fs_search / SQL index -- not found anywhere"). Recording a dead end keeps the next agent from repeating
    the same fruitless search, so it clears the value bar.
Do not reject a memory merely because "the assistant could regenerate the commands"
or "nothing was actually modified". If the procedure or the conclusion would save future work, keep it.

Do NOT extract:
  - Truly trivial single-step ops with no reusable procedure ("rename one file",
    “run ls once”)
  - One-off issues already resolved with no transferable lesson
  - Information that already exists in the repo/system (don't duplicate package.json)
  - Per-PR / per-session throwaway tasks unless they encode a reusable pattern

Heuristic: if the user (or a future agent doing the same kind of task) would want it
written down instead of figuring it out again, keep it. Multi-step operational procedures usually qualify -- including system/ops or read-only
analysis/search flows, not just code. Only drop it when it is a single command any user could find via `--help` in 30 seconds and it
carries no decision or gotcha worth remembering.

═══════════════════════════════════════════════════════════════════
CODE WORK -- information a future agent cannot re-derive, so it MUST be captured
═══════════════════════════════════════════════════════════════════
When the segment is software work on a repository, the reusable method is only half of it. The other half is where the work lives and
which version won. These cannot be recovered by reading the code later -- the code shows only the current state, not which branch it sits on,
nor which approaches were already rejected.

This section does not ask for a longer record; it asks you to spend the same budget on higher-value tokens: drop restated goals,
narration of what the assistant "then decided to" do, and pasted code that merely echoes; spend those characters on the four items below instead.
A record that omits the branch but quotes 30 lines of diff is a bad trade.

1. BRANCH / WORKING LOCATION -- write it verbatim into the fact `statement`. If the segment mentions or checks out
   `feat/x` / `fix/y`, that token must appear. An agent that resumes work on the wrong branch
   produces changes that are silently discarded afterwards, so this one token outranks every other detail.

2. WHICH VERSION WON -- when the user changes their mind, use one clause to name the winner and mark the loser:
   "settled on A; the interim B was dropped". Don't keep both versions in full -- one clause.
   Keeping only the latest version is wrong: an agent that later meets the rejected code must recognize it, otherwise it will "fix" the code
   back to the already-discarded variant. Phrases like `scratch that / go back to / actually, let's / no, not that`
   are the markers: what follows is the conclusion of record; what precedes it has been superseded.

3. EXACT COORDINATES, NOT PASTED CODE -- use the repo-relative path
   (`qutebrowser/utils/debug.py`, never bare `debug.py`) plus the function /
   class touched. If a signature changed, give the new signature. If the decision was about a constant, regex
   or literal, quote it verbatim -- but quote only that one item, not the surrounding block. One accurate line beats a paragraph of
   approximation.

4. VERIFIED vs MERELY WRITTEN -- four words are enough ("tests run and passing" /
   "confirmed by git diff only, tests not run"). "the edit is in" and "the edit is in
   and tested" lead a future agent to take completely different next steps.

Anti-pattern -- summarizing only the goal and dropping the coordinates:
  bad:  “refactored the colour parsing logic to be cleaner.”
  good: “on feat/q4, in qutebrowser/config/configtypes.py QtColor._parse_value:
         settled on `mult = 360.0 if kind == 'h' else 255.0`; the interim version that
         used 255.0 for everything was dropped. Tests not run.”
  (the good one is often shorter than the bad one -- it simply spends every character on information that cannot be re-derived)

═══════════════════════════════════════════════════════════════════
EXPERIENCE FIELDS
═══════════════════════════════════════════════════════════════════

task          — Query-shaped name; phrase it as how the user would ask later.
                Use "publish a release to the xx platform", not
                “publish_release_internal”.
                Must be self-contained (no "this", "the file", "we").

search_keys   — 0 to 5 alternative phrasings covering aspects that `task` alone
                wouldn't match but that matter to the solution. Phrase them as questions or topics future users would ask, and they must differ from task.
                Omit if task already covers everything.

solution      — Complete, self-contained content the user actually needs: commands,
                credential locations (not credential values), file paths, ordered steps, gotchas, decision rationale. If multi-turn end-to-end work forms
                one coherent task, aggregate it into one solution; split into multiple memories only when the user did unrelated tasks.
                memories.

boundary_envs — The concrete qualifications the solution depends on, multi-line "key: value":
                  runtime / SDK / library + version
                  API endpoint or method signature
                  external service / middleware version
                  env var NAME (not the value)
                  config file path
                  host / cluster / region identifier
                If the solution is environment-independent, write "" (empty string).
                Purpose: after the underlying SDK/middleware upgrades, future
                users can judge from this whether the solution still applies.

confidence  — 0.0 to 1.0
  1.0   the user explicitly stated it as a rule / instruction
  0.7   inferred from tool actions with sufficient evidence
  0.5   weak inference; a single observation only; uncertain

═══════════════════════════════════════════════════════════════════
FACT FIELD -- exactly one per session segment, no value bar
═══════════════════════════════════════════════════════════════════

You write ONE FACT for the whole segment -- a recoverable state record for this
session, not a polished retrospective and never a list of separate findings. A
future agent must be able to resume from it without mistaking a suggestion, an inference, or a partial search for a verified fact.

statement  — A compact summary of THIS session. It carries two jobs at once and must do both: it is the
             retrieval vector (a future agent searches it with a fresh issue report, worded the way
             users report problems), and it is also the payload that agent acts on once found.
             Therefore:

             OPEN with the task and the symptom, using the words a user would use.
                This is what guarantees findability -- keep the surface form close to how the problem was reported.
                Name the specific goal and object ("reclaim disk space under D:\\WXWork",
                “adjust the hue multiplier in QtColor._parse_value on feat/q4”,
                "locate the new_file_search folder"), never a generic "fix a bug" /
                "look for a file". Include the concrete symptom: error string, failing command,
                wrong output.

             THEN give the coordinates and the decision.
                For code work, write the repo-relative path, the symbol, and the branch. State what was finally SETTLED ON,
                and in the same clause mark what was dropped ("settled on X; the interim Y was
                dropped"). This part cannot be re-derived by reading the code -- the code shows only the winner, never
                that Y was tried and then rejected. An agent that later meets the rejected variant must be able to recognize it,
                or it will "fix" the code back to the discarded version.

             CLOSE with the evidence status.
                It must match exactly one of: completed, partially completed, failed, blocked,
                still awaiting verification; also state how it was checked ("tests run and passing" /
                "confirmed by git diff only, tests not run"). Include negative results and dead ends,
                but don't overstate them.

             ⚠️ Never stop at the narrative layer. "The user tried X and hit error Y; no solution found
             yet" is a wasted record: it retrieves easily but cannot guide a future agent to act. If
             this session genuinely reached no decision, say what was ruled out and what the next unexplored lead is --
             that is the reusable part.

             The subject of the sentences should be the code and the decision, not "the user". Use 2 to 4 coherent
             sentences, no labelled sections. Hard ceiling ~400 latin characters (~200 CJK characters).

             EVIDENCE FIDELITY -- mandatory:
             - Record only actions actually executed and evidence explicitly present in the trace. Keep material commands, flags,
               arguments, errors, and numbers as-is. Never complete command flags by convention,
               and never write advice/plans as actions already executed.
             - Preserve the distinction among attempted, observed, verified, inferred, and unresolved.
               Don't write those words as labels in the prose; express their meaning naturally, e.g. "tried ... but was
               blocked by Access denied" or "suspect ..., not yet verified".
             - Without tool output or explicit confirmation, never claim a fix succeeded, a cause was established, a file/folder
               does not exist, or tests passed.
             - Scope every negative finding to what was actually searched, and enumerate one by one all explicit coverage limits that could invalidate it,
               especially permission failures and maximum search depth:
               “not found within the searched parts of C:\\ and D:\\; some directories
               were skipped on Access denied, and the search was also depth-limited”.
               Don't omit any limit to make the summary prettier. Never promote a partial search to global absence.
             - State causality as certain only when the trace directly verified the causal link. Otherwise keep the hedges
               ("suspected"/"inferred"/"not yet verified"); don't rewrite a wrong test assumption
               into the system's root cause.

             Key point -- this text IS the retrieval vector, and it is the only content a future query gets matched against.
             A generic summary is worthless, because every session looks like a generic summary: it would match every query,
             yet answer none of them. So you must write the concrete anchors that make this session findable -- real
             file paths, function / symbol names, branch names, commands, error strings,
             versions, config keys and values, host / service names. If someone later searches for any one of these
             specifics, this summary must be what comes back.
               good: “fixing a failing `npm run build` in CI. Read
                      .github/workflows/ci.yml and package.json; traced it to webpack
                      5.74 hitting ERR_OSSL_EVP_UNSUPPORTED under node 18. Bumping the
                      workflow's node to 20 made the build pass; node 18 still
                      reproduces locally.”
               good: “on branch feat/q0, in qutebrowser/utils/debug.py signal_name:
                      PyQt<5.11 only exposes the name via repr(sig), and repr comes in
                      both a bracketed and a parenthesised form. Settled on two patterns
                      tried in order with fullmatch, breaking on the first hit; the
                      single-regex version was dropped. Confirmed by git diff, tests not
                      run.”
               bad:  “user wanted a build bug fixed, the agent looked at some config
                      files, changed some code, and fixed it.”
                      (no path, no command, no error string, no version -- no retrieval anchor whatsoever;
                       every session could be written this way)
               bad:  “fixed the signal-name parsing compatibility issue on a feature
                      branch.”
                      (which branch? which file? which function? which of the two competing versions won?
                       a week later, when it is retrieved, nothing here is still useful)
             Fidelity examples (bad → good):
               “ran Get-ChildItem -Force” → “ran Get-ChildItem”
                 (the trace did not contain `-Force`; do not invent it)
               “confirmed new_file_search does not exist” →
                 “new_file_search was not found within the searched parts of C:\\ and
                  D:\\; some directories were skipped on Access denied or by the depth
                  limit, so the investigation is only partly done”
               “identical payloads cause a spurious lock error” →
                 “the verification script assumed the last writer is always a fixed
                  rank; that assumption does not hold, so whether there is a lock bug
                  is still unverified”

             Do NOT record greetings or pure chatter. If the segment established nothing at all, output "fact": null.

files      — 0..N absolute or repo-relative FILE OR DIRECTORY paths actually
             operated on or investigated during this session. Keep investigated roots such as `C:\\` and `D:\\`. Only include paths that were
             actually accessed/searched, not paths merely suggested or planned; write [] when none apply.

═══════════════════════════════════════════════════════════════════
HOST CONTEXT (provided by the SDK)
═══════════════════════════════════════════════════════════════════
- workspace_id present: {workspace_present}
- branch present: {branch_present}

With no workspace_id, the solution cannot rely on undeclared project-local context -- it must spell out the repo / path / service it needs in `solution`
or `boundary_envs` so the memory stays usable on its own.

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

MANDATORY: you must output at least one experience; never return an empty list. Even if the task looks simple or succeeded in one shot,
still record the reusable essence of the work done (method / command / how-to). When the session is thin, the
single most valuable experience is enough -- but never output an empty "experiences" array.

"fact" is a SINGLE object, never a list -- one record per session. Use "fact": null only when the segment is
pure chatter that established nothing.
"""


RECONCILE_PROMPT_ZH = """\
You decide how to merge the new coding memory drafts with the user's existing memories.

For each new draft, choose one of the following actions:
  ADD     -- no existing memory describes the same task / aspect
  UPDATE  -- an existing memory describes the same task; replace it with the new draft
            (the solution may have changed, the env may have upgraded, etc.)
  SKIP    -- duplicates existing content; no new information

Additionally, when the new content invalidates or voids existing candidates, you may emit DELETE ops against existing candidates:
  DELETE  -- the new conversation explicitly supersedes / cancels / invalidates an existing memory,
            and it cannot be fixed via UPDATE. Examples:
              * the user says "don't use X anymore", and X exists only in another separate memory
                (not the one currently being ADD/UPDATEd)
              * the candidate's boundary_envs (service / endpoint / SDK) have disappeared,
                and the new conversation confirms it is deprecated

Rules:
- "Same task" means the existing memory and the new draft are both valid answers to the same future user query.
- If the solution differs, prefer UPDATE (do not keep historical versions).
- If both task and solution are highly consistent, choose SKIP.
- DELETE only when there is clear evidence that a candidate is invalid/obsolete. Never
  DELETE a candidate just because it looks "stale".
- When emitting DELETEs, the number of output ops may exceed the number of drafts.

═══════════════════════════════════════════════════════════════════
NEW DRAFTS
═══════════════════════════════════════════════════════════════════
{drafts_block}

═══════════════════════════════════════════════════════════════════
EXISTING CANDIDATES (retrieved by task similarity for each draft)
═══════════════════════════════════════════════════════════════════
{candidates_block}

═══════════════════════════════════════════════════════════════════
OUTPUT (strict JSON only, no markdown)
═══════════════════════════════════════════════════════════════════
{{
  "ops": [
    {{"draft_idx": 0, "action": "ADD"}},
    {{"draft_idx": 1, "action": "UPDATE", "target_experience_id": "exp_xxx", "reason": "..."}},
    {{"draft_idx": 1, "action": "DELETE", "target_experience_id": "exp_yyy", "reason": "user explicitly deprecated Y"}},
    {{"draft_idx": 2, "action": "SKIP",   "reason": "duplicate"}}
  ]
}}

If there are no drafts, output {{"ops": []}}.
"""


FACT_RECONCILE_PROMPT_ZH = """\
You maintain session-level WORKING FACTS. Each statement summarizes an entire trace/session:
the user's task, the rough approach, and the result. It is not an atomic claim.

Every new session fact below will definitely be stored; that is not for you to decide. Your only task is to identify
task continuation: is the new session continuing to advance the same user task recorded by some existing session
fact? If so, mark the older fact as SUPERSEDE so these links form a cross-session progress
chain for that task.

Mark an existing fact as superseded only when there is sufficient evidence that all of the following hold:
  - subjects match: both records revolve around the same entity -- the same person, the same file/module,
    the same host/service. A matching subject is necessary but not sufficient; and
  - the user goal is materially the same, not merely in the same topic/project; and
  - the new trace continues, retries, extends, verifies, or completes the work recorded by the existing trace.

Do NOT supersede when:
  - the two records have different subjects. Two people in the same group, two files in the same module,
    two hosts in the same cluster are parallel, not sequential: one person's state is not a newer version of another person's
    state. If the subject shows (unspecified), fall back to judging from the statement alone.
  - two sessions merely mention the same file, symbol, component, repository, error, or
    technology, but pursue different user goals.
  - one task merely provides context or technique for another task.
  - the relationship is uncertain. Judge conservatively: INSERT is the safe default; merging unrelated sessions destroys their each
    independent task histories.

Treat every statement as evidence-bounded text. Don't reinterpret an inference inside it as an already established
cause, and don't treat a shared conjecture/error as evidence that the user goals are the same. Your judgment covers task
continuation only; do not "correct", complete, or polish any session record.

Examples:
  existing: "need to fix the CI npm run build; traced to Node 18's
             ERR_OSSL_EVP_UNSUPPORTED; the workflow has not been modified yet."
  new:      "continuing the fix of the same CI build; Node in .github/workflows/ci.yml has been
             raised to 20, and npm run build passes."
                                                                 → SUPERSEDE

  existing: "need to fix the JWT expiry handling in auth/middleware.ts; refresh logic has been
             refresh logic.”
  new:      "need to add performance logging in auth/middleware.ts; a latency metric has been
             latency metric.”
                                                                 → keep both
            (same file/project, different user goals)

═══════════════════════════════════════════════════════════════════
NEW SESSION FACTS
═══════════════════════════════════════════════════════════════════
{drafts_block}

═══════════════════════════════════════════════════════════════════
EXISTING SESSION FACTS (retrieved by statement similarity for each new fact)
═══════════════════════════════════════════════════════════════════
{candidates_block}

═══════════════════════════════════════════════════════════════════
OUTPUT (strict JSON only, no markdown)
═══════════════════════════════════════════════════════════════════
{{
  "ops": [
    {{"draft_idx": 0, "supersedes": ["fact_xxx"], "reason": "continues the same CI fix task"}},
    {{"draft_idx": 1, "supersedes": []}}
  ]
}}

Only use fact ids that actually appear in the candidate list of the same draft_idx.
If no fact is superseded anywhere, output {{"ops": []}}.
"""


CHAIN_SUMMARY_PROMPT_ZH = """\
A user task in the working-memory store has advanced across multiple sessions. Below are the
session summaries in that task's progress chain, ordered from oldest to newest; the last one is the current session.

Summarize how the same task advanced across sessions to its current state. Cover every important stage in chronological order. For each
stage, when information allows, keep: the actual action or change, its direct result, and any unresolved
issue or limiting condition. State the current status explicitly at the end.

Rules:
  - This is compression only. Report only what the entries actually state; never add or complete a command, cause,
    motive, action, test result, success state, or global negative conclusion.
  - Preserve evidence states exactly: attempted is not completed, observed is not verified, inferred
    is not established, and unresolved must stay unresolved. Keep uncertainty words and
    search/permission/coverage limits.
  - State the current end status faithfully (completed, partially completed, failed, blocked, or
    still awaiting verification). Never upgrade the status because of a plausible-sounding causal story or something later entries
    did not verify.
  - If the entries mention a suspected cause and a later result, draw a causal link between them
    only when some entry directly states that the cause was verified.
  - Keep the concrete retrieval anchors: file paths, symbols, commands, error strings, versions,
    config keys/values, and the numeric values that matter to the progression.
  - For 2-3 generation chains, don't over-compress the progression into one sentence. When suitable, prefer 2-4 compact paragraphs,
    or well-readable prose that clearly presents the task goal, the stage-by-stage progression, and the current status.
  - Don't mechanically copy the full history. While keeping the important actions, direct results, and leftovers,
    omit repeated and low-value details.
  - Don't invent stage headings or dates. Mention dates only when the entries make dates meaningful;
    timestamps on their own are noise.
  - If the entries say earlier revisions were omitted, you must state explicitly that earlier history was omitted;
    don't imply the provided entries are the complete history.
  - Use the same language as the entries.
  - Write no preamble, and don't wrap the output in quotes.

Example
  entries:
    (2026-07-20) Need to fix the CI npm run build; traced to ERR_OSSL_EVP_UNSUPPORTED under
                 Node 18; the workflow had not been modified yet.
    (2026-07-24) Continued fixing the same CI issue; Node in .github/workflows/ci.yml was raised
                 to 20, but the dependency cache still reproduced the old error.
    (now)        Continued verifying the same fix; after clearing the npm cache, npm run build passed.
  output:
    The task goal was to fix the CI npm run build. The problem was first traced to
    ERR_OSSL_EVP_UNSUPPORTED under Node 18, with the workflow not yet modified at the time; the
    .github/workflows/ci.yml was then raised to Node 20, but the dependency cache still kept the old error
    appearing. Finally, after clearing the npm cache, npm run build passed, so this build fix is currently verified complete.

═══════════════════════════════════════════════════════════════════
ENTRIES
═══════════════════════════════════════════════════════════════════
{chain_block}

═══════════════════════════════════════════════════════════════════
OUTPUT (output only the summary itself, nothing else)
═══════════════════════════════════════════════════════════════════
"""
