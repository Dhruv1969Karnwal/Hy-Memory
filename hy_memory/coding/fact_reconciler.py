# -*- coding: utf-8 -*-
"""
Working Fact Reconciler (fact layer, independent of the experience-layer reconciler)

Input: List[WorkingFactDraft] (from the extractor)
Output: (List[WorkingFact], List[FactReconcileOp])

Each draft is the record of one whole trace/session and is always INSERTed. The only judgement is whether the new
trace keeps advancing the same user task as an older trace. If so, the old fact is SUPERSEDEd, forming a cross-session
task progress chain -- a soft delete rather than a physical one, so fact_link stays traceable, while the experience
chains the old fact accumulated are **inherited** by the new fact so experience does not fade away with it.

Reuses LLMConfig.model (shared with chat and the experience-layer reconciler).

"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..agent.llm_provider import LLMProvider
from ..core.embed_service import EmbedService
from ..utils.lang_detect import is_chinese
from .store import CodingMemoryStore
from .types import (


    MAX_FACT_CHAIN_ENTRIES,
    FactReconcileOp,
    WorkingFact,
    WorkingFactDraft,
    normalize_subject,
    short_date,
)

logger = logging.getLogger(__name__)


def _resolve_configured_temperature(llm_provider) -> Optional[float]:
    """Resolve the temperature from MemoryConfig.llm; None = the parameter is omitted from the request."""
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

# Fixed output budget for the evolution summary: room for the key actions, direct results and leftover state of a
# multi-stage task, without copying the whole history. Part of the summary contract, not a per-instance knob.
_CHAIN_SUMMARY_MAX_TOKENS = 600

# ---- Inheriting details from superseded facts ----
#
# Why this exists: fact_chain_summary is an LLM-compressed artefact whose first prompt rule is
# "This is compression only", and its output is capped by _CHAIN_SUMMARY_MAX_TOKENS=600. The longer the chain,
# the more of the earlier generations is lost in the summary, and concrete values go first (price ranges, score ranges),
# so a summary keeps the narrative but not the numbers.
#
# Inheritance copies instead of generating: no LLM call, none of its failure modes, 100% fidelity within the budget.
# Details are deliberately kept out of vectors (store._upsert_fact_vec), so appending any amount never changes recall scoring;
# it only changes "what can be read after a hit". Those two points together justify the whole approach.
#
# Three problems that must be handled:
#   1. Exponential growth. An ancestor's details may already contain the block it inherited in its own generation, so
#      appending directly doubles every generation. Only each ancestor's "own" details are taken (its inherited tail is cut
#      at the marker); the lineage is already walked in full by _collect_history, not accumulated generation by generation.
#   2. Context budget. If callers inject details verbatim into context, k facts each carrying several inherited generations
#      would burn through the context quickly, hence the total budget.
#   3. Trade-offs inside the budget. Fill from the newest generation backwards and stop when it no longer fits -- recent
#      values are likelier to still be valid. This step is lossy and what was lost is noted at the tail of the block, never truncated silently.
_DETAILS_INHERIT_MARKER = "--- inherited from superseded facts ---"# Total budget. Roughly the extractor's per-fact details cap (_MAX_DETAILS_CHARS=4000), i.e. the "inherited
# all history" so it takes up at most one more fact's worth of space and never overshadows the current state.
_MAX_INHERITED_DETAILS_CHARS = 4000
# Per-ancestor cap, so one huge details field cannot eat the whole budget and leave later generations with nothing.
_MAX_INHERITED_PER_FACT_CHARS = 1200
_INHERIT_DETAILS_DEFAULT = True
ENV_INHERIT_DETAILS = "HY_FACT_INHERIT_DETAILS"


def _inherit_details_default() -> bool:
    """Same as `_subject_gate_default`: env var fallback, constructor argument takes precedence."""
    raw = os.getenv(ENV_INHERIT_DETAILS)
    if raw is None or not raw.strip():
        return _INHERIT_DETAILS_DEFAULT
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _own_details(details: str) -> str:
    """Strip the section a fact inherited itself, keeping only the details its own generation produced.

    Returned unchanged when the marker is absent. This is the key step against exponential growth: the lineage is walked
    in full by _collect_history, so each generation contributes only its own share.
    """
    if not details:
        return ""
    head = details.split(_DETAILS_INHERIT_MARKER, 1)[0]
    return head.rstrip()


def _build_inherited_details(history: List[WorkingFact]) -> Tuple[str, int, int]:
    """Assemble the ancestors' own details into one inherited block tagged by generation.

    Args:
        history: historical facts in chronological order (the return value of _collect_history).

    Returns:
        (inherited block text, generations included, generations skipped for budget). Returns ("", 0, 0) when there is nothing to inherit.
    """
    # Fill from the newest generation backwards: recent values are likelier to still be valid, and summary fidelity for
    # distant generations is the lowest anyway; the budget is hard, so keep what is most likely to be used first.
    usable = [(i, f) for i, f in enumerate(history) if _own_details(f.details)]
    blocks: List[Tuple[int, str]] = []
    used = 0
    skipped = 0
    for pos, fact in reversed(usable):
        body = _own_details(fact.details)
        if len(body) > _MAX_INHERITED_PER_FACT_CHARS:
            body = body[:_MAX_INHERITED_PER_FACT_CHARS] + "\n…(truncated)"
        label = f"[gen{pos + 1}"
        if fact.subject:
            label += f" · {fact.subject}"
        # created_at is a datetime; short_date takes an ISO string (same usage as _summarize_chain)
        stamp = short_date(fact.created_at.isoformat() if fact.created_at else "")
        if stamp:
            label += f" · {stamp}"
        label += "]"
        chunk = f"{label}\n{body}"
        if used + len(chunk) > _MAX_INHERITED_DETAILS_CHARS:
            skipped += 1
            continue
        used += len(chunk)
        blocks.append((pos, chunk))
    if not blocks:
        return "", 0, skipped
    # Presentation stays chronological so it reads as one progression; only budget allocation runs newest first.
    blocks.sort(key=lambda item: item[0])
    text = f"{_DETAILS_INHERIT_MARKER}\n" + "\n\n".join(chunk for _, chunk in blocks)
    if skipped:
        text += f"\n\n[{skipped} earlier generation(s) omitted for length]"
    return text, len(blocks), skipped



# Switch for the mechanical subject gate.
#   True  = after recall, candidates whose normalized subject differs are removed from the list so the LLM never sees
#           them, and they are checked once more before SUPERSEDE runs.
#   False (default) = gate off; the supersede decision relies only on semantic recall + the LLM.
#
# Why a switch: the payoff depends on whether subjects are stable in the first place.
#   - When subjects are stable entity names (people, files, services) sitting side by side, the gate prevents
#     cross-subject merges with almost no collateral damage.
#   - When subjects are free text or positional labels, the same entity gets written under different subjects, and
#     equality comparison blocks genuine updates; if a later step depends on an earlier one, it then reads a stale value.
# A mechanism that only holds in some settings should not be the default, so it ships disabled.
#
# Known cost of leaving it off: cross-subject merges come back. The compensation is fact_chain_summary, which keeps
# the evolution of superseded facts, though its fidelity decays with chain distance.
_SUBJECT_GATE_DEFAULT = False
ENV_SUBJECT_GATE = "HY_FACT_SUBJECT_GATE"


def _subject_gate_default() -> bool:
    """Env-var fallback so the default behaviour can be switched without touching code.

    Unset → use `_SUBJECT_GATE_DEFAULT`. Set to 0/false/no/off → disabled.
    A value passed explicitly to `WorkingFactReconciler(subject_gate=...)` takes precedence over the environment variable.
    """
    raw = os.getenv(ENV_SUBJECT_GATE)
    if raw is None or not raw.strip():
        return _SUBJECT_GATE_DEFAULT
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _subject_conflict(new_subject: str, old_subject: str) -> bool:
    """Whether two facts clearly have different subjects (different → supersede forbidden).

    A conflict exists only when both sides declare a subject and the normalized values differ. An empty side returns False:
    empty does not mean "different subject", it means "no subject could be determined this time" (the extractor omitted it,
    the fallback stored the source text, or the row predates this change). Treating empty as a conflict would make facts in
    old stores impossible to supersede forever, i.e. vetoing a judgement with an unknown; handing it back to the LLM is the old behaviour.
    """
    a, b = normalize_subject(new_subject), normalize_subject(old_subject)
    return bool(a) and bool(b) and a != b



# ================================================================
# Prompt
# ================================================================
#
# The two subject rules below (a matching subject as a necessary condition for
# SUPERSEDE, and different subjects as a reason to keep both) are general: they
# are stated over persons, files/modules, and hosts/services alike, and they say
# what "successive" means for any memory store, not what any one dataset looks
# like.
#
# Avoid few-shot examples whose surface form resembles one specific dataset.
# General subject-isolation rules belong in the prompt; lookalike examples
# encourage pattern matching instead of teaching the rule.

FACT_RECONCILE_PROMPT = """\
You maintain session-level WORKING FACTS. Each statement summarizes one whole
trace/session: the user's task, the rough approach, and the result. It is not an
atomic claim.

Every new session fact below WILL be stored; that is not your decision. Your only
job is to detect task continuation: is the new session continuing to advance the
SAME user task as an existing session fact? If so, SUPERSEDE the older fact so
the links form a cross-session progress chain for that task.

Mark an existing fact as superseded ONLY when there is strong evidence that:
  - The subjects match: both records are about the SAME entity — the same
    person, the same file/module, the same host/service. A shared subject is
    necessary but not sufficient; AND
  - The user goal is materially the SAME, not merely in the same topic/project;
    AND
  - The new trace continues, retries, extends, verifies, or finishes the work
    recorded by the existing trace.

Do NOT supersede when:
  - The two records name different subjects. Two people in one group, two files
    in one module, two hosts in one cluster are PARALLEL, not successive: one
    person's state is not a newer version of another's. Where a subject is shown
    as (unspecified), fall back to judging by the statement alone.
  - The sessions merely mention the same file, symbol, component, repository,
    error, or technology but pursue different user goals.
  - One task only supplies context or a technique for another task.
  - The relationship is uncertain. Be conservative: INSERT is the safe default;
    merging unrelated sessions destroys their separate task history.

Treat each statement as evidence-bounded text. Do not reinterpret an inference
inside it as an established cause or use a shared conjecture/error as proof that
the user goals are the same. Your decision is only about task continuation; do
not "correct", complete, or embellish either session record.

Examples:
  existing: "The user needed to fix CI's npm run build; traced it to
             ERR_OSSL_EVP_UNSUPPORTED under Node 18, workflow not yet changed."
  new:      "The user continued fixing the same CI build; raised Node in
             .github/workflows/ci.yml to 20 and npm run build passed."
                                                                 → SUPERSEDE

  existing: "The user needed to fix JWT expiry handling in
             auth/middleware.ts; refresh logic was added."
  new:      "The user needed performance logging added to
             auth/middleware.ts; a latency metric was introduced."
                                                                 → keep both
            (same file/project, different user goal)

═══════════════════════════════════════════════════════════════════
NEW SESSION FACTS
═══════════════════════════════════════════════════════════════════
{drafts_block}

═══════════════════════════════════════════════════════════════════
EXISTING SESSION FACTS (per new fact, retrieved by statement similarity)
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

Only use fact ids that appear in the candidate list for that same draft_idx.
If nothing is superseded anywhere, output {{"ops": []}}.
"""


CHAIN_SUMMARY_PROMPT = """\
A user task in a working-memory store has progressed across sessions. Below are
the session summaries in that task's progress chain, oldest first, ending with
the current session.

Summarize how the SAME task progressed across sessions to its current state.
Cover every important stage in chronological order. For each stage, preserve
where available: the actual action or change, its direct result, and any
unresolved issue or limiting condition. End with an explicit current status.

Rules:
  - This is compression only. Report ONLY what the entries say; never add or
    complete a command, cause, motive, action, test result, success state, or
    global negative conclusion.
  - Preserve evidence status exactly: attempted is not completed, observed is
    not verified, inferred is not established, and unresolved stays unresolved.
    Keep uncertainty words and search/permission/coverage limits.
  - State the current ending status faithfully (completed, partially completed,
    failed, blocked, or still awaiting verification). Never upgrade it based on
    a plausible causal story or on a later entry that did not verify it.
  - When entries mention a suspected cause and a later result, connect them
    causally only if an entry directly says that cause was verified.
  - Preserve concrete retrieval anchors: file paths, symbols, commands, error
    strings, versions, config keys/values, and important numeric values that
    matter to the progress.
  - For chains of 2-3 generations, do not over-compress the progression into one
    sentence. Prefer 2-4 compact paragraphs when useful, or readable prose with
    a clear task goal, stage-by-stage progression, and current status.
  - Do not mechanically copy the full history. Omit repetition and low-value
    detail while retaining important actions, direct results, and leftovers.
  - Do not invent stage headings or dates. Mention a date only when the entries
    make it meaningful; bare timestamps are noise.
  - If the entries say older revisions were omitted, explicitly state that
    earlier history was omitted; do not imply the supplied entries are complete.
  - Write in the same language the entries are written in.
  - No preamble or enclosing quotes.

Example
  entries:
    (2026-07-20) The user needed to fix CI's npm run build; traced it to
                 ERR_OSSL_EVP_UNSUPPORTED under Node 18, workflow not yet
                 changed.
    (2026-07-24) The user continued that CI fix; raised Node in
                 .github/workflows/ci.yml to 20, but the dependency cache
                 still reproduced the old error.
    (now)        The user continued verifying the same fix; after clearing the
                 npm cache, npm run build passed.
  output:
    The task goal is fixing CI's npm run build. It was first traced to
    ERR_OSSL_EVP_UNSUPPORTED under Node 18, with the workflow not yet changed;
    then .github/workflows/ci.yml was raised to Node 20, but the dependency
    cache kept the old error present. Finally, after clearing the npm cache,
    npm run build passed, so this build fix is currently verified as complete.

═══════════════════════════════════════════════════════════════════
ENTRIES
═══════════════════════════════════════════════════════════════════
{chain_block}

═══════════════════════════════════════════════════════════════════
OUTPUT (the summary itself, nothing else)
═══════════════════════════════════════════════════════════════════
"""


def _fact_drafts_language_text(drafts: List[WorkingFactDraft]) -> str:
    parts = []
    for draft in drafts:
        # statement is the actual retrieval body and the main source of this session's language; subject is often an English
        # file or symbol name, so it must not outrank a Chinese statement when deciding the prompt language.
        parts.extend([draft.statement, draft.details, draft.subject])
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def build_fact_reconcile_prompt(
    drafts: List[WorkingFactDraft],
    candidates: List[List[WorkingFact]],
) -> Tuple[str, str]:
    """Return the language-matched fact reconcile prompt and language."""
    if is_chinese(_fact_drafts_language_text(drafts)):
        from .prompts_zh import FACT_RECONCILE_PROMPT_ZH

        template = FACT_RECONCILE_PROMPT_ZH
        language = "zh"
    else:
        template = FACT_RECONCILE_PROMPT
        language = "en"
    return template.format(
        drafts_block=WorkingFactReconciler._format_drafts(drafts),
        candidates_block=WorkingFactReconciler._format_candidates(drafts, candidates),
    ), language


def build_chain_summary_prompt(
    chain_block: str,
    *,
    current_statement: str,
) -> Tuple[str, str]:
    """Return the language-matched cross-session summary prompt and language."""
    if is_chinese(current_statement or chain_block):
        from .prompts_zh import CHAIN_SUMMARY_PROMPT_ZH

        template = CHAIN_SUMMARY_PROMPT_ZH
        language = "zh"
    else:
        template = CHAIN_SUMMARY_PROMPT
        language = "en"
    return template.format(chain_block=chain_block), language


# ================================================================
# WorkingFactReconciler
# ================================================================

class WorkingFactReconciler:
    """
    Fact-layer reconcile + persistence executor.

    One call:
      1. Each draft recalls the top-N active facts by statement embedding
      2. A single LLM call decides which old facts are predecessor sessions of the same user task
      3. SUPERSEDE is handled first (soft delete + experience-chain inheritance), then new facts are INSERTed
      4. Only when a supersede really happened, one more LLM call writes the cross-session task progress summary

    Step 4 only fires when something was superseded, so the vast majority of writes cost nothing extra.
    """

    def __init__(
        self,
        store: CodingMemoryStore,
        embed_service: EmbedService,
        llm_provider: LLMProvider,
        *,
        max_tokens: int = 1200,
        temperature: Optional[float] = None,
        candidates_per_draft: int = 5,
        enable_chain_summary: bool = True,
        subject_gate: Optional[bool] = None,
        inherit_details: Optional[bool] = None,
    ):
        self.store = store
        self.embed_service = embed_service
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        # None → reuse LLMConfig.temperature (avoids hardcoded values tripping kimi/deepseek temperature constraints)
        if temperature is None:
            # None passed all the way to the provider = the parameter is omitted; the previous 0.1 fallback bypassed user config and
            # returned 400 on gateways accepting only a specific temperature (Opus 5 accepts only 1 or nothing)
            temperature = _resolve_configured_temperature(llm_provider)
        self.temperature = temperature
        self.candidates_per_draft = candidates_per_draft
        # Disabled means no evolution summary, saving one call
        self.enable_chain_summary = enable_chain_summary
        # Mechanical subject gate: None → read HY_FACT_SUBJECT_GATE, then fall back to the default.
        # When off, only semantic recall + the LLM decide supersession; see `_SUBJECT_GATE_DEFAULT` at the top of the file.
        self.subject_gate = (
            _subject_gate_default() if subject_gate is None else bool(subject_gate)
        )
        # Inheriting details from superseded facts: None → read HY_FACT_INHERIT_DETAILS,
        # then fall back to the default. See the notes above `_build_inherited_details`.
        self.inherit_details = (
            _inherit_details_default() if inherit_details is None else bool(inherit_details)
        )

    async def reconcile(
        self,
        drafts: List[WorkingFactDraft],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        source: str = "default",
    ) -> Tuple[List[WorkingFact], List[FactReconcileOp]]:
        """
        Run reconcile + persistence.

        Returns:
            (new facts actually persisted, ops executed). If the LLM dies it degrades to all INSERT with no supersede:
            keeping one stale fact is better than losing conclusions because a single call failed.
        """
        if not drafts:
            return [], []

        candidates = await self._retrieve_candidates(
            drafts, user_id=user_id, source=source,
        )

        try:
            supersedes = await self._llm_decide(drafts, candidates)
        except Exception as e:
            logger.warning(
                f"[fact-reconcile] LLM decide failed, insert without supersede: {e}"
            )
            supersedes = {}

        facts, ops = await self._execute(
            drafts,
            candidates,
            supersedes,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            workspace_id=workspace_id,
            branch=branch,
            source=source,
        )
        logger.info(
            f"[fact-reconcile] drafts={len(drafts)} inserted={len(facts)} "
            f"superseded={sum(1 for o in ops if o.action == 'SUPERSEDE')}"
        )
        return facts, ops

    # ------------------------------------------------------------
    # Recall candidates
    # ------------------------------------------------------------

    async def _retrieve_candidates(
        self, drafts: List[WorkingFactDraft], *, user_id: str, source: str
    ) -> List[List[WorkingFact]]:
        """For each draft, recall the top-N active facts of the same user by statement.

        With `self.subject_gate` on (default), recall passes through the subject gate: candidates with a different subject are
        dropped from the list, so the LLM never sees them and cannot even misjudge. This is not an optimization that saves a call
        but a fix for a real defect: several parallel subjects in one workspace (one itinerary shared by several people,
        one script shared by several hosts) have semantically very similar statements, and pure semantics + LLM would read them
        as successive versions of the same task.

        When off, the whole recall is handed to the LLM unchanged. Right for datasets where subject is not a stable entity name
        (free text, positional labels); there exact string equality splits one entity into different subjects and blocks real updates.
        """
        try:
            embeddings = await self.embed_service.embed_batch(
                [d.statement for d in drafts]
            )
        except Exception as e:
            logger.warning(f"[fact-reconcile] embed_batch failed: {e}")
            return [[] for _ in drafts]

        result: List[List[WorkingFact]] = []
        for draft, emb in zip(drafts, embeddings):
            try:
                hits = await self.store.search_facts(
                    emb, user_id=user_id, top=self.candidates_per_draft,
                    # The write side also takes the lexical channel: if the candidates are wrong the LLM gets no chance to decide right.
                    # No min_score here -- the write side wants "which predecessors are possible", so the LLM sees a wider list;
                    # threshold filtering is the read side's job.
                    query_text=draft.statement,
                    include_sources=[source],
                )
            except Exception as e:
                logger.warning(f"[fact-reconcile] retrieve candidates failed: {e}")
                result.append([])
                continue
            if not self.subject_gate:
                result.append([fact for fact, _ in hits])
                continue
            kept, blocked = [], []
            for fact, _ in hits:
                if _subject_conflict(draft.subject, fact.subject):
                    blocked.append(fact)
                else:
                    kept.append(fact)
            if blocked:
                logger.info(
                    f"[fact-reconcile] subject gate blocked {len(blocked)} candidate(s) "
                    f"for subject={draft.subject!r}: "
                    + ", ".join(f"{f.fact_id}({f.subject!r})" for f in blocked)
                )
            result.append(kept)
        return result

    # ------------------------------------------------------------
    # LLM decision
    # ------------------------------------------------------------

    async def _llm_decide(
        self,
        drafts: List[WorkingFactDraft],
        candidates: List[List[WorkingFact]],
    ) -> Dict[int, List[str]]:
        """Returns {draft_idx: [old fact_ids it supersedes, ...]}."""
        # With not a single candidate there is nothing to decide, saving one LLM call
        if not any(candidates):
            return {}

        prompt, _ = build_fact_reconcile_prompt(drafts, candidates)
        resp = await self.llm_provider.complete(
            prompt=prompt,
            max_tokens=self.max_tokens,
        )
        return self._parse_ops(resp.content)

    # ------------------------------------------------------------
    # Evolution summary
    # ------------------------------------------------------------

    async def _collect_history(
        self, retired: List[WorkingFact], *, user_id: str
    ) -> Tuple[List[WorkingFact], bool]:
        """Flatten out "how this task progressed before", chronologically, returning (generations, ever truncated).

        History is not stored separately, it lives in the database: superseded facts are kept as they are, fact_link points to the
        successor, and walking back yields the full lineage. retired are the facts just replaced this time, each traced upward.

        Walking too deep blows up the prompt; past MAX_FACT_CHAIN_ENTRIES the oldest entries are dropped and the caller is told
        that something was dropped, so the LLM does not assume this is the whole history. Only "was anything dropped" is reported, not "how many generations":
        an exact count would need the whole chain walked recursively, which no prompt line is worth; one extra step is enough to tell whether anything older exists.
        """
        by_id: Dict[str, WorkingFact] = {}
        for old in retired:
            try:
                ancestors = await self.store.get_fact_ancestors(
                    old.fact_id, user_id=user_id,
                    max_depth=MAX_FACT_CHAIN_ENTRIES + 1,
                )
            except Exception as e:
                logger.warning(f"[fact-reconcile] walk ancestors failed: {e}")
                ancestors = []
            # Replacing several facts at once merges their lineages; common ancestors deduplicate naturally by fact_id
            for f in ancestors:
                by_id[f.fact_id] = f
            by_id[old.fact_id] = old

        ordered = sorted(by_id.values(), key=lambda f: f.created_at or datetime.min)
        truncated = len(ordered) > MAX_FACT_CHAIN_ENTRIES
        return ordered[-MAX_FACT_CHAIN_ENTRIES:], truncated

    async def _summarize_chain(
        self, history: List[WorkingFact], current: str, *, truncated: bool = False
    ) -> str:
        """Have the LLM write the cross-session progress of one task into a short summary.

        Returns an empty string on failure rather than raising: this step is not worth failing the whole write for, and the history
        is not lost either -- it stays in the database, and the next time this fact is superseded the chain is walked again and the summary repairs itself.
        """
        if not self.enable_chain_summary or not history:
            return ""

        rows = [
            f"({short_date(f.created_at.isoformat() if f.created_at else '')}) "
            f"{f.statement}".strip()
            for f in history
        ]
        if truncated:
            rows.insert(0, "(... older revisions omitted)")
        rows.append(f"(now) {current}")
        prompt, _ = build_chain_summary_prompt(
            "\n".join(rows),
            current_statement=current,
        )

        try:
            resp = await self.llm_provider.complete(
                prompt=prompt,
                max_tokens=_CHAIN_SUMMARY_MAX_TOKENS,
            )
        except Exception as e:
            logger.warning(f"[fact-reconcile] chain summary failed: {e}")
            return ""

        return self._clean_summary(getattr(resp, "content", "") or "")

    @staticmethod
    def _clean_summary(text: str) -> str:
        """Strip decorations models like to add: code fences, outer quotes, "Summary:"-style preambles."""
        s = (text or "").strip()
        if s.startswith("```"):
            # Drop the first line ``` or ```json, plus the closing fence
            s = s.split("\n", 1)[-1] if "\n" in s else ""
            s = s.rsplit("```", 1)[0].strip()
        for prefix in ("\u603b\u7ed3\uff1a", "\u603b\u7ed3:", "Summary:", "summary:"):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        # Only strip when the whole sentence is wrapped in one pair of quotes. Leave it alone if it also quotes something else
        # (a path, a variable name); otherwise `"config" points to "env"` would be shaved down to `config" points to "env`.
        if len(s) >= 2 and s[0] in "\"'“「" and s[-1] in "\"'”」":
            inner = s[1:-1]
            if s[0] not in inner and s[-1] not in inner:
                s = inner.strip()
        # Models sometimes spread the text over lines; collapse it to one line so prompt formatting stays clean
        return " ".join(s.split())

    @staticmethod
    def _format_drafts(drafts: List[WorkingFactDraft]) -> str:
        rows = [
            f"draft[{i}]: subject={d.subject!r} statement={d.statement!r}"
            if d.subject
            else f"draft[{i}]: subject=(unspecified) statement={d.statement!r}"
            for i, d in enumerate(drafts)
        ]
        return "\n".join(rows) if rows else "(none)"

    @staticmethod
    def _format_candidates(
        drafts: List[WorkingFactDraft],
        candidates: List[List[WorkingFact]],
    ) -> str:
        rows = []
        for i in range(len(drafts)):
            cands = candidates[i] if i < len(candidates) else []
            if not cands:
                rows.append(f"for draft[{i}]: (no candidates)")
                continue
            cand_rows = [
                f"  - fact_id={c.fact_id}\n"
                f"    subject: {repr(c.subject) if c.subject else '(unspecified)'}\n"
                f"    statement: {c.statement!r}\n"
                f"    created_at: {c.created_at.isoformat() if c.created_at else 'n/a'}"
                for c in cands
            ]
            rows.append(f"for draft[{i}]:\n" + "\n".join(cand_rows))
        return "\n\n".join(rows) if rows else "(none)"

    @staticmethod
    def _parse_ops(text: str) -> Dict[int, List[str]]:
        """Parse ops leniently, returning {draft_idx: [fact_id...]}."""
        if not text:
            return {}
        s = text.strip()
        # Remove the markdown fence
        if s.startswith("```"):
            body = [line for line in s.split("\n") if not line.startswith("```")]
            s = "\n".join(body).strip() or s

        obj: Optional[Dict[str, Any]] = None
        try:
            obj = json.loads(s)
        except Exception:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end > start:
                try:
                    obj = json.loads(s[start: end + 1])
                except Exception:
                    obj = None

        if not isinstance(obj, dict):
            logger.warning(f"[fact-reconcile] failed to parse ops output: {text!r}")
            return {}
        ops_raw = obj.get("ops")
        if not isinstance(ops_raw, list):
            return {}

        result: Dict[int, List[str]] = {}
        for raw in ops_raw:
            if not isinstance(raw, dict):
                continue
            try:
                idx = int(raw.get("draft_idx"))
            except Exception:
                continue
            ids = raw.get("supersedes") or []
            if not isinstance(ids, list):
                continue
            clean = [str(x).strip() for x in ids if str(x).strip()]
            if clean:
                result.setdefault(idx, []).extend(clean)
        return result

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    async def _execute(
        self,
        drafts: List[WorkingFactDraft],
        candidates: List[List[WorkingFact]],
        supersedes: Dict[int, List[str]],
        *,
        user_id: str,
        agent_id: str,
        session_id: Optional[str],
        workspace_id: Optional[str],
        branch: Optional[str] = None,
        source: str = "default",
    ) -> Tuple[List[WorkingFact], List[FactReconcileOp]]:
        inserted: List[WorkingFact] = []
        ops: List[FactReconcileOp] = []
        # An old fact may be superseded only once, otherwise fact_link points at a later one and the chain breaks
        already_superseded: set = set()

        for idx, draft in enumerate(drafts):
            new_fact = WorkingFact(
                fact_id=str(uuid.uuid4()),
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                workspace_id=workspace_id,
                branch=branch,
                source=source,
                statement=draft.statement,
                files=list(draft.files),
                subject=draft.subject,
                details=draft.details,
                status="active",
            )

            allowed = {c.fact_id for c in (candidates[idx] if idx < len(candidates) else [])}
            targets = [
                fid for fid in supersedes.get(idx, [])
                if fid in allowed and fid not in already_superseded
            ]

            # Inherit the experience chains before soft-deleting old facts, so the chain is already complete when the new fact lands
            inherited: List[str] = []
            retired: List[WorkingFact] = []
            for old_id in targets:
                old = await self.store.get_fact(old_id, user_id=user_id)
                if old is None:
                    continue
                # The gate already filtered once during candidate selection; check again here. Candidates are recalled by statement,
                # and the LLM may only supersede ids from that list, so this should be unreachable. Reaching it means some earlier
                # step leaked; better to let this supersede have no effect -- keeping an extra old fact is recoverable, while deleting
                # another subject's state is not. The re-check is skipped when the gate is off, otherwise the gate is not fully closed.
                if self.subject_gate and _subject_conflict(draft.subject, old.subject):
                    logger.warning(
                        f"[fact-reconcile] refused cross-subject SUPERSEDE: "
                        f"{old.fact_id}({old.subject!r}) by draft subject {draft.subject!r}"
                    )
                    continue
                try:
                    ok = await self.store.supersede_fact(old_id, new_fact.fact_id)
                except Exception as e:
                    logger.warning(f"[fact-reconcile] SUPERSEDE failed: {e}")
                    continue
                if not ok:
                    continue
                already_superseded.add(old_id)
                for eid in old.derived_experience_ids:
                    if eid not in inherited:
                        inherited.append(eid)
                retired.append(old)
                ops.append(FactReconcileOp(
                    action="SUPERSEDE",
                    draft_idx=idx,
                    fact_id=old_id,
                    superseded_by=new_fact.fact_id,
                ))

            new_fact.derived_experience_ids = inherited
            if retired:
                history, truncated = await self._collect_history(
                    retired, user_id=user_id
                )
                new_fact.fact_chain_summary = await self._summarize_chain(
                    history, new_fact.statement, truncated=truncated
                )
                # Beyond the summary, inherit each generation's own details verbatim. The summary preserves the narrative, inheritance
                # the numbers; they do not overlap because details never enter the summary (_collect_history only reads statement).
                if self.inherit_details:
                    block, taken, skipped = _build_inherited_details(history)
                    if block:
                        new_fact.details = (
                            f"{new_fact.details}\n\n{block}".strip()
                            if new_fact.details
                            else block
                        )
                        logger.info(
                            f"[fact-reconcile] inherited details from {taken} generation(s)"
                            + (f", {skipped} skipped for length" if skipped else "")
                            + f" into {new_fact.fact_id}"
                        )

            try:
                await self.store.insert_fact(new_fact)
            except Exception as e:
                logger.warning(f"[fact-reconcile] INSERT failed: {e}")
                continue

            # Backfill the new fact into the inherited experiences so both sides of the chain stay symmetric
            for eid in inherited:
                await self._append_source_fact(eid, new_fact.fact_id, user_id=user_id)

            inserted.append(new_fact)
            ops.append(FactReconcileOp(
                action="INSERT", draft_idx=idx, fact_id=new_fact.fact_id
            ))

        return inserted, ops

    async def _append_source_fact(
        self, experience_id: str, fact_id: str, *, user_id: str
    ) -> None:
        try:
            exp = await self.store.get_by_id(experience_id, user_id=user_id)
            if exp is None:
                # The experience was already DELETEd; the dead link is covered by the existence filter at retrieval time
                return
            if fact_id in exp.source_fact_ids:
                return
            await self.store.set_experience_source_facts(
                experience_id, list(exp.source_fact_ids) + [fact_id]
            )
        except Exception as e:
            logger.warning(f"[fact-reconcile] backfill source_fact_ids failed: {e}")
