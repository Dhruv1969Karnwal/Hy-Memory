# -*- coding: utf-8 -*-
"""
Coding Memory Reconciler

Input: List[CodingMemoryDraft] (from the extractor)
Output: List[ReconcileOp] -- ADD / UPDATE / DELETE / SKIP, **the op count may exceed the draft count**
        (DELETE is emitted as a side output when the LLM observes an "explicit negation/invalidation" between a draft and a candidate)

Executor: reconcile() calls the store to carry out the actual ADD / UPDATE / DELETE persistence.

Reuses LLMConfig.model (shared with the chat reconciler).

For the public scene and how to call it, see docs/scene_coding_usage.md.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..agent.llm_provider import LLMProvider
from ..core.embed_service import EmbedService
from ..utils.lang_detect import is_chinese
from .judge import _store_coding_log
from .store import CodingMemoryStore
from .types import CodingMemory, CodingMemoryDraft, ReconcileOp



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


# ================================================================
# Prompt
# ================================================================

RECONCILE_PROMPT = """\
You decide how to merge new coding memory drafts with the user's existing memories.

For each new draft, decide one of:
  ADD     — no existing memory describes the same task/aspect
  UPDATE  — an existing memory describes the same task; the new draft replaces it
            (solution may have changed, env may have upgraded, etc.)
  SKIP    — duplicate of existing; nothing new

Additionally, you may emit DELETE ops on existing candidates when the new
content makes them obsolete or invalid:
  DELETE  — the new conversation EXPLICITLY supersedes/cancels/invalidates an
            existing memory that cannot be repaired via UPDATE. Examples:
              * user says "don't use X anymore" and X is only present as a
                separate memory (not the one being added/updated)
              * candidate's boundary_envs (service / endpoint / SDK) is gone
                and the new conversation confirms the deprecation

Rules:
- "Same task" means the existing memory and the new draft would both be valid
  answers to the same future user query.
- If solution differs, prefer UPDATE (we don't keep history).
- If both task and solution match closely, SKIP.
- DELETE only when there is EXPLICIT evidence the candidate is invalid/obsolete.
  Never DELETE just because a candidate looks "stale".
- Output ops can outnumber drafts when DELETEs are produced.

═══════════════════════════════════════════════════════════════════
NEW DRAFTS
═══════════════════════════════════════════════════════════════════
{drafts_block}

═══════════════════════════════════════════════════════════════════
EXISTING CANDIDATES (per draft, retrieved by task similarity)
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


def _drafts_language_text(drafts: List[CodingMemoryDraft]) -> str:
    """Build language-routing text from new drafts, not retrieved candidates."""
    parts = []
    for draft in drafts:
        parts.extend([
            draft.task,
            *(draft.search_keys or []),
            draft.solution,
            draft.boundary_envs,
        ])
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def build_experience_reconcile_prompt(
    drafts: List[CodingMemoryDraft],
    candidates_per_draft: List[List[CodingMemory]],
) -> Tuple[str, str]:
    """Return the language-matched experience reconcile prompt and language."""
    drafts_block = CodingMemoryReconciler._format_drafts(drafts)
    candidates_block = CodingMemoryReconciler._format_candidates(
        drafts, candidates_per_draft,
    )
    if is_chinese(_drafts_language_text(drafts)):
        from .prompts_zh import RECONCILE_PROMPT_ZH

        template = RECONCILE_PROMPT_ZH
        language = "zh"
    else:
        template = RECONCILE_PROMPT
        language = "en"
    return template.format(
        drafts_block=drafts_block,
        candidates_block=candidates_block,
    ), language


# ================================================================
# CodingMemoryReconciler
# ================================================================

class CodingMemoryReconciler:
    """
    Reconcile + persistence executor.

    One call:
      1. Retrieve the top-3 candidates for each draft by task embedding
      2. Produce ReconcileOp[] from a single LLM decision
      3. Call the store to execute ADD / UPDATE / DELETE
    """

    def __init__(
        self,
        store: CodingMemoryStore,
        embed_service: EmbedService,
        llm_provider: LLMProvider,
        *,
        max_tokens: int = 1500,
        temperature: Optional[float] = None,
        candidates_per_draft: int = 3,
        cache: Any = None,
    ):
        self.store = store
        self.embed_service = embed_service
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        # When cache is set, persist a CODING_RECONCILE pipeline log (best-effort)
        self.cache = cache
        # None → reuse LLMConfig.temperature (avoids a hardcoded value tripping the kimi/deepseek temperature constraint)
        if temperature is None:
            # Passing None all the way to the provider = the parameter is omitted; the previous 0.1 fallback bypassed user config,
            # and gateways that accept only specific temperatures (Opus 5 accepts 1 or nothing) would answer 400 directly
            temperature = _resolve_configured_temperature(llm_provider)
        self.temperature = temperature
        self.candidates_per_draft = candidates_per_draft

    async def reconcile(
        self,
        drafts: List[CodingMemoryDraft],
        *,
        request_id: str = "",
        user_id: str = "",
        agent_id: str = "default_agent",
        session_id: str = "",
    ) -> List[ReconcileOp]:
        """
        Run reconcile + persistence. Returns the executed ReconcileOp[] (the ADD/UPDATE/DELETE/SKIP ops that succeeded).

        Execution errors (e.g. an LLM failure) fail safe to a plain ADD for every draft.

        When cache is set, a CODING_RECONCILE pipeline log is written after the LLM decision and before persistence,
        so the reconcile decision is already visible even if the store insert crashes.
        """
        if not drafts:
            return []

        # Backfill trace identity from the draft (the writer usually passes it explicitly; this is a fallback)
        if drafts:
            user_id = user_id or drafts[0].user_id
            agent_id = agent_id or drafts[0].agent_id
            session_id = session_id or (drafts[0].session_id or "")

        # 1. Retrieve candidates
        candidates_per_draft: List[List[CodingMemory]] = await self._retrieve_candidates(drafts)

        # 2. LLM decision
        _t0 = time.perf_counter()
        _decide_prompt = ""
        _decide_raw = ""
        _decide_resp = None
        try:
            ops, _decide_prompt, _decide_raw, _decide_resp = await self._llm_decide(
                drafts, candidates_per_draft
            )
        except Exception as e:
            logger.warning(f"[coding-reconcile] LLM decide failed, fallback to ADD-all: {e}")
            ops = [ReconcileOp(action="ADD", draft_idx=i) for i in range(len(drafts))]
            _decide_raw = f"[LLM error] {type(e).__name__}: {str(e)[:300]}"

        if not ops:
            # LLM returned no ops although drafts exist → fallback ADD-all
            logger.info("[coding-reconcile] LLM returned no ops, fallback to ADD-all")
            ops = [ReconcileOp(action="ADD", draft_idx=i) for i in range(len(drafts))]

        _decide_elapsed = round((time.perf_counter() - _t0) * 1000, 2)
        # CODING_RECONCILE: logged as soon as the LLM decides (before persistence, so a crash loses nothing)
        await _store_coding_log(
            self.cache, request_id=request_id, user_id=user_id, agent_id=agent_id,
            session_id=session_id, step="CODING_RECONCILE",
            prompt=_decide_prompt, response=_decide_raw,
            parsed=json.dumps({"ops": [o.to_dict() for o in ops]}, ensure_ascii=False, default=str),
            elapsed_ms=_decide_elapsed,
            prompt_tokens=getattr(_decide_resp, "prompt_tokens", 0) if _decide_resp else 0,
            completion_tokens=getattr(_decide_resp, "completion_tokens", 0) if _decide_resp else 0,
            total_tokens=getattr(_decide_resp, "tokens_used", 0) if _decide_resp else 0,
        )

        # 3. Validate + execute
        executed = await self._execute(ops, drafts)
        logger.info(
            f"[coding-reconcile] drafts={len(drafts)} ops={len(executed)} "
            + ", ".join(f"{a}={sum(1 for o in executed if o.action == a)}"
                        for a in ("ADD", "UPDATE", "DELETE", "SKIP"))
        )
        return executed

    # ------------------------------------------------------------
    # Retrieve candidates
    # ------------------------------------------------------------

    async def _retrieve_candidates(
        self, drafts: List[CodingMemoryDraft]
    ) -> List[List[CodingMemory]]:
        """For each draft, retrieve the top-N candidates of the same user via multi-path embedding over task + search_keys.

        Retrieving by task alone misses candidates that should be merged when the new draft words the same task
        differently (one thing, different wording). So search_keys are used as extra query entries, and the
        multi-path results are deduplicated by experience_id (keeping the highest score) to widen coverage and reduce missed ADDs.
        """
        result: List[List[CodingMemory]] = []
        for d in drafts:
            try:
                # task is the primary query entry, search_keys are extra entries (deduplicated, blanks dropped)
                queries = [d.task] + [k for k in (d.search_keys or []) if k and k != d.task]
                embs = await self.embed_service.embed_batch(queries)

                # Multi-path retrieval → aggregate by experience_id, keeping the highest score
                best: Dict[str, float] = {}
                for emb in embs:
                    hits = await self.store.search_by_query_embedding(
                        emb,
                        user_id=d.user_id,
                        workspace_id=d.workspace_id,
                        branch=d.branch,
                        top=self.candidates_per_draft * 4,
                        include_sources=[d.source],
                    )
                    for h in hits:
                        eid = h["experience_id"]
                        sc = h.get("score", 0.0)
                        if eid not in best or sc > best[eid]:
                            best[eid] = sc
                # Sort by score and take the top-N
                cand_ids = [eid for eid, _ in sorted(best.items(), key=lambda x: -x[1])][
                    : self.candidates_per_draft
                ]
                cands = await self.store.get_many(cand_ids, user_id=d.user_id)
                result.append(cands)
            except Exception as e:
                logger.warning(f"[coding-reconcile] retrieve candidates failed: {e}")
                result.append([])
        return result

    # ------------------------------------------------------------
    # LLM decision
    # ------------------------------------------------------------

    async def _llm_decide(
        self,
        drafts: List[CodingMemoryDraft],
        candidates_per_draft: List[List[CodingMemory]],
    ) -> Tuple[List[ReconcileOp], str, str, Any]:
        """Return (ops, prompt, raw_response, llm_resp) so the caller can write the CODING_RECONCILE log."""
        prompt, _ = build_experience_reconcile_prompt(drafts, candidates_per_draft)

        resp = await self.llm_provider.complete(
            prompt=prompt,
            max_tokens=self.max_tokens,
        )
        return self._parse_ops(resp.content), prompt, resp.content, resp

    @staticmethod
    def _format_drafts(drafts: List[CodingMemoryDraft]) -> str:
        rows = []
        for i, d in enumerate(drafts):
            rows.append(
                f"draft[{i}]:\n"
                f"  task: {d.task!r}\n"
                f"  search_keys: {d.search_keys}\n"
                f"  solution: {_truncate(d.solution, 800)!r}\n"
                f"  boundary_envs: {_truncate(d.boundary_envs, 400)!r}"
            )
        return "\n\n".join(rows) if rows else "(none)"

    @staticmethod
    def _format_candidates(
        drafts: List[CodingMemoryDraft],
        candidates_per_draft: List[List[CodingMemory]],
    ) -> str:
        rows = []
        for i, d in enumerate(drafts):
            cands = candidates_per_draft[i] if i < len(candidates_per_draft) else []
            if not cands:
                rows.append(f"for draft[{i}]: (no candidates)")
                continue
            cand_rows = []
            for c in cands:
                cand_rows.append(
                    f"  - experience_id={c.experience_id}\n"
                    f"    task: {c.task!r}\n"
                    f"    solution: {_truncate(c.solution, 600)!r}\n"
                    f"    boundary_envs: {_truncate(c.boundary_envs, 300)!r}\n"
                    f"    updated_at: {c.updated_at.isoformat() if c.updated_at else 'n/a'}"
                )
            rows.append(f"for draft[{i}]:\n" + "\n".join(cand_rows))
        return "\n\n".join(rows) if rows else "(none)"

    @staticmethod
    def _parse_ops(text: str) -> List[ReconcileOp]:
        """Parse the ops list leniently."""
        if not text:
            return []
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

        if not obj or not isinstance(obj, dict):
            logger.warning(f"[coding-reconcile] failed to parse ops output: {text!r}")
            return []
        ops_raw = obj.get("ops") or []
        if not isinstance(ops_raw, list):
            return []

        ops: List[ReconcileOp] = []
        for raw in ops_raw:
            if not isinstance(raw, dict):
                continue
            action = (raw.get("action") or "").strip().upper()
            if action not in ("ADD", "UPDATE", "DELETE", "SKIP"):
                continue
            di = raw.get("draft_idx")
            if di is not None:
                try:
                    di = int(di)
                except Exception:
                    di = None
            ops.append(ReconcileOp(
                action=action,  # type: ignore[arg-type]
                draft_idx=di,
                target_experience_id=raw.get("target_experience_id"),
                reason=raw.get("reason"),
            ))
        return ops

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    async def _execute(
        self,
        ops: List[ReconcileOp],
        drafts: List[CodingMemoryDraft],
    ) -> List[ReconcileOp]:
        """
        Validate + execute ops. Returns the ops that executed successfully (invalid ops are filtered out).

        Validation rules:
        - ADD     requires a valid draft_idx
        - UPDATE  requires a valid draft_idx + a non-empty target_experience_id
        - DELETE  requires a non-empty target_experience_id
        - SKIP    performs no store operation
        """
        executed: List[ReconcileOp] = []
        # draft_idx values already consumed by UPDATE → prevents the same draft from being ADDed/UPDATEd twice
        consumed: set = set()

        for op in ops:
            if op.action == "ADD":
                if op.draft_idx is None or not (0 <= op.draft_idx < len(drafts)):
                    logger.warning(f"[coding-reconcile] drop invalid ADD: {op.to_dict()}")
                    continue
                if op.draft_idx in consumed:
                    logger.warning(f"[coding-reconcile] drop ADD on consumed draft: {op.to_dict()}")
                    continue
                d = drafts[op.draft_idx]
                memory = self._draft_to_new_memory(d)
                try:
                    await self.store.insert(memory)
                    op.target_experience_id = memory.experience_id  # write the id back for the caller to read
                    executed.append(op)
                    consumed.add(op.draft_idx)
                except Exception as e:
                    logger.warning(f"[coding-reconcile] ADD failed: {e}")

            elif op.action == "UPDATE":
                if op.draft_idx is None or not (0 <= op.draft_idx < len(drafts)):
                    logger.warning(f"[coding-reconcile] drop invalid UPDATE: {op.to_dict()}")
                    continue
                if not op.target_experience_id:
                    logger.warning(f"[coding-reconcile] drop UPDATE missing target: {op.to_dict()}")
                    continue
                if op.draft_idx in consumed:
                    logger.warning(f"[coding-reconcile] drop UPDATE on consumed draft: {op.to_dict()}")
                    continue
                d = drafts[op.draft_idx]
                memory = self._draft_to_new_memory(d, experience_id=op.target_experience_id)
                # UPDATE overwrites the body without keeping a trace, but the fact chain is metadata and must survive (§4: UPDATE takes the union)
                prev = await self.store.get_by_id(
                    op.target_experience_id, user_id=d.user_id
                )
                if prev is not None:
                    memory.source_fact_ids = list(prev.source_fact_ids)
                try:
                    await self.store.update(memory)
                    executed.append(op)
                    consumed.add(op.draft_idx)
                except Exception as e:
                    logger.warning(f"[coding-reconcile] UPDATE failed: {e}")

            elif op.action == "DELETE":
                if not op.target_experience_id:
                    logger.warning(f"[coding-reconcile] drop DELETE missing target: {op.to_dict()}")
                    continue
                # DELETE needs to know the user_id; infer it from draft_idx (when present) or the first draft
                user_id = self._infer_user_id(op, drafts)
                if not user_id:
                    logger.warning(f"[coding-reconcile] drop DELETE no user_id context: {op.to_dict()}")
                    continue
                try:
                    ok = await self.store.delete(op.target_experience_id, user_id=user_id)
                    if ok:
                        executed.append(op)
                except Exception as e:
                    logger.warning(f"[coding-reconcile] DELETE failed: {e}")

            elif op.action == "SKIP":
                if op.draft_idx is not None and 0 <= op.draft_idx < len(drafts):
                    consumed.add(op.draft_idx)
                executed.append(op)

        # Fallback: every draft must be handled least once
        for i in range(len(drafts)):
            if i in consumed:
                continue
            logger.info(f"[coding-reconcile] draft[{i}] not handled, fallback ADD")
            d = drafts[i]
            memory = self._draft_to_new_memory(d)
            try:
                await self.store.insert(memory)
                executed.append(ReconcileOp(
                    action="ADD", draft_idx=i,
                    target_experience_id=memory.experience_id,
                    reason="fallback ADD (no LLM op)",
                ))
            except Exception as e:
                logger.warning(f"[coding-reconcile] fallback ADD failed: {e}")

        return executed

    @staticmethod
    def _infer_user_id(
        op: ReconcileOp, drafts: List[CodingMemoryDraft]
    ) -> Optional[str]:
        if op.draft_idx is not None and 0 <= op.draft_idx < len(drafts):
            return drafts[op.draft_idx].user_id
        # Fallback: use the first non-empty user_id in drafts (one reconcile call is usually one user)
        for d in drafts:
            if d.user_id:
                return d.user_id
        return None

    @staticmethod
    def _draft_to_new_memory(
        d: CodingMemoryDraft,
        *,
        experience_id: Optional[str] = None,
    ) -> CodingMemory:
        return CodingMemory(
            experience_id=experience_id or str(uuid.uuid4()),
            user_id=d.user_id,
            agent_id=d.agent_id,
            task=d.task,
            search_keys=list(d.search_keys),
            solution=d.solution,
            boundary_envs=d.boundary_envs,
            workspace_id=d.workspace_id,
            branch=d.branch,
            session_id=d.session_id,
            files=list(d.files),
            confidence=d.confidence,
            source=d.source,
            extraction_source=d.extraction_source,
        )


# ================================================================
# Helpers
# ================================================================

def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."
