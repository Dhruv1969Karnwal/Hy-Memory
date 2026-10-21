# -*- coding: utf-8 -*-
"""
CodingWriter - write-side coordinator

Chains judge / preproc / extractor / the two reconcilers / store into one end-to-end flow:

    classify_messages_scene(messages, llm)
        ├── chat → the caller applies strip_tool_messages to messages and uses the chat pipeline
        └── working / coding
             ├── recent history: the last K user-turns (context only)
             ├── new messages: every turn of this call (per-message length cap only)
             ├── extractor.extract(...)            ← one LLM call producing drafts for both layers
             ├── reconciler.reconcile(experience drafts) ← ADD/UPDATE/DELETE/SKIP
             ├── fact_reconciler.reconcile(fact drafts)  ← INSERT/SUPERSEDE
             └── _crosslink(...)                   ← full cross-linking across the batch
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..agent.llm_provider import LLMProvider
from ..core.embed_service import EmbedService
from ..pipelines.base import ChatMessage
from .extractor import CodingMemoryExtractor
from .fact_reconciler import WorkingFactReconciler
from .judge import classify_messages_is_coding
from .preproc import has_any_tool_message, prepare_extractor_messages
from .reconciler import CodingMemoryReconciler
from .store import CodingMemoryStore
from .types import FactReconcileOp, ReconcileOp, WorkingFact

logger = logging.getLogger(__name__)


class CodingWriter:
    """
    Coding write-side coordinator.

    Lifecycle: store / extractor / reconciler are injected from outside; it does not own initialize/close.
    """

    def __init__(
        self,
        store: CodingMemoryStore,
        extractor: CodingMemoryExtractor,
        reconciler: CodingMemoryReconciler,
        llm_provider: LLMProvider,
        embed_service: EmbedService,
        fact_reconciler: Optional[WorkingFactReconciler] = None,
        cache: Any = None,
    ):
        self.store = store
        self.extractor = extractor
        self.reconciler = reconciler
        self.llm_provider = llm_provider
        self.embed_service = embed_service
        self.fact_reconciler = fact_reconciler or WorkingFactReconciler(
            store, embed_service, llm_provider
        )
        # The cache only carries request_id through the stages (each stage traces itself); the writer does not trace centrally.
        self.cache = cache

    async def write(
        self,
        messages: List[ChatMessage],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: str = "default",
    ) -> Dict[str, Any]:
        """
        End-to-end write for a working/coding segment. Returns a response dict (compatible with the client.add shape):
            {
                "success": True,
                "scene": "productivity",
                "request_id": "...",
                "elapsed_ms": ...,
                "ops": [
                    {"action": "ADD", "draft_idx": 0, "target_experience_id": "...", "reason": ...},
                    ...
                ],
                "experience_ids": ["..."],   # every experience_id touched by ADD/UPDATE
                "fact_ops": [
                    {"action": "INSERT", "draft_idx": 0, "fact_id": "...", ...},
                    ...
                ],
                "fact_ids": ["..."],         # the fact_ids this run actually persisted
            }
        """
        request_id = request_id or str(uuid.uuid4())
        t0 = time.perf_counter()

        if not messages:
            return self._empty_response(request_id, t0)

        tool_max_bytes = int(
            getattr(self.store.config.coding, "tool_result_max_bytes", 2048) or 2048
        )

        # 1. Separate history from the new input.
        #    - history_messages: the last 5 user-turns of earlier writes, context only
        #    - current_messages: every turn of this write, capped per message size only, never trimmed by turn count
        history_turns = int(
            getattr(self.store.config.extractor, "history_turns", 5) or 5
        )
        try:
            history_messages = await self.store.get_recent_trace_messages(
                user_id=user_id,
                agent_id=agent_id or "default_agent",
                max_user_turns=history_turns,
                source=source,
            )
        except Exception as e:
            logger.warning(f"[coding-write] recent trace lookup failed, continue without history: {e}")
            history_messages = []
        current_messages = prepare_extractor_messages(
            messages, tool_max_bytes=tool_max_bytes,
        )

        # 2. Extract the two-layer drafts (one LLM call)
        #
        # Deliberately hide the user's existing tasks from the extractor. Extraction should faithfully read what happened in
        # this segment; how it relates to what is already stored is reconcile's job. Shown the existing tasks, the extractor
        # bends a genuinely different task towards one of those wordings, the reconciler then judges it an UPDATE and merges
        # it away, and the difference is gone -- mismatched wording costs at most one extra row, biased wording writes it wrong.
        try:
            drafts, fact_drafts = await self.extractor.extract(
                messages,
                history_messages=history_messages,
                tool_result_max_bytes=tool_max_bytes,
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                branch=branch,
                session_id=session_id,
                request_id=request_id,
            )
        finally:
            # History serves the next write; context must survive even when this call crosses no value threshold.
            try:
                await self.store.append_trace(
                    current_messages,
                    request_id=request_id,
                    user_id=user_id,
                    agent_id=agent_id or "default_agent",
                    session_id=session_id,
                    workspace_id=workspace_id,
                    branch=branch,
                    source=source,
                )
            except Exception as e:
                logger.warning(f"[coding-write] append trace failed: {e}")

        if not drafts and not fact_drafts:
            logger.info(
                f"[coding-write] no drafts produced (value bar); request_id={request_id}"
            )
            return self._empty_response(request_id, t0)

        # The extractor only produces content; the call source is injected by the SDK entry point and must not be LLM-generated.
        for draft in drafts:
            draft.source = source

        # 3. Experience-layer reconcile + persistence
        ops: List[ReconcileOp] = (
            await self.reconciler.reconcile(
                drafts,
                request_id=request_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id or "",
            )
            if drafts else []
        )

        # 4. Fact-layer reconcile + persistence (independent path, unaffected by experience-layer results)
        facts: List[WorkingFact] = []
        fact_ops: List[FactReconcileOp] = []
        if fact_drafts:
            facts, fact_ops = await self.fact_reconciler.reconcile(
                fact_drafts,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                workspace_id=workspace_id,
                branch=branch,
                source=source,
            )

        experience_ids = [op.target_experience_id for op in ops
                          if op.action in ("ADD", "UPDATE") and op.target_experience_id]
        fact_ids = [f.fact_id for f in facts]

        # 5. Full cross-linking within the batch
        await self._crosslink(experience_ids, facts, user_id=user_id)

        add_cnt = sum(1 for op in ops if op.action == "ADD")
        update_cnt = sum(1 for op in ops if op.action == "UPDATE")
        delete_cnt = sum(1 for op in ops if op.action == "DELETE")
        skip_cnt = sum(1 for op in ops if op.action == "SKIP")
        ops_summary = {
            "add": add_cnt,
            "update": update_cnt,
            "delete": delete_cnt,
            "skip": skip_cnt,
            "total": add_cnt + update_cnt + delete_cnt + skip_cnt,
        }
        insert_cnt = sum(1 for op in fact_ops if op.action == "INSERT")
        supersede_cnt = sum(1 for op in fact_ops if op.action == "SUPERSEDE")

        return {
            "success": True,
            "scene": "productivity",
            "request_id": request_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "ops": [op.to_dict() for op in ops],
            "ops_summary": ops_summary,
            "experience_ids": experience_ids,
            "fact_ops": [op.to_dict() for op in fact_ops],
            "fact_ops_summary": {
                "insert": insert_cnt,
                "supersede": supersede_cnt,
                "total": insert_cnt + supersede_cnt,
            },
            "fact_ids": fact_ids,
        }

    # ------------------------------------------------------------
    # Cross-linking
    # ------------------------------------------------------------

    async def _crosslink(
        self,
        experience_ids: List[str],
        facts: List[WorkingFact],
        *,
        user_id: str,
    ) -> None:
        """Link the experiences and facts persisted in the same batch to each other.

        Only real writes are linked: experiences take the ADD/UPDATE ids, facts take the ones actually INSERTed; SKIP and
        DELETE are naturally absent from the list. Both sides take a union -- the older chain of an UPDATE must survive, since the
        fact side may already have adopted a batch of experiences through SUPERSEDE, which must not be overwritten.
        """
        if not experience_ids and not facts:
            return
        fact_ids = [f.fact_id for f in facts]

        # Forward: experience ← fact
        if fact_ids:
            for eid in experience_ids:
                try:
                    exp = await self.store.get_by_id(eid, user_id=user_id)
                    if exp is None:
                        continue
                    merged = _union(exp.source_fact_ids, fact_ids)
                    if merged != list(exp.source_fact_ids):
                        await self.store.set_experience_source_facts(eid, merged)
                except Exception as e:
                    logger.warning(f"[coding-write] crosslink experience {eid} failed: {e}")

        # Reverse: fact ← experience
        if experience_ids:
            for f in facts:
                try:
                    merged = _union(f.derived_experience_ids, experience_ids)
                    if merged != list(f.derived_experience_ids):
                        await self.store.set_fact_derived_experiences(f.fact_id, merged)
                        f.derived_experience_ids = merged
                except Exception as e:
                    logger.warning(
                        f"[coding-write] crosslink fact {f.fact_id} failed: {e}"
                    )

    @staticmethod
    def _empty_response(request_id: str, t0: float) -> Dict[str, Any]:
        return {
            "success": True,
            "scene": "productivity",
            "request_id": request_id,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "ops": [],
            "ops_summary": {"add": 0, "update": 0, "delete": 0, "skip": 0, "total": 0},
            "experience_ids": [],
            "fact_ops": [],
            "fact_ops_summary": {"insert": 0, "supersede": 0, "total": 0},
            "fact_ids": [],
        }


def _union(base: List[str], extra: List[str]) -> List[str]:
    """Order-preserving union."""
    out = list(base)
    seen = set(out)
    for x in extra:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ================================================================
# Top-level helper (used by client.async_add)
# ================================================================

async def is_coding_segment(
    messages: List[ChatMessage],
    llm_provider: LLMProvider,
) -> bool:
    """
    Decide whether the whole message list is a coding scenario. Without tool messages this is False immediately (O(1) short circuit).
    """
    if not has_any_tool_message(messages):
        return False
    return await classify_messages_is_coding(messages, llm_provider)
