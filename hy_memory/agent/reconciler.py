"""
Agent Memory - MemoryReconciler

In the Lite pipeline, after the LLM extracts new facts/identity and before they are persisted,
they are reconciled against the existing memories in the vector database:

Flow:
  1. (Optional) Pre-process: the LLM generates search queries (MEMORY_ENABLE_SEARCH_QUERY=true, off by default)
  2. Candidate search: by default every new memory runs fused vector + BM25 retrieval
     (enlarged vector pool + separate keyword channel + additive(sem/bm25/max(tag,entity)), min_score=0.4);
     when search queries are enabled, the rewritten queries get an extra vector search
  3. One LLM call (Reconcile) taking the new memories + the candidate old memories as input
  4. Returns the list of operation instructions (ADD / DELETE, executed in order)
  5. The caller performs the persistence operations per instruction and updates the old node statuses

Operation schema (two ops freely combined to express three semantics):
  ADD    → insert a new node
           supersedes=[]         → case A: entirely new information (unrelated to any old memory)
           supersedes=[id, ...]  → case B: contradiction/supersede (the old node no longer holds and joins the evolution chain)
  DELETE → logically delete the old node (status=SHADOW, is_latest=False, not added to the evolution chain)
           case C: merge absorption -- used together with ADD(supersedes=[]) to express
                   "this ADD absorbed the content of some old memories, which are no longer retrieved"

  Examples:
    Case B contradiction: ADD(content="drinks tea", supersedes=[id_coffee])
    Case C merge: ADD(content="has PI heritage and applies it in music", supersedes=[]) + DELETE(id_heritage)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .llm_provider import (
    LLMProvider,
    _content_fingerprint,
    _log_llm_content_enabled,
)
from ..config import MemoryConfig

logger = logging.getLogger(__name__)


def is_canonical_state_node(node: Any) -> bool:
    """Canonical state projections are owned by System2's exact-key lifecycle."""
    return (getattr(node, "custom", None) or {}).get("memory_kind") == "canonical_state"


def _log_ops_detail(ops: List["ReconcileOp"]) -> None:
    """Log the per-op reconcile results as one INFO line (JSON list) to ease online debugging."""
    if not ops:
        return
    view = [
        {
            "op": o.op,
            "memory_id": o.memory_id or None,
            "content": o.content,
            "tags": list(o.tags or []),
            "topic_paths": list(o.topic_paths or []),
            "event_facet": dict(o.event_facet or {}),
            "topic_assignments": list(o.topic_assignments or []),
            "structure_references": list(o.structure_references or []),
            "related": list(o.related or []),
            "reason": o.supersede_reason or o.reason or "",
        }
        for o in ops
    ]
    logger.info(
        f"[reconciler] ops detail: {json.dumps(view, ensure_ascii=False, default=str)}"
    )


# ================================================================
# Operation instructions
# ================================================================


@dataclass
class ReconcileOp:
    """
    A single operation instruction (v2 flat format).

    op = "ADD": a pure new node, entirely new information that references no existing memory.

    op = "SUPERSEDE": state evolution / contradiction chain. The new memory replaces the old one and becomes the chain head.
      - memory_id points at the superseded old node
      - content describes only the new or changed information (the old node stays in the chain automatically)
      - supersede_reason describes why the statements conflict
      - execution: old node → SHADOW + is_latest=False; new node supersedes=[old_id] (joins the evolution chain)

    op = "UPDATE": merge and refine. The new and old memory describe the same topic without contradicting each other, so they merge into one dense node.
      - memory_id points at the old node being updated
      - content is the complete merged text
      - execution: old node → SHADOW + is_latest=False; the new node is created on its own (no evolution chain, no supersedes)
      - the old node is kept in the VDB as history (status=SHADOW) but forms no chain relation
    """

    op: str = "ADD"
    # --- Shared fields ---
    content: Optional[str] = None  # Final stored content
    layer: Optional[str] = None  # "L2_FACT" / "L4_IDENTITY"
    owner: Optional[str] = None  # 'user' / 'agent' (only meaningful for L2_FACT)
    tags: List[str] = field(default_factory=list)  # 1-3 keywords; proper nouns keep their capitalization
    tag_ners: List[str] = field(default_factory=list)  # NER labels aligned one-to-one with tags
    # Explicit structure at write time. topic_paths go from broad to narrow; event_facet only describes factual semantics
    # and never carries the question type or routing hint of a future query.
    topic_paths: List[List[str]] = field(default_factory=list)
    event_facet: Dict[str, Any] = field(default_factory=dict)
    topic_assignments: List[Dict[str, Any]] = field(default_factory=list)
    structure_references: List[Dict[str, str]] = field(default_factory=list)
    topic_paths_provided: bool = False
    event_facet_provided: bool = False
    topic_assignments_provided: bool = False
    speculate: Optional[str] = None  # Implicit preference inference (identity only)
    # --- SUPERSEDE / UPDATE fields ---
    memory_id: Optional[str] = None  # Target old node ID (single node; for a multi-node chain the newest one on it)
    memory_ids: List[str] = field(default_factory=list)  # SUPERSEDE multi-node chain: ordered (oldest→newest) existing node IDs; folds several old memories on the same dimension into one evolution chain
    supersede_reason: str = ""  # Contradiction description for SUPERSEDE
    # --- Relation graph (pro mode, see docs/pro_graph_design.md) ---
    # LLM-annotated relations to existing memories: [{"target_id": <existing memory_id>, "relation_type": ...}]
    # relation_type ∈ causes / elaborates / contrasts / same_series.
    # reconcile already lays out existing and new memories together, so emitting one more field only costs output tokens
    # and no extra LLM call; besides, the endpoints of causes / contrasts edges usually have very low cosine similarity, so vector retrieval would never surface them.
    related: List[Dict[str, str]] = field(default_factory=list)
    # --- Common ---
    reason: str = ""


@dataclass
class ReconcileResult:
    """Reconcile result"""

    ops: List[ReconcileOp] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    # True = reconcile failed and the "ADD everything" fallback was used (see _fallback_add_ops).
    # The result is still usable (success=True), but no dedup or evolution decision ran, so duplicates may be introduced.
    degraded: bool = False
    degraded_reason: Optional[str] = None
    # LLM token consumption of the reconcile pipeline (sum of the search-query and reconcile LLM calls)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Similarity snapshot of the candidate pool: {new memory content: [(existing_node_id, score), ...]}
    #
    # reconcile already runs hybrid retrieval for every new memory and computes fused scores, then discards them. Storing them
    # as `semantic_neighbor` edges on the write side costs **zero extra computation** (see docs/pro_graph_design.md §2.2).
    # Keyed by content instead of node_id: the new memories are not persisted yet and have no id.
    candidate_scores: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    # reconcile candidate retrieval already embedded every new memory. Handing the vectors of the original texts back to the
    # writer lets ADD-all / degraded ADD persist directly, avoiding a second embedding request for the same fact.
    # Reuse only on exactly identical content; LLM-rewritten UPDATE/SUPERSEDE content is recomputed to prevent mismatches.
    new_memory_embeddings: Dict[str, List[float]] = field(default_factory=dict)


# ================================================================
# Prompts
# ================================================================

SEARCH_QUERY_PROMPT = """You are a search query generator. Given a list of newly extracted memories, generate a set of short search queries that can be used to find related existing memories in a vector database.

The goal is to maximize recall — find existing memories that are semantically related to the new memories, even if the wording is very different.

## New memories:
{new_memories}

## Instructions

Generate search queries that cover:
- Key topics, entities, and themes mentioned in the memories
- Rephrased or abstracted versions of the core concepts
- Related concepts that might exist in the user's memory store

Output a JSON array of query strings (5-15 queries, short and focused):

["query1", "query2", "query3", ...]

Output JSON array only, no other text."""

RECONCILE_PROMPT = """You are a memory management system.

Your task is to merge new memories into an existing memory base while keeping it:

* lossless
* non-contradictory
* compact
* highly retrievable

---

# Operations

## ADD

Use when the information is entirely new.

```json
{{
  "op": "ADD",
  "content": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

---

## UPDATE

Use when:

* the new memory is compatible with an existing one
* they describe the same topic
* merging improves completeness or retrievability

Rules:

* merge losslessly
* preserve all useful information
* do NOT create duplicates

### Same occurrence vs. another occurrence

Use **UPDATE** only when the new memory adds detail to the same real-world
occurrence or durable fact already recorded.

If the same person does the same kind of activity/event again at another time
(another trip, game, injury, purchase, health scare, etc.), keep the occurrence
as a separate **ADD**. Preserve its own date and details. When an existing memory
is clearly another occurrence in that same concrete series, link it with
`related: [{{"memory_id":"...","relation":"same_series"}}]`.

Do not merge separate dated occurrences into one long count/timeline memory.
The retrieval layer builds timelines from the linked L2 memories.

```json
{{
  "op": "UPDATE",
  "memory_id": "...",
  "content": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

---

## SUPERSEDE

Use ONLY when:

* the new memory contradicts an existing memory
* both cannot be true at the same time
* the old memory would now give the wrong current-state answer

Examples:

* city changed
* employer changed
* relationship status changed

Do NOT use for:

* added detail
* compatible preferences
* wording differences
* semantic equivalence

Rules:

* content must contain ONLY the new state
* do NOT copy unchanged information
* the system keeps the old node and links it onto the evolution chain automatically; you don't need to handle history

```json
{{
  "op": "SUPERSEDE",
  "memory_id": "...",
  "content": "...",
  "supersede_reason": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

### Folding several existing memories into one chain

If TWO OR MORE existing memories track the SAME dimension and the new memory is
the latest change on that dimension, you may fold them into a single evolution
chain. Use `memory_ids` (an ordered list, OLDEST → NEWEST) instead of a single
`memory_id`:

```json
{{
  "op": "SUPERSEDE",
  "memory_ids": ["<oldest_id>", "<newer_id>"],
  "content": "<only the newest state>",
  "supersede_reason": "...",
  "owner": "user",
  "tags": ["..."]
}}
```

The listed memories are chained oldest → newest and the new memory becomes the
chain head. `content` still describes ONLY the newest state — never copy the
older memories' text into it; they remain queryable in the chain history.

---

# Rules

* Preserve all meaningful information.
* Skip exact duplicates.
* Prefer merging related fragments.
* Separate dated occurrences stay separate ADDs; use `same_series` only for the
  same subject and same concrete activity/event type.
* Keep each memory focused on one coherent topic.
* Do not invent chronology.
* `memory_at: null` means unknown time.
* **NEVER merge or supersede across different subjects.** The subject is whoever
  the content is about: "The user ..." vs "John ..." vs "Mary ...". Memories about
  different people are independent facts even when the sentence is otherwise
  identical — "The user likes coffee" and "John likes coffee" are two separate
  memories, never an UPDATE or a SUPERSEDE of each other.
* `owner` (`user`/`agent`): copy it from the memory being written. NEVER merge or supersede across different owners — a user's fact and the assistant's fact are distinct even on the same topic.
* A SUPERSEDE either targets ONE memory (`memory_id`) or folds several memories on the same dimension into one chain (`memory_ids`, oldest→newest). An UPDATE targets exactly ONE memory.
* Never touch the same existing memory from more than one op.
* Output language matches input memory language.

---

# Optional: `related`

On any op you MAY add a `related` array recording how the memory being written
relates to an EXISTING memory. This does not change what gets stored — it only
records a link.

```json
{{
  "op": "ADD",
  "content": "...",
  "owner": "user",
  "tags": ["..."],
  "related": [{{"memory_id": "<existing_id>", "relation": "causes"}}]
}}
```

Only these four relations are allowed:

* `causes` — the two memories are in a cause/effect relationship
  (e.g. "started running daily" → "knee started hurting")
* `elaborates` — adds detail to, or is a later development of, the existing memory
* `contrasts` — they are in tension but BOTH still true (if one makes the other
  false, that is a SUPERSEDE, not a `contrasts` link)
* `same_series` — separate dated occurrences of the same concrete activity or
  event type for the same subject (for example, two different skiing trips)

Rules:

* Only link to memories that appear in "Existing memories" — never invent an id.
* Only emit a link when the relationship is **explicit and specific**. Two memories
  merely being about the same broad topic is NOT a relation — leave `related` out.
* `same_series` requires the same subject and a concrete repeated action/event;
  shared broad tags such as health, travel, sport, food, or work are insufficient.
* Omit the field entirely when there is nothing to link. Zero links is the normal case.
* Never use `related` as a substitute for UPDATE / SUPERSEDE.

---
{few_shot_section}
###################
Input Data
###################

Current date: {current_date}

## Existing memories

{existing_memories}

## New memories

{new_memories}

---

# Output

Return ONLY a JSON array inside a fenced ```json code block.

If no changes are needed:

```json
[]
```

Now produce the JSON array."""

RECONCILE_STRUCTURE_METADATA_PROMPT_EN = """

Additional structural output contract:

For every ADD, UPDATE, or SUPERSEDE result, preserve or refine `topic_paths`,
`event_facet`, and `topic_assignments`. `event_facet` contains only `subject`,
`participants`, `action`, `objects`, `locations`, and `state`.
`topic_assignments` contains subject-specific `{subject, role, path}` entries;
each path segment is `{name, type}` and type is `topic|facet|series|episode`.
Types follow `topic -> facet -> series -> episode`; levels may be skipped but
types may not repeat or move backward.
Use Series only for repeated occurrences of one concrete activity/state line and
Episode only for one bounded real-world occurrence. Never infer these types from
depth, a broad shared tag, or unrelated co-occurrence. When an Existing memory
already has a semantically equivalent subject-specific path, reuse its exact node
names and types instead of creating a synonym fork. To request exact reuse, emit
`structure_references: [{"memory_id":"<existing_id>", "subject":"...",
"through":"topic|facet|series|episode"}]`. The referenced ID must appear in
Existing memories. `through` means copy that existing path prefix through the
named structural type; an explicitly authored deeper suffix remains new. Reuse
is forbidden when the subject, concrete activity/state line, or Episode boundary
differs. These fields describe the memory, never a future query form such as
"when", "how many", or "who".

Series is a reconcile-time relation: establish it only when the new memory and
at least one independent Existing memory are distinct occurrences of the same
concrete activity/state line. Multiple propositions from the current write do
not prove repetition.
"""


def append_reconcile_structure_prompt(
    prompt: str, *, enabled: bool, language: str,
) -> str:
    """Append the structure contract only for an explicit experiment config."""
    if not enabled:
        return prompt
    if language == "zh":
        from .prompts_zh import RECONCILE_STRUCTURE_METADATA_PROMPT_ZH
        return prompt + RECONCILE_STRUCTURE_METADATA_PROMPT_ZH
    return prompt + RECONCILE_STRUCTURE_METADATA_PROMPT_EN


def build_reconcile_prompt(
    *,
    new_memories: List[str],
    new_memories_block: str,
    existing_memories_block: str,
    current_date: str,
    few_shot_enabled: bool = False,
    structure_metadata_enabled: bool = False,
) -> str:
    """Build the language-matched reconcile prompt used by the write path."""
    from ..utils.lang_detect import is_chinese

    if is_chinese("\n".join(new_memories)):
        from .prompts_zh import RECONCILE_FEW_SHOT_ZH, RECONCILE_PROMPT_ZH

        few_shot = ("\n" + RECONCILE_FEW_SHOT_ZH + "\n") if few_shot_enabled else ""
        prompt = RECONCILE_PROMPT_ZH.format(
            current_date=current_date,
            new_memories=new_memories_block,
            existing_memories=existing_memories_block,
            few_shot_section=few_shot,
        )
        return append_reconcile_structure_prompt(
            prompt,
            enabled=structure_metadata_enabled,
            language="zh",
        )

    few_shot = ("\n" + RECONCILE_FEW_SHOT_EN + "\n") if few_shot_enabled else ""
    prompt = RECONCILE_PROMPT.format(
        current_date=current_date,
        new_memories=new_memories_block,
        existing_memories=existing_memories_block,
        few_shot_section=few_shot,
    )
    return append_reconcile_structure_prompt(
        prompt,
        enabled=structure_metadata_enabled,
        language="en",
    )


# Reconcile few-shot examples (injected into {few_shot_section} only when
# few_shot_enabled). Each shows the existing/new input and the expected ops,
# with a one-line takeaway. IDs are shortened for readability.
RECONCILE_FEW_SHOT_EN = """
###################
Examples
###################

## Example 1 — One new memory: part conflicts with an existing chain, part is unrelated

Existing memories:
[
  {"memory_id": "m1", "content": "The user works as a product manager at Stripe.", "owner": "user", "history_versions": [{"content": "The user works as a product manager at a fintech startup."}]}
]
New memories:
[
  {"content": "The user has left Stripe and is now a product lead at Notion, and has recently taken up pottery on weekends.", "owner": "user"}
]

Output:
```json
[
  {"op": "SUPERSEDE", "memory_id": "m1", "content": "The user works as a product lead at Notion.", "supersede_reason": "employer changed from Stripe to Notion", "owner": "user", "tags": ["work"]},
  {"op": "ADD", "content": "The user has recently taken up pottery on weekends.", "owner": "user", "tags": ["hobby"]}
]
```
A SINGLE new memory carries two things: the job change (same dimension as the existing employment chain m1, which already has history → SUPERSEDE, becomes the new chain head) and an unrelated detail (pottery → ADD). Split the one memory into the right op per part — the new head describes ONLY the employment state, never the pottery.

## Example 2 — Fold two unlinked existing memories + the new change into one 3-node chain

Existing memories:
[
  {"memory_id": "a1", "content": "The user is learning to play the guitar.", "owner": "user", "memory_at": "2024-02-10 09:00"},
  {"memory_id": "a2", "content": "The user can now play a few full songs on the guitar.", "owner": "user", "memory_at": "2024-09-05 09:00"}
]
New memories:
[
  {"content": "The user now performs guitar at local open-mic nights.", "owner": "user"}
]

Output:
```json
[
  {"op": "SUPERSEDE", "memory_ids": ["a1", "a2"], "content": "The user performs guitar at local open-mic nights.", "supersede_reason": "guitar skill progressed: learning → playing songs → performing live", "owner": "user", "tags": ["music", "hobby"]}
]
```
a1 and a2 were never linked, but they track the same dimension (guitar progress) and the new memory is the latest step. Fold them with `memory_ids` ordered oldest→newest; they become a 3-node chain (a1 ← a2 ← new head). The head `content` describes ONLY the current state — it does not restate "learning" or "a few songs".

## Example 3 — New head must NOT copy the old memory's content

Existing memories:
[
  {"memory_id": "c1", "content": "The user lives in a rented one-bedroom apartment in Shibuya, Tokyo, pays ¥180,000/month, loves the 5-minute walk to the station, but complains the place is noisy at night and has no parking.", "owner": "user"}
]
New memories:
[
  {"content": "The user just bought and moved into a three-bedroom house in the suburbs of Yokohama with a garden and a two-car garage.", "owner": "user"}
]

Output:
```json
[
  {"op": "SUPERSEDE", "memory_id": "c1", "content": "The user bought and moved into a three-bedroom house in suburban Yokohama, with a garden and a two-car garage.", "supersede_reason": "residence changed from a rented Shibuya apartment to an owned Yokohama house", "owner": "user", "tags": ["housing"]}
]
```
The new memory SUPERSEDEs c1 and becomes the chain head. The head `content` describes ONLY the new Yokohama house. Do NOT drag the old apartment's distinctive details (Shibuya, ¥180,000 rent, the station walk, the noise/parking complaints) into the new node — those clearly belong to the old residence, not the new one. They stay in chain history and remain retrievable; copying them would make the head self-contradictory (renting in Shibuya AND owning in Yokohama at once).

## Example 4 — Different subjects are never merged, however similar the wording

Existing memories:
[
  {"memory_id": "d1", "content": "The user drinks black coffee every morning.", "owner": "user"}
]
New memories:
[
  {"content": "John drinks black coffee every morning.", "owner": "user"},
  {"content": "The user now drinks black coffee twice a day, morning and after lunch.", "owner": "user"}
]

Output:
```json
[
  {"op": "ADD", "content": "John drinks black coffee every morning.", "owner": "user", "tags": ["food"]},
  {"op": "UPDATE", "memory_id": "d1", "content": "The user drinks black coffee twice a day, in the morning and after lunch.", "owner": "user", "tags": ["food"]}
]
```
The two new memories are near-identical in wording but have DIFFERENT subjects. "John ..." is about another person, so it is a plain ADD — never an UPDATE or SUPERSEDE of d1, even though d1 says almost the same thing about the user. Only the memory whose subject is also "The user" may merge with d1.

## Example 5 — Recurring event: bump count and keep a timeline (UPDATE, not ADD)

Existing memories:
[
  {"memory_id": "i1", "content": "John injured his ankle last season and needed time off plus physical therapy.", "owner": "user", "memory_at": "2023-05-21 19:48"}
]
New memories:
[
  {"content": "John twisted his ankle again during a recent game.", "owner": "user", "memory_at": "2023-10-13 13:50"}
]

Output:
```json
[
  {"op": "UPDATE", "memory_id": "i1", "content": "John has injured his ankle two times: first last season (around the 2022–2023 season), needing time off and physical therapy; second during a game around 2023-10-13.", "owner": "user", "tags": ["health", "basketball", "injury"]}
]
```
Same event type happening again → UPDATE the old memory. State the **count** and list occurrences with times when known. Do NOT ADD a second "John injured his ankle" fact, and do NOT SUPERSEDE (the first injury did not stop being true).
"""


# ================================================================
# MemoryReconciler
# ================================================================


class MemoryReconciler:
    """
    Memory reconciler

    In the lite pipeline, after new facts/identity are extracted,
    it reconciles them against the existing memories in the vector database and then decides the persistence operations.
    """

    # Candidates kept per new memory after hybrid fusion
    SEARCH_TOPK = 20
    # Hybrid vector pool enlargement factor (limit × 3)
    VEC_POOL_MULTIPLIER = 3
    # Fused score threshold (additive; sem is not discounted when BM25 is absent)
    HYBRID_MIN_SCORE = 0.4
    # Vector topk for the rewritten search-query path (only with MEMORY_ENABLE_SEARCH_QUERY=true)
    QUERY_SEARCH_TOPK = 20
    SEARCH_THRESHOLD = 0.3
    # Candidates finally kept after chain merging (per path)
    FINAL_TOPK = 10
    # max_tokens for the reconcile LLM call (base value + dynamic expansion)
    # Each ADD emits roughly 200-300 tokens (content + supersedes + reason)
    # Hard ceiling of 8000 tokens so a single call cannot run away
    MAX_TOKENS = 2000
    PER_OP_TOKENS = 300
    BASE_OVERHEAD_TOKENS = 500
    MAX_TOKENS_UPPER_BOUND = 8000

    def __init__(
        self,
        config: MemoryConfig,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.config = config
        self._llm: Optional[LLMProvider] = llm_provider
        # Ablation switch: when off, only the new memories are searched directly and no search queries are generated
        self._enable_search_query = self._read_enable_search_query()

    @staticmethod
    def _read_enable_search_query() -> bool:
        v = os.getenv("MEMORY_ENABLE_SEARCH_QUERY", "false").lower().strip()
        return v not in ("false", "0", "no", "off")

    def _get_llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = LLMProvider(self.config)
        return self._llm

    async def _generate_search_queries(self, new_memories: List[str]) -> tuple:
        """
        Pre-process: derive search queries from all new memories,
        used to strengthen the recall of vector search.

        Returns:
            (queries: List[str], llm_response: Optional[LLMResponse])
        """
        mem_lines = "\n".join(f"{i+1}. {m}" for i, m in enumerate(new_memories))

        # Select prompt based on input language
        from ..utils.lang_detect import is_chinese

        if is_chinese("\n".join(new_memories)):
            from .prompts_zh import SEARCH_QUERY_PROMPT_ZH

            prompt = SEARCH_QUERY_PROMPT_ZH.format(new_memories=mem_lines)
        else:
            prompt = SEARCH_QUERY_PROMPT.format(new_memories=mem_lines)

        try:
            llm = self._get_llm()
            response = await llm.complete(
                prompt=prompt,
                max_tokens=500,
            )
            text = response.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            queries = json.loads(text)
            if isinstance(queries, list):
                queries = [q for q in queries if isinstance(q, str) and q.strip()]
                logger.debug(
                    f"[reconciler] generated {len(queries)} search queries: {queries}"
                )
                return queries, response
        except Exception as e:
            logger.warning(f"[reconciler] search query generation failed: {e}")

        return [], None

    @staticmethod
    def _fallback_add_ops(
        new_memories: List[str],
        new_memories_with_meta: Optional[List[Dict]] = None,
    ) -> List[ReconcileOp]:
        """Fallback when reconcile fails completely: **ADD the extraction results verbatim**.

        Why this fallback is mandatory: reconcile only decides how new facts and old memories combine. It dying (gateway 500,
        unparseable JSON) does not mean the extraction is worthless -- the previous behaviour discarded an entire batch of intact facts,
        and production measurements lost whole blocks of content that way (including the correct answers of evaluation questions). Duplicates beat lost content:
        duplicates can still be cleaned up by later dedup / reconcile, while lost content is gone for good.
        """
        metas = new_memories_with_meta or []
        by_content = {
            (m.get("content") or ""): m for m in metas if isinstance(m, dict)
        }
        ops: List[ReconcileOp] = []
        for text in new_memories:
            if not text:
                continue
            meta = by_content.get(text) or {}
            ops.append(ReconcileOp(
                op="ADD",
                content=text,
                layer=meta.get("layer"),
                tags=meta.get("tags") or [],
                owner=meta.get("owner"),
                topic_paths=meta.get("topic_paths") or [],
                event_facet=meta.get("event_facet") or {},
                topic_assignments=meta.get("topic_assignments") or [],
                topic_paths_provided=bool(
                    meta.get("topic_paths_provided", "topic_paths" in meta)
                ),
                event_facet_provided=bool(
                    meta.get("event_facet_provided", "event_facet" in meta)
                ),
                topic_assignments_provided=bool(meta.get(
                    "topic_assignments_provided", "topic_assignments" in meta,
                )),
                reason="reconcile unavailable → stored as-is (degraded)",
            ))
        return ops

    async def reconcile(
        self,
        new_memories: List[str],
        user_id: str,
        agent_id: str,
        vector_store,  # VectorStoreBase
        embed_service,  # EmbedService
        layers=None,  # Optional[List[MemoryLayer]]
        cache=None,  # Optional[CacheBase], used to write pipeline_logs
        request_id: str = "",
        current_time: str = "",
        new_memories_with_meta: Optional[List[Dict]] = None,  # Full meta including tags
        session_id: str = "",  # Isolation dimension, passed through to pipeline_logs
        graph_store=None,  # Optional GraphStore: Entity MENTIONS boost
        new_memory_embeddings: Optional[Dict[str, List[float]]] = None,
        source: str = "default",
    ) -> ReconcileResult:
        """
        Reconcile new_memories.

        Args:
            new_memories: list of new memory contents (content only, used for embed search)
            new_memories_with_meta: optional, full meta including tags
        """
        if not new_memories:
            return ReconcileResult(ops=[], success=True)

        from ..models.memory import MemoryLayer, MemoryStatus, MemoryNode
        import asyncio

        if layers is None:
            layers = [MemoryLayer.L2_FACT, MemoryLayer.L4_IDENTITY]

        try:
            import time as _time

            # -- Step 1: Search query generation (ablatable) --
            search_queries: List[str] = []
            sq_response = None
            t_sq = _time.perf_counter()

            if self._enable_search_query:
                search_queries, sq_response = await self._generate_search_queries(
                    new_memories
                )
            sq_elapsed = (_time.perf_counter() - t_sq) * 1000

            # Token accounting: consumption of this search-query LLM call (the reconcile call is added later)
            _tok_pt = sq_response.prompt_tokens if sq_response else 0
            _tok_ct = sq_response.completion_tokens if sq_response else 0
            _tok_total = sq_response.tokens_used if sq_response else 0

            # Write the SEARCH_QUERY pipeline log
            if cache and request_id:
                mem_lines = "\n".join(f"{i+1}. {m}" for i, m in enumerate(new_memories))
                if self._enable_search_query:
                    from ..utils.lang_detect import is_chinese as _is_zh

                    if _is_zh("\n".join(new_memories)):
                        from .prompts_zh import SEARCH_QUERY_PROMPT_ZH

                        sq_prompt = SEARCH_QUERY_PROMPT_ZH.format(
                            new_memories=mem_lines
                        )
                    else:
                        sq_prompt = SEARCH_QUERY_PROMPT.format(new_memories=mem_lines)
                else:
                    sq_prompt = "[search query disabled — hybrid retrieval per new memory]"
                try:
                    await cache.store_pipeline_log(
                        request_id=request_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        step="SEARCH_QUERY",
                        prompt=sq_prompt,
                        response=json.dumps(search_queries, ensure_ascii=False),
                        parsed=json.dumps(search_queries, ensure_ascii=False),
                        elapsed_ms=sq_elapsed,
                        prompt_tokens=sq_response.prompt_tokens if sq_response else 0,
                        completion_tokens=(
                            sq_response.completion_tokens if sq_response else 0
                        ),
                        total_tokens=sq_response.tokens_used if sq_response else 0,
                    )
                except Exception as e:
                    logger.warning(f"[reconciler] store SEARCH_QUERY log failed: {e}")

            # -- Step 2: candidate search, merged into one candidate pool --
            candidate_map: Dict[str, MemoryNode] = {}
            candidate_scores: Dict[str, float] = {}

            supplied_embeddings = dict(new_memory_embeddings or {})
            mem_embeddings: List[Optional[List[float]]] = [
                supplied_embeddings.get(text) for text in new_memories
            ]
            missing_mem_indices = [
                i for i, emb in enumerate(mem_embeddings) if not emb
            ]
            texts_to_embed = [new_memories[i] for i in missing_mem_indices]
            texts_to_embed.extend(search_queries)
            computed_embeddings = (
                await embed_service.embed_batch(texts_to_embed)
                if texts_to_embed else []
            )
            for i, emb in zip(
                missing_mem_indices,
                computed_embeddings[: len(missing_mem_indices)],
            ):
                mem_embeddings[i] = emb
            query_embeddings = computed_embeddings[len(missing_mem_indices):]

            def _merge_hits(hits: List[Dict[str, Any]], source: str) -> None:
                logger.debug(f"[reconciler] search {source}: {len(hits)} results")
                for r in hits:
                    node = r.get("node")
                    score = float(r.get("score", 0))
                    if not node:
                        continue
                    # Canonical states are materialized and versioned by nightly
                    # evoling using an exact state_key.  Letting the free-form S1
                    # reconciler mutate them turns a stable state back into an
                    # event summary and breaks deterministic lifecycle handling.
                    if is_canonical_state_node(node):
                        continue
                    nid = node.node_id
                    if nid not in candidate_map:
                        candidate_map[nid] = node
                        candidate_scores[nid] = score
                    else:
                        candidate_scores[nid] = max(candidate_scores[nid], score)

            from ..pipelines._retrieval.reconcile_retrieval import (
                ReconcileHybridRetriever,
                ReconcileRetrievalConfig,
            )

            hybrid = ReconcileHybridRetriever(
                ReconcileRetrievalConfig.from_env(
                    limit=self.SEARCH_TOPK,
                    vec_pool_multiplier=self.VEC_POOL_MULTIPLIER,
                    min_score=self.HYBRID_MIN_SCORE,
                )
            )

            new_memory_embeddings = {
                text: emb
                for text, emb in zip(new_memories, mem_embeddings)
                if text and emb
            }

            hybrid_indices = [
                i for i in range(len(new_memories))
                if i < len(mem_embeddings) and mem_embeddings[i]
            ]

            query_indices = []
            if self._enable_search_query and search_queries:
                query_indices = [
                    i for i in range(len(search_queries))
                    if i < len(query_embeddings)
                ]

            _t_candidate_search = _time.perf_counter()
            hybrid_result = []
            query_result = []
            tasks = []
            if hybrid_indices:
                tasks.append(hybrid.search_candidates_batch(
                    [new_memories[i] for i in hybrid_indices],
                    [mem_embeddings[i] for i in hybrid_indices],
                    vector_store=vector_store,
                    user_id=user_id,
                    agent_id=agent_id,
                    layers=layers,
                    embed_service=embed_service,
                    graph_store=graph_store,
                    include_sources=[source],
                ))
            if query_indices:
                tasks.append(vector_store.search_batch([
                    {
                        "query_embedding": query_embeddings[i],
                        "user_id": user_id,
                        "agent_ids": None,
                        "layers": layers,
                        "limit": self.QUERY_SEARCH_TOPK,
                        "score_threshold": self.SEARCH_THRESHOLD,
                        "status_filter": [
                            MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED,
                        ],
                        "only_latest": False,
                        "include_sources": [source],
                    }
                    for i in query_indices
                ]))
            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                cursor = 0
                if hybrid_indices:
                    hybrid_result = task_results[cursor]
                    cursor += 1
                if query_indices:
                    query_result = task_results[cursor]
            _candidate_search_ms = round(
                (_time.perf_counter() - _t_candidate_search) * 1000, 2,
            )
            # For the relation graph: remember the hybrid candidates and their scores per new memory.
            # These scores used to be thrown away after being computed; building semantic_neighbor edges from them is zero extra computation on the write side.
            # Keyed by content -- the new memories are not persisted yet and have no node_id.
            per_memory_candidates: Dict[str, List[Tuple[str, float]]] = {}

            if isinstance(hybrid_result, Exception):
                logger.warning("[reconciler] hybrid batch failed: %s", hybrid_result)
                hybrid_result = []
            if isinstance(query_result, Exception):
                logger.warning("[reconciler] query batch failed: %s", query_result)
                query_result = []

            for source_index, result in zip(hybrid_indices, hybrid_result):
                _merge_hits(result, f"memory_hybrid[{source_index}]")
                per_memory_candidates[new_memories[source_index]] = [
                            (r["node"].node_id, float(r.get("score", 0.0)))
                            for r in (result or [])
                            if r.get("node") is not None
                            and not is_canonical_state_node(r["node"])
                ]
            for source_index, result in zip(query_indices, query_result):
                _merge_hits(
                    [
                        {"node": r.get("node"), "score": r.get("score", 0)}
                        for r in result
                    ],
                    f"query[{source_index}]",
                )

            logger.debug(
                f"[reconciler] search: {len(new_memories)} memories (hybrid) + "
                f"{len(search_queries)} queries → {len(candidate_map)} raw candidates"
            )

            # -- Step 2.5: merge by evolution chain (reuses the reader-side _trace_full_chain) --
            # The recall pool may contain SUPERSEDED old nodes (chain bodies). Every candidate node is traced in both
            # directions (supersedes / superseded_by) to complete the whole chain, and the chain head (is_latest) represents it
            # for deduplication: several hits on one chain (head + bodies) collapse into a single candidate, the head as representative,
            # and the head score becomes the highest score among all hits on that chain.
            # Note: ancestor nodes serve deduplication only and never enter the prompt (see Step 4:
            # on supersede only the head id + content are exposed, history is omitted, so the LLM cannot act on the wrong node).
            from ..pipelines._retrieval.evolution import trace_full_chains

            chain_head_node: Dict[str, MemoryNode] = {}   # head_id → head node
            chain_best_score: Dict[str, float] = {}       # head_id → highest hit score on the chain
            node_to_head: Dict[str, str] = {}             # any hit node_id → its chain head id
            chain_members: Dict[str, List[str]] = {}      # head_id → all node_ids of the chain (head + bodies)

            _t_chain_expand = _time.perf_counter()
            try:
                traced_chains = await trace_full_chains(
                    vector_store, list(candidate_map.values()),
                )
            except Exception as e:
                logger.warning("[reconciler] batch chain trace failed: %s", e)
                traced_chains = {}
            for nid, node in candidate_map.items():
                if nid in node_to_head:
                    continue  # Already covered by an expanded chain
                chain = traced_chains.get(nid) or [node]
                head = chain[0]  # _trace_full_chain guarantees [0] is the chain head (is_latest first)
                head_id = head.node_id
                chain_head_node.setdefault(head_id, head)
                chain_members[head_id] = [cn.node_id for cn in chain]
                # Every node on this chain that happened to be recalled contributes its score to the head candidate score
                for cn in chain:
                    node_to_head[cn.node_id] = head_id
                    if cn.node_id in candidate_scores:
                        chain_best_score[head_id] = max(
                            chain_best_score.get(head_id, 0.0),
                            candidate_scores[cn.node_id],
                        )

            # Sort by chain-head candidate score and take the top entries (with generous headroom)
            sorted_chains = sorted(
                chain_head_node.keys(),
                key=lambda hid: chain_best_score.get(hid, 0.0),
                reverse=True,
            )[: self.FINAL_TOPK * 2]

            # Final candidates: one chain head per chain (the prompt only shows the head)
            final_candidates: Dict[str, MemoryNode] = {}
            for head_id in sorted_chains:
                final_candidates[head_id] = chain_head_node[head_id]
                candidate_scores[head_id] = chain_best_score.get(head_id, 0.0)

            candidate_map = final_candidates
            _chain_expand_ms = round(
                (_time.perf_counter() - _t_chain_expand) * 1000, 2,
            )
            logger.debug(
                f"[reconciler] after chain merge: {len(candidate_map)} chain groups "
                f"(traced from {len(node_to_head)} recalled chain nodes)"
            )

            # -- Step 2.7: candidate composition instrumentation --
            # Candidate retrieval is now user-scoped (no longer narrowed by agent), so the pool mixes in memories of other
            # agents. With topk unchanged, this agent's candidates can be squeezed out ("dilution") for multi-agent users.
            # The composition is therefore persisted to the pipeline log, so after release it can be read
            # straight from the logs instead of waiting for business feedback.
            #
            # For single-agent users cross_agent_count is always 0 -- the same set is retrieved before and after the widening.
            try:
                _cand_own_agent = 0
                _cand_cross_agent = 0
                _cand_owner_user = 0
                _cand_owner_agent = 0
                _cross_agent_ids: set = set()
                for _n in candidate_map.values():
                    _naid = getattr(_n, "agent_id", "") or ""
                    if _naid and _naid != agent_id:
                        _cand_cross_agent += 1
                        _cross_agent_ids.add(_naid)
                    else:
                        _cand_own_agent += 1
                    if (getattr(_n, "owner", None) or "user") == "agent":
                        _cand_owner_agent += 1
                    else:
                        _cand_owner_user += 1

                _cand_parsed = {
                    "candidate_count": len(candidate_map),
                    "own_agent_count": _cand_own_agent,
                    "cross_agent_count": _cand_cross_agent,
                    "cross_agent_ids": sorted(_cross_agent_ids)[:20],
                    "owner_user_count": _cand_owner_user,
                    "owner_agent_count": _cand_owner_agent,
                    "recalled_chain_nodes": len(node_to_head),
                    "final_topk": self.FINAL_TOPK * 2,
                    # Dilution can only occur once the candidate pool is saturated; not saturated means no contention
                    "topk_saturated": len(candidate_map) >= self.FINAL_TOPK * 2,
                    "scope": "user",
                    "new_memory_count": len(new_memories),
                    "hybrid_query_count": len(hybrid_indices),
                    "query_query_count": len(query_indices),
                    "dense_batch_count": int(bool(hybrid_indices))
                    + int(bool(query_indices)),
                    "keyword_batch_count": int(bool(hybrid_indices)),
                    "candidate_search_ms": _candidate_search_ms,
                    "chain_expand_ms": _chain_expand_ms,
                    "vector_store_local": bool(
                        getattr(vector_store, "_is_local", False)
                    ),
                }
                logger.info(
                    f"[reconciler] candidates={len(candidate_map)} "
                    f"own_agent={_cand_own_agent} cross_agent={_cand_cross_agent} "
                    f"owner_user={_cand_owner_user} owner_agent={_cand_owner_agent} "
                    f"saturated={_cand_parsed['topk_saturated']}"
                )
                if cache:
                    await cache.store_pipeline_log(
                        request_id=request_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        step="RECONCILE_CANDIDATES",
                        prompt="",
                        response="",
                        parsed=json.dumps(_cand_parsed, ensure_ascii=False),
                    )
            except Exception as e:
                logger.debug(f"[reconciler] store RECONCILE_CANDIDATES log failed: {e}")

            # -- Step 2.6: candidate deduplication (pre-search) --
            # All candidates are chain heads (is_latest). Duplicates are only checked on L2_FACT + L4_IDENTITY (the repeat hotspots);
            # other layers do not participate (consistent with the search pipeline). A hit is set to SHADOW (logical deletion,
            # the duplicated chain head together with its history chain) and removed from candidate_map so the LLM never sees
            # duplicate candidates.
            #
            # Batching is decided by owner (group_items_by_owner): owner=user memories belong to the user side and are
            # deduplicated across agents, while owner=agent memories stay isolated per agent.
            try:
                import time as _time
                from ..pipelines._retrieval.dedup import (
                    DedupItem, execute_dedup, group_items_by_owner,
                )

                _DEDUP_LAYERS = {MemoryLayer.L2_FACT, MemoryLayer.L4_IDENTITY}
                head_ids = [
                    hid for hid, node in candidate_map.items()
                    if node.layer in _DEDUP_LAYERS
                ]
                _t_emb = _time.perf_counter()
                head_embs = await vector_store.get_embeddings(head_ids)
                _get_emb_ms = round((_time.perf_counter() - _t_emb) * 1000, 2)
                dedup_items = []
                for hid in head_ids:
                    node = candidate_map[hid]
                    emb = head_embs.get(hid)
                    if not emb:
                        continue
                    dedup_items.append(DedupItem(
                        node_id=hid,
                        embedding=emb,
                        content=node.content or "",
                        is_latest=True,
                        is_chain_head=True,
                        gmt_created=(node.gmt_created.timestamp() if node.gmt_created else None),
                        chain_node_ids=chain_members.get(hid, [hid]),
                        # Batching dimension: owner decides whether agents are merged
                        owner=getattr(node, "owner", None),
                        user_id=getattr(node, "user_id", "") or user_id,
                        agent_id=getattr(node, "agent_id", "") or agent_id,
                        source=getattr(node, "source", "default") or "default",
                    ))

                if len(dedup_items) >= 2:
                    # Pass the exact user_id/agent_id of each batch so the DEDUP log attribution is correct
                    for _batch in group_items_by_owner(
                        dedup_items,
                        default_user_id=user_id,
                        default_agent_id=agent_id,
                    ):
                        plan = await execute_dedup(
                            _batch["items"],
                            vector_store=vector_store,
                            cache=cache,
                            trigger="reconcile",
                            request_id=request_id,
                            user_id=_batch["user_id"],
                            agent_id=_batch["agent_id"],
                            delete_from_store=True,
                            extra_parsed={
                                "embedding_source": "vdb_get_embeddings",
                                "get_embeddings_ms": _get_emb_ms,
                                "embed_count": len(head_ids),
                                "embed_ok_count": len(dedup_items),
                                "owner_bucket": _batch["owner"],
                                "source_bucket": _batch["source"],
                                "batch_size": len(_batch["items"]),
                            },
                        )
                        for did in plan.get("delete_ids", []):
                            candidate_map.pop(did, None)
            except Exception as e:
                logger.warning(f"[reconciler] candidate dedup failed: {e}")

            # -- Step 3: with no candidates at all, everything becomes ADD (supersedes=[]) --
            if not candidate_map:
                logger.debug(
                    f"[reconciler] no candidates found, all ADD ({len(new_memories)} items)"
                )
                all_add_ops = []
                if new_memories_with_meta:
                    for meta in new_memories_with_meta:
                        all_add_ops.append(
                            ReconcileOp(
                                op="ADD",
                                content=meta.get("content", ""),
                                layer=meta.get("layer"),
                                owner=meta.get("owner") or "user",
                                tags=list(meta.get("tags") or []),
                                speculate=meta.get("speculate"),
                                topic_paths=meta.get("topic_paths") or [],
                                event_facet=meta.get("event_facet") or {},
                                topic_assignments=meta.get("topic_assignments") or [],
                                topic_paths_provided=bool(
                                    meta.get("topic_paths_provided", "topic_paths" in meta)
                                ),
                                event_facet_provided=bool(
                                    meta.get("event_facet_provided", "event_facet" in meta)
                                ),
                                topic_assignments_provided=bool(meta.get(
                                    "topic_assignments_provided",
                                    "topic_assignments" in meta,
                                )),
                            )
                        )
                else:
                    all_add_ops = [
                        ReconcileOp(op="ADD", content=m) for m in new_memories
                    ]

                if cache and request_id:
                    try:
                        await cache.store_pipeline_log(
                            request_id=request_id,
                            user_id=user_id,
                            agent_id=agent_id,
                            session_id=session_id,
                            step="RECONCILE",
                            prompt="[skipped - no candidates found in search]",
                            response=json.dumps(
                                [{"op": "ADD", "content": m} for m in new_memories],
                                ensure_ascii=False,
                            ),
                            parsed=json.dumps(
                                [
                                    {"op": o.op, "content": o.content}
                                    for o in all_add_ops
                                ],
                                ensure_ascii=False,
                            ),
                            elapsed_ms=0,
                        )
                    except Exception as e:
                        logger.warning(
                            f"[reconciler] store skipped RECONCILE log failed: {e}"
                        )
                logger.info(
                    f"[reconciler] no candidates → all ADD: "
                    f"{len(new_memories)} new memories → {len(all_add_ops)} ops"
                )
                _log_ops_detail(all_add_ops)
                return ReconcileResult(
                    ops=all_add_ops,
                    success=True,
                    prompt_tokens=_tok_pt,
                    completion_tokens=_tok_ct,
                    total_tokens=_tok_total,
                    candidate_scores=per_memory_candidates,
                    new_memory_embeddings=new_memory_embeddings,
                )

            # -- Step 4: build the prompt --
            _structure_metadata_on = bool(getattr(
                self.config.llm, "structure_metadata_enabled", False
            ))

            # Collect all existing tags of this user (deduplicated, so the LLM can reuse them instead of diverging)
            _existing_tags_set: set = set()
            try:
                for node in candidate_map.values():
                    if getattr(node, "tags", None):
                        _existing_tags_set.update(node.tags)
            except Exception:
                pass
            # Helper: format memory_at as date string to minute precision
            def _format_memory_at(val) -> Optional[str]:
                if val is None:
                    return None
                if isinstance(val, datetime):
                    return val.strftime("%Y-%m-%d %H:%M")
                if isinstance(val, str) and val:
                    try:
                        dt = datetime.fromisoformat(val)
                        return dt.strftime("%Y-%m-%d %H:%M")
                    except (ValueError, TypeError):
                        return val
                return None

            # Rendering of new_memories (JSON list format, aligned with existing_memories)
            if new_memories_with_meta:
                new_mem_list = []
                for meta in new_memories_with_meta:
                    item: Dict[str, Any] = {
                        "content": meta.get("content", ""),
                        "owner": meta.get("owner") or "user",
                        "memory_at": _format_memory_at(current_time) if current_time else None,
                        "tags": list(meta.get("tags") or []),
                    }
                    if _structure_metadata_on and meta.get("topic_paths"):
                        item["topic_paths"] = meta["topic_paths"]
                    if _structure_metadata_on and meta.get("event_facet"):
                        item["event_facet"] = meta["event_facet"]
                    if _structure_metadata_on and meta.get("topic_assignments"):
                        item["topic_assignments"] = meta["topic_assignments"]
                    new_mem_list.append(item)
                new_mem_lines = json.dumps(new_mem_list, ensure_ascii=False, indent=2)
            else:
                new_mem_list = [{"content": m, "owner": "user", "memory_at": _format_memory_at(current_time) if current_time else None, "tags": []} for m in new_memories]
                new_mem_lines = json.dumps(new_mem_list, ensure_ascii=False, indent=2)

            def _sort_key(n):
                t = n.memory_at or n.gmt_created
                return t.isoformat() if t else ""

            sorted_candidates = sorted(candidate_map.values(), key=_sort_key)

            existing_mem_list = []
            for node in sorted_candidates:
                item: Dict[str, Any] = {
                    "memory_id": node.node_id,
                    "content": node.content,
                    "owner": getattr(node, "owner", None) or "user",
                    "memory_at": _format_memory_at(node.memory_at),
                }
                if getattr(node, "tags", None):
                    item["tags"] = list(node.tags)
                if _structure_metadata_on:
                    _custom = getattr(node, "custom", None) or {}
                    if _custom.get("topic_paths"):
                        item["topic_paths"] = _custom["topic_paths"]
                    if _custom.get("event_facet"):
                        item["event_facet"] = _custom["event_facet"]
                    if _custom.get("topic_assignments"):
                        item["topic_assignments"] = _custom["topic_assignments"]
                # Each evolution chain exposes only the head memory_id + content; history_versions are deliberately
                # **omitted**: reconcile should only act on the currently valid version (head), and exposing
                # ADD/UPDATE/SUPERSEDE; exposing historical old nodes invites wrong ops from the LLM (e.g. trying to
                # supersede a node that has already been superseded). Full chain expansion only happens on the reader side.
                existing_mem_list.append(item)
            existing_lines = json.dumps(existing_mem_list, ensure_ascii=False, indent=2)

            # Build current_date (minute precision)
            _current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

            prompt = build_reconcile_prompt(
                new_memories=new_memories,
                new_memories_block=new_mem_lines,
                existing_memories_block=existing_lines,
                current_date=_current_date_str,
                few_shot_enabled=bool(
                    getattr(self.config.llm, "few_shot_enabled", False)
                ),
                structure_metadata_enabled=_structure_metadata_on,
            )

            # -- Step 5: LLM call + retries --
            if _log_llm_content_enabled():
                logger.debug(f"[reconciler] full prompt: {prompt!r}")
            else:
                logger.debug(
                    "[reconciler] prompt hidden; len=%d sha256=%s",
                    len(prompt), _content_fingerprint(prompt),
                )
            # Estimate max_tokens dynamically
            # Take the larger of the base and the dynamic value, clamped to the upper bound.
            dyn_max_tokens = min(
                self.MAX_TOKENS_UPPER_BOUND,
                max(
                    self.MAX_TOKENS,
                    len(new_memories) * self.PER_OP_TOKENS + self.BASE_OVERHEAD_TOKENS,
                ),
            )
            t_recon = _time.perf_counter()
            llm = self._get_llm()
            response = None
            raw_response = ""
            ops = []
            _parse_ok = False
            # A JSON parse failure retries once (first attempt + 1 retry = 2 attempts in total).
            # Combined with the json_repair fallback in _parse_ops, most truncated outputs are already fixed on the first attempt.
            # Both attempts use the same temperature (it used to be a [base, base] list, i.e. a constant),
            # and temperature is no longer passed explicitly -- the provider resolves it from config,
            # otherwise omit_temperature would be bypassed by the explicit argument here.
            _MAX_PARSE_ATTEMPTS = 2
            _llm_error: Optional[str] = None

            for _attempt in range(_MAX_PARSE_ATTEMPTS):
                try:
                    response = await llm.complete(
                        prompt=prompt,
                        max_tokens=dyn_max_tokens,
                    )
                except Exception as e:
                    # The LLM is completely unavailable (the provider already retried per max_retries; gateway 500 /
                    # connection errors and the like). It **must not bubble up** -- that would discard the whole extraction batch.
                    # Leave it to the ADD-all fallback below.
                    _llm_error = f"{type(e).__name__}: {e}"
                    logger.warning(f"[reconciler] RECONCILE LLM unavailable: {_llm_error}")
                    break
                # Every reconcile LLM call (retries included) counts toward token consumption
                _tok_pt += response.prompt_tokens
                _tok_ct += response.completion_tokens
                _tok_total += response.tokens_used
                raw_response = response.content
                ops = self._parse_ops(raw_response)
                ops = self._inherit_exact_input_structure(
                    ops, new_memories_with_meta,
                )
                if _structure_metadata_on:
                    ops = self._resolve_structure_references(ops, candidate_map)

                _is_empty_array = self._strip_code_fence(raw_response) in ("[]", "")
                if ops or _is_empty_array:
                    _parse_ok = True
                    break
                logger.warning(
                    f"[reconciler] RECONCILE JSON parse failed "
                    f"(attempt {_attempt+1}/{_MAX_PARSE_ATTEMPTS}), "
                    f"retrying: raw="
                    f"{raw_response[:200] if _log_llm_content_enabled() else '<redacted>'}"
                )

            recon_elapsed = (_time.perf_counter() - t_recon) * 1000

            # -- Step 6: parsing still failed --
            if not _parse_ok:
                _n_attempts = _MAX_PARSE_ATTEMPTS
                if _llm_error:
                    error_msg = f"RECONCILE LLM unavailable: {_llm_error}"
                    error_code = "LLM_UNAVAILABLE"
                elif raw_response.strip():
                    error_msg = (
                        f"RECONCILE JSON parse failed after {_n_attempts} attempts: "
                        f"{raw_response[:200] if _log_llm_content_enabled() else '<redacted>'}"
                    )
                    error_code = "JSON_PARSE_FAILED"
                else:
                    error_msg = (
                        f"RECONCILE LLM returned empty response after {_n_attempts} attempts"
                    )
                    error_code = "EMPTY_RESPONSE" 
                logger.warning(f"[reconciler] {error_msg}")

                if cache and request_id:
                    try:
                        await cache.store_pipeline_log(
                            request_id=request_id,
                            user_id=user_id,
                            agent_id=agent_id,
                            session_id=session_id,
                            step="RECONCILE",
                            prompt=prompt,
                            response=raw_response,
                            parsed=json.dumps(
                                {
                                    "error": error_code,
                                    "message": error_msg,
                                    "fallback": "ADD_ALL",
                                    "fallback_count": len(new_memories or []),
                                },
                                ensure_ascii=False,
                            ),
                            elapsed_ms=recon_elapsed,
                            # When the LLM is unavailable response is None -- this very line would raise
                            # AttributeError, and since it sits on the fallback path it would take the fallback down with it
                            prompt_tokens=getattr(response, "prompt_tokens", 0) or 0,
                            completion_tokens=getattr(response, "completion_tokens", 0) or 0,
                            total_tokens=getattr(response, "tokens_used", 0) or 0,
                        )
                    except Exception as e:
                        logger.warning(
                            f"[reconciler] store failed RECONCILE log failed: {e}"
                        )

                # Fallback: unparseable JSON does not mean the extraction is worthless -- ADD everything, keep the content.
                _fb = self._fallback_add_ops(new_memories, new_memories_with_meta)
                logger.warning(
                    f"[reconciler] falling back to ADD-all: {len(_fb)} memories "
                    f"({error_code})"
                )
                return ReconcileResult(
                    ops=_fb,
                    success=bool(_fb),
                    error=None if _fb else error_msg,
                    degraded=bool(_fb),
                    degraded_reason=error_code,
                    prompt_tokens=_tok_pt,
                    completion_tokens=_tok_ct,
                    total_tokens=_tok_total,
                    new_memory_embeddings=new_memory_embeddings,
                )

            # Write the RECONCILE pipeline log
            if cache and request_id:
                related_ids = []
                for op in ops:
                    if op.op in ("SUPERSEDE", "UPDATE") and op.memory_id:
                        related_ids.append(op.memory_id)
                try:
                    await cache.store_pipeline_log(
                        request_id=request_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        step="RECONCILE",
                        prompt=prompt,
                        response=response.content,
                        parsed=json.dumps(
                            [
                                {
                                    "op": o.op,
                                    "content": o.content,
                                    "layer": o.layer,
                                    "memory_id": o.memory_id,
                                    "tags": o.tags,
                                    "topic_paths": o.topic_paths,
                                    "event_facet": o.event_facet,
                                    "topic_assignments": o.topic_assignments,
                                    "structure_references": o.structure_references,
                                    "related": o.related,
                                    "speculate": o.speculate,
                                    "supersede_reason": o.supersede_reason,
                                    "reason": o.reason,
                                }
                                for o in ops
                            ],
                            ensure_ascii=False,
                        ),
                        memory_ids=related_ids,
                        elapsed_ms=recon_elapsed,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        total_tokens=response.tokens_used,
                    )
                except Exception as e:
                    logger.warning(f"[reconciler] store RECONCILE log failed: {e}")

            logger.info(
                f"[reconciler] reconciled {len(new_memories)} new memories, "
                f"{len(candidate_map)} candidates → {len(ops)} ops"
            )
            _log_ops_detail(ops)
            return ReconcileResult(
                ops=ops,
                success=True,
                prompt_tokens=_tok_pt,
                completion_tokens=_tok_ct,
                total_tokens=_tok_total,
                candidate_scores=per_memory_candidates,
                new_memory_embeddings=new_memory_embeddings,
            )

        except Exception as e:
            logger.error(f"[reconciler] failed: {e}", exc_info=True)
            # The fallback also applies when any other stage of the reconcile pipeline blows up (candidate retrieval, embed,
            # prompt assembly...). Previously this returned ops=[] and the writer discarded the entire extraction batch.
            _fb = self._fallback_add_ops(new_memories, new_memories_with_meta)
            if _fb:
                logger.warning(
                    f"[reconciler] falling back to ADD-all after unexpected error: "
                    f"{len(_fb)} memories"
                )
                return ReconcileResult(
                    ops=_fb, success=True,
                    degraded=True, degraded_reason=f"EXCEPTION:{type(e).__name__}",
                    new_memory_embeddings=dict(new_memory_embeddings or {}),
                )
            return ReconcileResult(ops=[], success=False, error=str(e))

    @staticmethod
    def _inherit_exact_input_structure(
        ops: List[ReconcileOp],
        new_memories_with_meta: Optional[List[Dict]],
    ) -> List[ReconcileOp]:
        """Preserve extractor structure only when provenance is unambiguous.

        Reconcile may rewrite content, so this intentionally uses exact content
        identity. No fuzzy or positional matching is allowed: uncertain
        provenance remains empty rather than attaching an L2 to the wrong node.
        """
        by_content = {
            str(meta.get("content") or ""): meta
            for meta in (new_memories_with_meta or [])
            if isinstance(meta, dict) and meta.get("content")
        }
        from ..structure import (
            normalize_event_facet,
            normalize_topic_assignments,
            normalize_topic_paths,
        )

        for op in ops:
            meta = by_content.get(str(op.content or ""))
            if not meta:
                continue
            if not op.topic_paths_provided:
                op.topic_paths = normalize_topic_paths(meta.get("topic_paths"))
                op.topic_paths_provided = bool(
                    meta.get("topic_paths_provided", "topic_paths" in meta)
                )
            if not op.event_facet_provided:
                op.event_facet = normalize_event_facet(meta.get("event_facet"))
                op.event_facet_provided = bool(
                    meta.get("event_facet_provided", "event_facet" in meta)
                )
            if not op.topic_assignments_provided:
                op.topic_assignments = normalize_topic_assignments(
                    meta.get("topic_assignments")
                )
                op.topic_assignments_provided = bool(meta.get(
                    "topic_assignments_provided", "topic_assignments" in meta,
                ))
        return ops

    @staticmethod
    def _parse_related(raw: Any) -> List[Dict[str, str]]:
        """Parse the `related` field of an op (relation graph, see docs/pro_graph_design.md).

        Accepts [{"memory_id": ..., "relation": ...}] and leniently tolerates the target_id / relation_type
        spellings of those keys. Only whitelisted relation_type values are accepted -- weak models invent their own relation words, and putting them in would only pollute
        the edge-type prior table. Invalid entries are dropped silently (relations are auxiliary signals and must not drag down the main flow).
        """
        from ..pipelines._graph.projector import LLM_RELATION_TYPES

        if not isinstance(raw, list):
            return []
        out: List[Dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            target = str(
                item.get("memory_id") or item.get("target_id") or ""
            ).strip()
            rtype = str(
                item.get("relation") or item.get("relation_type") or ""
            ).strip().lower()
            if not target or rtype not in LLM_RELATION_TYPES:
                continue
            out.append({"target_id": target, "relation_type": rtype})
        return out

    @staticmethod
    def _resolve_structure_references(
        ops: List[ReconcileOp], candidate_map: Dict[str, Any],
    ) -> List[ReconcileOp]:
        """Resolve only explicit, in-candidate path references."""
        from ..structure import canonicalize_assignments_from_references

        existing_custom = {
            str(memory_id): dict(getattr(node, "custom", None) or {})
            for memory_id, node in (candidate_map or {}).items()
        }
        for op in ops:
            if not op.structure_references:
                continue
            resolved = canonicalize_assignments_from_references(
                op.topic_assignments,
                op.event_facet,
                op.structure_references,
                existing_custom,
            )
            if resolved:
                op.topic_assignments = resolved
                op.topic_assignments_provided = True
        return ops

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Strip the ```json ... ``` / ``` ... ``` markdown fences and return the inner content."""
        text = (text or "").strip()
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text

    def _parse_ops(self, text: str) -> List[ReconcileOp]:
        """
        Parse the RECONCILE output returned by the LLM.

        Supported formats:
        - Flat list (preferred): [{op, content, ...}, {op, content, ...}]
        - Group format (backward compatible): [{"reason": ..., "ops": [...]}]

        Supported op types: ADD, SUPERSEDE, UPDATE
        """
        text = self._strip_code_fence(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            data = None
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = None
            if data is None:
                # Fallback: json_repair repairs common truncated LLM output (missing commas/quotes, unclosed brackets, ...)
                try:
                    from json_repair import repair_json

                    repaired = repair_json(text, return_objects=True)
                    if repaired not in (None, "", {}):
                        logger.info("[reconciler] JSON recovered via json_repair")
                        data = repaired
                except Exception as _re:
                    logger.debug(f"[reconciler] json_repair failed: {_re}")
            if data is None:
                logger.warning(f"[reconciler] failed to parse ops JSON: {text[:200]}")
                return []

        if not isinstance(data, list):
            data = [data]

        # -- Flatten: collect (op_dict, group_reason) pairs into a flat list --
        flat_items: List[tuple] = []  # (op_dict, group_reason)
        for entry in data:
            if not isinstance(entry, dict):
                continue
            # Branch A: group structure (backward compatible) - reason plus an ops sub-list
            if "ops" in entry and isinstance(entry.get("ops"), list):
                group_reason = str(entry.get("reason") or "").strip()
                for op_dict in entry["ops"]:
                    if isinstance(op_dict, dict):
                        flat_items.append((op_dict, group_reason))
            # Branch B: flat op structure (preferred)
            elif "op" in entry:
                flat_items.append((entry, ""))
            else:
                logger.warning(
                    f"[reconciler] unrecognized entry shape, skipped: {entry}"
                )

        # Pass 1: structural parsing
        parsed: List[ReconcileOp] = []
        for op_dict, group_reason in flat_items:
            op_type = str(op_dict.get("op", "")).upper()
            op_reason = str(op_dict.get("reason") or "").strip() or group_reason

            # Common field extraction
            raw_tags = op_dict.get("tags", []) or []
            if not isinstance(raw_tags, list):
                raw_tags = [raw_tags]
            # Keep proper-noun capitalization (Jasper / Canada); only strip, never lower.
            # Topic words may still come back lowercase from the LLM; NER attributes go into tag_ners.
            tags = [str(t).strip() for t in raw_tags if t and str(t).strip()][:3]
            try:
                from ..pipelines._retrieval.ner import annotate_tag_ners
                tag_ners = annotate_tag_ners(tags)
            except Exception:
                tag_ners = [""] * len(tags)
            speculate = op_dict.get("speculate") or None
            if speculate and isinstance(speculate, str):
                speculate = speculate.strip() or None

            # owner: only user/agent accepted (assistant→agent), otherwise None
            _raw_owner = op_dict.get("owner")
            owner = None
            if _raw_owner:
                _ov = str(_raw_owner).strip().lower()
                if _ov in ("user", "agent"):
                    owner = _ov
                elif _ov == "assistant":
                    owner = "agent"

            related = self._parse_related(op_dict.get("related"))
            from ..structure import (
                normalize_event_facet,
                normalize_structure_references,
                normalize_topic_assignments,
                normalize_topic_paths,
            )
            topic_paths = normalize_topic_paths(op_dict.get("topic_paths"))
            event_facet = normalize_event_facet(op_dict.get("event_facet"))
            topic_assignments = normalize_topic_assignments(
                op_dict.get("topic_assignments")
            )
            structure_references = normalize_structure_references(
                op_dict.get("structure_references")
            )
            topic_paths_provided = "topic_paths" in op_dict
            event_facet_provided = "event_facet" in op_dict
            topic_assignments_provided = "topic_assignments" in op_dict

            if op_type == "ADD":
                parsed.append(
                    ReconcileOp(
                        op="ADD",
                        content=op_dict.get("content"),
                        layer=op_dict.get("layer"),
                        owner=owner,
                        tags=tags,
                        tag_ners=tag_ners,
                        topic_paths=topic_paths,
                        event_facet=event_facet,
                        topic_assignments=topic_assignments,
                        structure_references=structure_references,
                        topic_paths_provided=topic_paths_provided,
                        event_facet_provided=event_facet_provided,
                        topic_assignments_provided=topic_assignments_provided,
                        speculate=speculate,
                        related=related,
                        reason=op_reason,
                    )
                )

            elif op_type == "SUPERSEDE":
                # Accept a single id (memory_id) or a multi-id chain (memory_ids, ordered oldest→newest)
                raw_ids = op_dict.get("memory_ids")
                id_list: List[str] = []
                if isinstance(raw_ids, list):
                    id_list = [str(x).strip() for x in raw_ids if x and str(x).strip()]
                mid = str(op_dict.get("memory_id") or "").strip()
                if mid and mid not in id_list:
                    # memory_id counts as the newest target on the chain and is appended at the end
                    id_list.append(mid)
                if not id_list:
                    logger.warning(
                        f"[reconciler] SUPERSEDE op missing memory_id(s), skipped: {op_dict}"
                    )
                    continue
                # memory_id always points at the newest existing node on the chain (the new head supersedes it directly)
                parsed.append(
                    ReconcileOp(
                        op="SUPERSEDE",
                        memory_id=id_list[-1],
                        memory_ids=id_list,
                        content=op_dict.get("content"),
                        layer=op_dict.get("layer"),
                        owner=owner,
                        tags=tags,
                        tag_ners=tag_ners,
                        topic_paths=topic_paths,
                        event_facet=event_facet,
                        topic_assignments=topic_assignments,
                        structure_references=structure_references,
                        topic_paths_provided=topic_paths_provided,
                        event_facet_provided=event_facet_provided,
                        topic_assignments_provided=topic_assignments_provided,
                        speculate=speculate,
                        supersede_reason=str(
                            op_dict.get("supersede_reason") or ""
                        ).strip(),
                        related=related,
                        reason=op_reason,
                    )
                )

            elif op_type == "UPDATE":
                mid = str(op_dict.get("memory_id") or "").strip()
                if not mid:
                    logger.warning(
                        f"[reconciler] UPDATE op missing memory_id, skipped: {op_dict}"
                    )
                    continue
                parsed.append(
                    ReconcileOp(
                        op="UPDATE",
                        memory_id=mid,
                        content=op_dict.get("content"),
                        layer=op_dict.get("layer"),
                        owner=owner,
                        tags=tags,
                        tag_ners=tag_ners,
                        topic_paths=topic_paths,
                        event_facet=event_facet,
                        topic_assignments=topic_assignments,
                        structure_references=structure_references,
                        topic_paths_provided=topic_paths_provided,
                        event_facet_provided=event_facet_provided,
                        topic_assignments_provided=topic_assignments_provided,
                        speculate=speculate,
                        related=related,
                        reason=op_reason,
                    )
                )

            elif op_type == "DELETE":
                # Backward compatibility: old prompts may still return DELETE, mapped to SHADOW for the old node
                mid = str(op_dict.get("memory_id") or "").strip()
                if not mid:
                    logger.warning(
                        f"[reconciler] DELETE op missing memory_id, skipped: {op_dict}"
                    )
                    continue
                logger.info(
                    f"[reconciler] legacy DELETE op mapped (memory_id={mid}), "
                    f"will shadow the node"
                )
                parsed.append(
                    ReconcileOp(
                        op="UPDATE",
                        memory_id=mid,
                        content=None,  # Marks a shadow-only op with no content
                        reason=op_reason,
                    )
                )

            else:
                logger.warning(
                    f"[reconciler] unknown op '{op_type}', skipped: {op_dict}"
                )
                continue

        # Pass 2: verify that no existing memory is touched by more than one op
        touched: Dict[str, str] = {}  # memory_id → first-seen op description
        validated: List[ReconcileOp] = []
        for op in parsed:
            # SUPERSEDE multi-node chain: register every id on the chain; UPDATE and single-node SUPERSEDE register memory_id
            if op.op == "SUPERSEDE" and op.memory_ids:
                target_ids = list(op.memory_ids)
            elif op.op in ("SUPERSEDE", "UPDATE"):
                target_ids = [op.memory_id] if op.memory_id else []
            else:
                target_ids = []
            if target_ids:
                clash = next((t for t in target_ids if t in touched), None)
                if clash is not None:
                    logger.warning(
                        f"[reconciler] memory_id {clash} already touched by '{touched[clash]}', "
                        f"current op '{op.op} ({op.reason[:60]})' dropped"
                    )
                    continue
                for t in target_ids:
                    touched[t] = f"{op.op}({op.reason[:40]})"
            validated.append(op)

        return validated
