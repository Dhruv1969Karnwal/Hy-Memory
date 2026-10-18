"""Deterministic RCL-13 trigger semantics, independent of any host hook."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

_HISTORY = re.compile(
    r"\u4e4b\u524d|\u4ee5\u524d|\u4e0a\u6b21|\u5386\u53f2|\u8fc7\u53bb|\u5f53\u65f6|\u521a\u624d|\u524d\u9762|\u8bb0\u5f97|\u56de\u5fc6|"
    r"previous|before|last time|history|earlier|remember",
    re.I,
)
_CONTINUE = re.compile(
    r"\u7ee7\u7eed|\u63a5\u7740|\u7eed\u63a5|\u63a5\u4e0a|\u505a\u5230\u54ea|\u4e0b\u4e00\u6b65|\u672a\u5b8c\u6210|"
    r"continue|resume|pick up|where (?:did|were) we",
    re.I,
)
_PERSONAL = re.compile(
    r"\u6211\u7684.{0,12}(\u504f\u597d|\u4e60\u60ef|\u8981\u6c42|\u51b3\u5b9a)|\u6211.{0,8}(\u559c\u6b22|\u4e0d\u559c\u6b22|\u8981\u6c42|\u51b3\u5b9a)|"
    r"my (?:preference|decision)|what did i decide",
    re.I,
)
_GREETING = re.compile(
    r"^(?:\u4f60\u597d|\u60a8\u597d|\u55e8|\u54c8\u55bd|hello|hi|hey)[!\uff01,.\uff0c\u3002\s]*$",
    re.I,
)


class AutoRecallDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["search", "skip", "skill_guidance"]
    reason: Literal[
        "explicit_query",
        "history_reference",
        "task_continuation",
        "past_preference_or_decision",
        "disabled",
        "permission_required",
        "already_satisfied",
        "ordinary_message",
        "turn_identity_unavailable",
    ]
    automatic: bool


def decide_auto_recall(
    query: str,
    *,
    explicit_lookup: bool = False,
    enabled: bool = True,
    read_allowed: bool = True,
    reliable_turn_identity: bool = True,
    already_satisfied: bool = False,
) -> AutoRecallDecision:
    """Choose whether a host may search; never reads context or creates a grant."""
    text = query.strip()
    if not text:
        raise ValueError("auto_recall_query_required")
    if explicit_lookup:
        if not read_allowed:
            return AutoRecallDecision(action="skip", reason="permission_required", automatic=False)
        return AutoRecallDecision(action="search", reason="explicit_query", automatic=False)
    if not enabled:
        return AutoRecallDecision(action="skip", reason="disabled", automatic=True)
    if not read_allowed:
        return AutoRecallDecision(action="skip", reason="permission_required", automatic=True)
    if already_satisfied:
        return AutoRecallDecision(action="skip", reason="already_satisfied", automatic=True)
    reason: (
        Literal["history_reference", "task_continuation", "past_preference_or_decision"] | None
    ) = None
    if _CONTINUE.search(text):
        reason = "task_continuation"
    elif _PERSONAL.search(text):
        reason = "past_preference_or_decision"
    elif _HISTORY.search(text):
        reason = "history_reference"
    if reason is None or _GREETING.fullmatch(text):
        return AutoRecallDecision(action="skip", reason="ordinary_message", automatic=True)
    if not reliable_turn_identity:
        return AutoRecallDecision(
            action="skill_guidance", reason="turn_identity_unavailable", automatic=True
        )
    return AutoRecallDecision(action="search", reason=reason, automatic=True)
