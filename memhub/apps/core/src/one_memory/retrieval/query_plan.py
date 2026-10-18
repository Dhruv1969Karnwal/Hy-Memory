"""Deterministic local QueryPlan. No HTTP schema, no store, no model call."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from one_memory.retrieval.models import QueryPlan, TimeRange

_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_PATH = re.compile(r"(?:~|/|\./|\.\./|[A-Za-z]:\\)[^\s]+|[^\s]+/[^\s]+\.[A-Za-z0-9]{1,8}")
_SECRET_QUERY = frozenset(
    {
        "token",
        "access_token",
        "signature",
        "sig",
        "key",
        "password",
        "api_key",
        "auth",
        "authorization",
    }
)

_ISO_DAY = r"(\d{4})-(\d{2})-(\d{2})"
_ISO_RANGE = re.compile(rf"{_ISO_DAY}\s*(?:\u5230|\u81f3|~|—|–)\s*{_ISO_DAY}")
_ISO_SINGLE = re.compile(_ISO_DAY)
_RECENT_DAYS = re.compile(r"(?:\u6700\u8fd1|\u8fd1|\u8fc7\u53bb)\s*(\d{1,3})\s*\u5929")
_RECENT_VAGUE = re.compile(r"\u6700\u8fd1|\u8fd1\u671f")
#: Asking what holds *now* is a question about the present, so a date mentioned
#: alongside it describes the conversation, not the window to search.
_PRESENT = re.compile(r"\u73b0\u5728|\u76ee\u524d|\u5f53\u524d|\u6700\u65b0|\u5982\u4eca")
_DEFAULT_TIMEZONE = "Asia/Shanghai"
TimeMode = Literal["none", "hard", "soft"]


def plan_local_query(
    query: str,
    *,
    now: datetime | None = None,
    timezone: str = _DEFAULT_TIMEZONE,
) -> QueryPlan:
    """Keep the original question; only attach signals provable from its text."""
    zone = ZoneInfo(timezone)
    urls = _urls(query)
    paths = _paths(query)
    moment = (now or datetime.now(zone)).astimezone(zone)
    # Objects carry their own digits; a date inside a link is part of the link.
    time_range, time_mode = _time(_without_objects(query), moment, zone)
    suppressed = time_mode != "none" and _PRESENT.search(query) is not None
    return QueryPlan(
        original_query=query,
        intent="fact",
        queries=(query,),
        object_keys=tuple(dict.fromkeys((*urls, *paths))),
        time_range=None if suppressed else time_range,
        time_mode="none" if suppressed else time_mode,
        ambiguous_constraints=("time_reference_not_scoping",) if suppressed else (),
    )


def _time(
    text: str,
    now: datetime,
    zone: ZoneInfo,
) -> tuple[TimeRange | None, TimeMode]:
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def span(start_day: datetime, days: int) -> tuple[TimeRange, Literal["hard"]]:
        return TimeRange(start=start_day, end=start_day + timedelta(days=days)), "hard"

    if found := _ISO_RANGE.search(text):
        first = _day(found.group(1, 2, 3), zone)
        last = _day(found.group(4, 5, 6), zone)
        if first is not None and last is not None and first <= last:
            return TimeRange(start=first, end=last + timedelta(days=1)), "hard"
    if (found := _ISO_SINGLE.search(text)) and (
        only := _day(found.group(1, 2, 3), zone)
    ) is not None:
        return span(only, 1)
    for phrase, offset in (("\u4eca\u5929", 0), ("\u6628\u5929", -1), ("\u524d\u5929", -2)):
        if phrase in text:
            return span(midnight + timedelta(days=offset), 1)
    monday = midnight - timedelta(days=midnight.weekday())
    if "\u672c\u5468" in text:
        return span(monday, 7)
    if "\u4e0a\u5468" in text:
        return span(monday - timedelta(days=7), 7)
    if found := _RECENT_DAYS.search(text):
        days = int(found.group(1))
        if days >= 1:
            # Inclusive of today, so "the last 3 days" ends at tomorrow's midnight.
            return span(midnight - timedelta(days=days - 1), days)
    if _RECENT_VAGUE.search(text):
        # No number to honour; inventing a window would filter on a guess.
        return None, "soft"
    return None, "none"


def _day(parts: tuple[str, ...], zone: ZoneInfo) -> datetime | None:
    year, month, number = (int(part) for part in parts)
    try:
        return datetime(year, month, number, tzinfo=zone)
    except ValueError:
        return None


def _without_objects(query: str) -> str:
    text = _URL.sub(" ", query)
    return _PATH.sub(" ", text)


def _urls(query: str) -> tuple[str, ...]:
    found: list[str] = []
    for raw in _URL.findall(query):
        cleaned = raw.rstrip(").,;\"'")
        parsed = urlsplit(cleaned)
        if not parsed.scheme or not parsed.hostname:
            continue
        query_pairs = [
            (name, value)
            for name, value in parse_qsl(parsed.query, keep_blank_values=True)
            if name.lower() not in _SECRET_QUERY
        ]
        found.append(
            urlunsplit(
                (
                    parsed.scheme.lower(),
                    parsed.netloc.lower(),
                    parsed.path,
                    urlencode(query_pairs),
                    parsed.fragment,
                )
            )
        )
    return tuple(found)


def _paths(query: str) -> tuple[str, ...]:
    url_spans = {raw.rstrip(").,;\"'") for raw in _URL.findall(query)}
    found: list[str] = []
    for match in _PATH.findall(query):
        cleaned = match.rstrip(").,;\"'")
        if _URL.match(cleaned) or "://" in cleaned:
            continue
        if any(cleaned in url or url.endswith(cleaned) for url in url_spans):
            continue
        found.append(cleaned)
    return tuple(found)
