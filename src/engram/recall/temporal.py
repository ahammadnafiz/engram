"""Natural-language timeframe resolution for memory recall.

Converts a human temporal phrase ("yesterday", "last week", "past month")
into a ``(since, until)`` datetime window, anchored on a reference time.
Uses the optional ``dateparser`` dependency (``engram[temporal]``); when it is
not installed, resolution degrades gracefully to ``(None, None)`` so recall
still works without a temporal filter.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def resolve_timeframe(
    phrase: str | None,
    *,
    base: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Resolve a temporal phrase to a ``(since, until)`` window.

    Args:
        phrase: A natural-language temporal expression (e.g. "yesterday",
            "last month"). Empty/None returns ``(None, None)``.
        base: Reference "now" the phrase is relative to. Defaults to the
            current UTC time.

    Returns:
        ``(since, until)`` as timezone-aware datetimes, or ``(None, None)``
        when the phrase is empty, unparseable, or ``dateparser`` is missing.

    The window is interpreted in the past: "yesterday" spans that whole day,
    "last week" the preceding week, etc. A bare day resolves to that day's
    00:00..23:59:59.999999.
    """
    if not phrase or not phrase.strip():
        return None, None

    base = base or _utcnow()
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    try:
        from dateparser.search import search_dates  # type: ignore[import-untyped]
    except ImportError:
        return None, None

    settings = {
        "RELATIVE_BASE": base.replace(tzinfo=None),
        "PREFER_DATES_FROM": "past",
        "RETURN_TIME_SPAN": True,
    }
    try:
        found = search_dates(phrase, settings=settings)
    except Exception:
        return None, None
    if not found:
        return None, None

    dates = [dt for _label, dt in found if isinstance(dt, datetime)]
    if not dates:
        return None, None

    dates = [d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d for d in dates]

    if len(dates) >= 2:
        since, until = min(dates), max(dates)
    else:
        # Single point -> treat as the whole day it lands in.
        day = dates[0]
        since = day.replace(hour=0, minute=0, second=0, microsecond=0)
        until = since + timedelta(days=1) - timedelta(microseconds=1)
    return since, until
