"""When did the event actually happen? - extracting event time from article text.

The problem this exists to solve, in one real example. On 2026-10-25 the system emailed an
URGENT "prediction confirmed" alert citing an article about a landslide. The article was
published on 11 October and described an event from 11 October. The pipeline had only the
RSS ``pubDate`` to go on, treated the item as fresh the day it was scraped, and raised an
alarm about a two-week-old event. Google News RSS re-surfaces older stories constantly
(``when:7d`` is a *search* window, not a freshness guarantee), so this is not a rare case.

A publication date is not an event date. This module reads the article for when the event
is actually said to have happened.

Design - the same split as ``geocoding.py``, for the same reason
----------------------------------------------------------------
**The model extracts a phrase; deterministic code resolves it to a timestamp.** An LLM asked
for a date returns a confident, well-formed, frequently wrong one, and a hallucinated
timestamp is indistinguishable from a real one downstream. So the model is only ever allowed
to point at the span of text it believes states the event time, and
:func:`resolve_phrase` - plain, tested Python - turns that span into a time. Any timestamp
key the model emits anyway is stripped before a caller can read it
(:func:`_strip_llm_timestamps`), mirroring ``_strip_llm_coordinates``.

Everything resolves relative to ``published_utc``, never to ``now()``: an article scraped
today may have been published weeks ago, which is precisely the bug above.

Refusal is a supported outcome. When no phrase is found the result carries
``confidence="publication_fallback"`` and an ``ambiguity_days`` wide enough to say so. Callers
must branch on ``confidence``, not silently treat the returned time as observed fact.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("nepal_mhews.event_time")

# Nepal Standard Time. News reports local calendar days; resolving them in UTC would put
# every event up to 5h45m off and could move it across a day boundary.
NPT = timezone(timedelta(hours=5, minutes=45))

# An article cannot report an event that has not happened. A resolution landing after
# publication is a parse error, with this much slack for timezone edges and same-day
# "breaking" updates filed slightly ahead of the wire timestamp.
FUTURE_TOLERANCE_HOURS = 18.0

# An LLM call is only worth making when there is prose beyond the headline. See the call
# site for the measurement behind this.
MIN_BODY_CHARS_FOR_LLM = 200

# How far back a resolved event may sit before we stop believing the phrase belongs to the
# article's own event rather than to background or historical context ("after the 2015
# earthquake", "unlike the floods of 2021").
MAX_LOOKBACK_DAYS = 400.0

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# Bikram Sambat months. BS->AD is NOT attempted: month lengths vary year to year and need a
# lookup table that must be maintained or it silently drifts. Detected, reported, refused.
_BS_MONTHS = {
    "baishakh", "baisakh", "baishak", "jestha", "jeth", "ashar", "asar", "ashadh",
    "shrawan", "sawan", "shravan", "bhadra", "bhadau", "ashoj", "asoj", "ashwin",
    "kartik", "mangsir", "marga", "poush", "push", "magh", "falgun", "phalgun", "chaitra",
}

# Time of day -> (hours after local midnight, half-width of the window in hours).
_DAYPARTS = {
    "early morning": (5.0, 2.0), "morning": (9.0, 3.0), "midday": (12.0, 1.5),
    "noon": (12.0, 1.0), "afternoon": (15.0, 3.0), "evening": (19.0, 2.5),
    "night": (22.0, 3.0), "midnight": (24.0, 1.5), "overnight": (2.0, 4.0),
    "dawn": (5.5, 1.5), "dusk": (18.5, 1.5),
}

_DAYPART_RE = re.compile(
    r"\b(early\s+morning|overnight|midnight|morning|midday|noon|afternoon|evening|night|dawn|dusk)\b",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"\b(?:at|around|about|roughly)?\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_ALT = "|".join(_WEEKDAYS)
_BS_ALT = "|".join(sorted(_BS_MONTHS, key=len, reverse=True))

# Ordered most specific first. The first rule that matches wins, so "11 October 2026" is
# never mis-read by the year-less rule below it.
_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("iso_date", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("day_month_year", re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I)),
    ("month_day_year", re.compile(
        rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)),
    ("day_month", re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\b\.?(?!\s*\d{{4}})", re.I)),
    ("month_day", re.compile(
        rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!,?\s*\d{{4}})", re.I)),
    ("bikram_sambat", re.compile(rf"\b({_BS_ALT})\s+(\d{{1,2}})\b", re.I)),
    ("days_ago", re.compile(r"\b(\d{1,2}|a|one|two|three|four|five|six|seven)\s+days?\s+ago\b", re.I)),
    ("weeks_ago", re.compile(r"\b(\d{1,2}|a|one|two|three)\s+weeks?\s+ago\b", re.I)),
    ("last_weekday", re.compile(rf"\blast\s+({_WEEKDAY_ALT})\b", re.I)),
    ("weekday", re.compile(
        rf"\b(?:on|since|from)?\s*({_WEEKDAY_ALT})\b", re.I)),
    ("day_before_yesterday", re.compile(r"\bday\s+before\s+yesterday\b", re.I)),
    ("yesterday", re.compile(r"\byesterday\b", re.I)),
    ("today", re.compile(r"\b(?:today|this\s+morning|this\s+afternoon|this\s+evening)\b", re.I)),
    ("overnight", re.compile(r"\bovernight\b|\blast\s+night\b", re.I)),
)

_NUMBER_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7}

# Confidence tiers, strongest first.
_CONFIDENCE_RANK = {
    "explicit_date": 3,
    "relative_resolved": 2,
    "detected_unresolvable": 1,
    "publication_fallback": 0,
}

_RULE_CONFIDENCE = {
    "iso_date": "explicit_date",
    "day_month_year": "explicit_date",
    "month_day_year": "explicit_date",
    "day_month": "explicit_date",
    "month_day": "explicit_date",
    "bikram_sambat": "detected_unresolvable",
    "days_ago": "relative_resolved",
    "weeks_ago": "relative_resolved",
    "last_weekday": "relative_resolved",
    "weekday": "relative_resolved",
    "day_before_yesterday": "relative_resolved",
    "yesterday": "relative_resolved",
    "today": "relative_resolved",
    "overnight": "relative_resolved",
}

# Base uncertainty in days for each rule before a daypart or clock time narrows it.
_RULE_AMBIGUITY = {
    "iso_date": 1.0, "day_month_year": 1.0, "month_day_year": 1.0,
    "day_month": 1.0, "month_day": 1.0,
    "days_ago": 1.5, "weeks_ago": 4.0, "last_weekday": 2.0, "weekday": 1.5,
    "day_before_yesterday": 1.0, "yesterday": 1.0, "today": 1.0, "overnight": 1.0,
}


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentence_of(text: str, start: int, end: int) -> str:
    """The sentence containing [start, end), so a caller can audit the match in context."""
    left = max(text.rfind(". ", 0, start), text.rfind("! ", 0, start),
               text.rfind("? ", 0, start), text.rfind("\n", 0, start))
    left = 0 if left < 0 else left + 1
    right = min([p for p in (text.find(". ", end), text.find("! ", end),
                             text.find("? ", end)) if p != -1] or [len(text)])
    return text[left:min(right + 1, len(text))].strip()


def _local_day(pub_local: datetime, year: int, month: int, day: int) -> Optional[datetime]:
    try:
        return pub_local.replace(year=year, month=month, day=day,
                                 hour=0, minute=0, second=0, microsecond=0)
    except ValueError:  # 31 February and friends
        return None


def _apply_time_of_day(
    day_start_local: datetime, context: str, rule: str
) -> Tuple[datetime, float, str]:
    """Narrow a resolved calendar day using a daypart or clock time in the same sentence."""
    ambiguity = _RULE_AMBIGUITY.get(rule, 1.0)
    detail = "day only"

    clock = _CLOCK_RE.search(context)
    if clock:
        hour = int(clock.group(1))
        minute = int(clock.group(2) or 0)
        meridiem = clock.group(3).lower().replace(".", "")
        if 1 <= hour <= 12 and minute < 60:
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            return (day_start_local + timedelta(hours=hour, minutes=minute),
                    min(ambiguity, 0.1), f"clock time {clock.group(0).strip()}")

    part = _DAYPART_RE.search(context)
    if part:
        key = re.sub(r"\s+", " ", part.group(1).lower())
        offset, half_width = _DAYPARTS.get(key, (12.0, 6.0))
        return (day_start_local + timedelta(hours=offset),
                min(ambiguity, (2 * half_width) / 24.0), f"daypart '{key}'")

    # No time given: place the event mid-day locally so the stated day is not biased to
    # its own boundary, and keep the full-day uncertainty.
    return day_start_local + timedelta(hours=12), ambiguity, detail


def resolve_phrase(
    phrase: str, published_utc: float, *, context: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Turn a date/time phrase into a UTC timestamp, relative to ``published_utc``.

    Pure and deterministic - this is the half of the module an LLM is never allowed to do.
    Returns ``None`` when the phrase resolves to nothing credible.
    """
    phrase = _clean(phrase)
    if not phrase:
        return None
    context = _clean(context) or phrase
    pub_utc = datetime.fromtimestamp(float(published_utc), tz=timezone.utc)
    pub_local = pub_utc.astimezone(NPT)

    for rule, pattern in _PATTERNS:
        m = pattern.search(phrase)
        if not m:
            continue
        day_start: Optional[datetime] = None

        if rule == "iso_date":
            day_start = _local_day(pub_local, int(m.group(1)), int(m.group(2)), int(m.group(3)))
        elif rule in ("day_month_year", "month_day_year"):
            if rule == "day_month_year":
                day, month_name, year = m.group(1), m.group(2), m.group(3)
            else:
                month_name, day, year = m.group(1), m.group(2), m.group(3)
            day_start = _local_day(pub_local, int(year), _MONTHS[month_name.lower()], int(day))
        elif rule in ("day_month", "month_day"):
            if rule == "day_month":
                day, month_name = m.group(1), m.group(2)
            else:
                month_name, day = m.group(1), m.group(2)
            month = _MONTHS[month_name.lower()]
            day_i = int(day)
            # No year stated. Try the publication year, then the one before: an article
            # filed in January about "December 30" means the previous December.
            for year in (pub_local.year, pub_local.year - 1):
                candidate = _local_day(pub_local, year, month, day_i)
                if candidate is None:
                    continue
                if candidate <= pub_local + timedelta(hours=FUTURE_TOLERANCE_HOURS):
                    day_start = candidate
                    break
        elif rule == "bikram_sambat":
            # Detected and reported, deliberately not converted - see _BS_MONTHS.
            return {
                "event_utc": None,
                "confidence": "detected_unresolvable",
                "resolution_method": "bikram_sambat_detected",
                "ambiguity_days": None,
                "matched_rule": rule,
                "matched_text": m.group(0),
                "note": (
                    f"Bikram Sambat date '{m.group(0)}' found. BS->AD conversion needs a "
                    f"per-year month-length table and is not attempted; guessing would be "
                    f"worse than abstaining."
                ),
            }
        elif rule in ("days_ago", "weeks_ago"):
            token = m.group(1).lower()
            n = _NUMBER_WORDS.get(token, None)
            if n is None:
                try:
                    n = int(token)
                except ValueError:
                    continue
            delta = timedelta(days=n * (7 if rule == "weeks_ago" else 1))
            day_start = (pub_local - delta).replace(hour=0, minute=0, second=0, microsecond=0)
        elif rule in ("weekday", "last_weekday"):
            target = _WEEKDAYS[m.group(1).lower()]
            back = (pub_local.weekday() - target) % 7
            if rule == "last_weekday" and back == 0:
                back = 7
            if rule == "weekday" and back == 0:
                back = 0  # "on Sunday" in a Sunday paper means today
            day_start = (pub_local - timedelta(days=back)).replace(
                hour=0, minute=0, second=0, microsecond=0)
        elif rule == "day_before_yesterday":
            day_start = (pub_local - timedelta(days=2)).replace(
                hour=0, minute=0, second=0, microsecond=0)
        elif rule == "yesterday":
            day_start = (pub_local - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0)
        elif rule == "today":
            day_start = pub_local.replace(hour=0, minute=0, second=0, microsecond=0)
        elif rule == "overnight":
            # "overnight"/"last night" spans the boundary; the small hours of pub day.
            day_start = pub_local.replace(hour=0, minute=0, second=0, microsecond=0)

        if day_start is None:
            continue

        event_local, ambiguity, detail = _apply_time_of_day(day_start, context, rule)
        event_utc = event_local.astimezone(timezone.utc)

        # Sanity gates. A phrase that resolves into the future, or implausibly far into the
        # past, is a misparse (a version number, an address, a historical aside).
        ahead_h = (event_utc - pub_utc).total_seconds() / 3600.0
        if ahead_h > FUTURE_TOLERANCE_HOURS:
            logger.debug("rejecting %r: resolves %.1f h after publication", m.group(0), ahead_h)
            continue
        behind_d = (pub_utc - event_utc).total_seconds() / 86400.0
        if behind_d > MAX_LOOKBACK_DAYS:
            logger.debug("rejecting %r: resolves %.0f d before publication", m.group(0), behind_d)
            continue

        return {
            "event_utc": event_utc.timestamp(),
            "confidence": _RULE_CONFIDENCE.get(rule, "relative_resolved"),
            "resolution_method": f"{rule} ({detail})",
            "ambiguity_days": round(ambiguity, 3),
            "matched_rule": rule,
            "matched_text": m.group(0),
            "days_before_publication": round(behind_d, 2),
        }
    return None


# ---------------------------------------------------------------------------
# LLM phrase extraction - a span of text, never a timestamp
# ---------------------------------------------------------------------------
_FORBIDDEN_LLM_KEYS = {
    "event_utc", "timestamp", "epoch", "unix", "unix_time", "iso", "iso_date",
    "datetime", "date", "event_date", "event_time", "resolved", "resolved_date",
    "year", "month", "day", "hour", "minute",
}

_EXTRACTION_PROMPT = """You are reading a Nepali disaster news report to find WHEN the event happened.

Return JSON with exactly these fields:
  phrase     - the VERBATIM words from the report that state when the event occurred,
               copied exactly as written (e.g. "on Sunday morning", "October 11",
               "two days ago"). null if the report never says when it happened.
  sentence   - the full sentence that phrase appears in, copied verbatim. null if none.
  refers_to_event - true if the phrase is the timing of the DISASTER ITSELF; false if it
               is the timing of something else (a rescue operation, a government
               statement, a past disaster mentioned for comparison, a forecast).
  confidence - 0.0 to 1.0, how sure you are.

Rules:
  - Copy the phrase EXACTLY as it appears. Do not reformat, translate or normalise it.
  - DO NOT output a date, a timestamp, a year, or any resolved time. Those fields are
    discarded. Your only job is to point at the words.
  - Ignore the article's publication date. You want the event's time.
  - A sentence about relief efforts or an old disaster is NOT the event time; set
    refers_to_event to false.
  - If the report does not say when it happened, return null. Guessing is worse than
    abstaining.

Report:
---
{text}
---
"""

_EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "phrase": {"type": "STRING", "nullable": True},
        "sentence": {"type": "STRING", "nullable": True},
        "refers_to_event": {"type": "BOOLEAN"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["phrase", "sentence", "refers_to_event", "confidence"],
}


def _strip_llm_timestamps(obj: Any) -> bool:
    """Delete any resolved-time key from a model response. True if anything was removed.

    The hard line, mirroring ``geocoding._strip_llm_coordinates``: the model may point at
    words, it may never supply a time. A fabricated timestamp is indistinguishable from a
    real one once it is downstream, so it is removed before any caller can read it.
    """
    found = False
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if str(key).strip().lower() in _FORBIDDEN_LLM_KEYS:
                obj.pop(key, None)
                found = True
            else:
                found = _strip_llm_timestamps(obj[key]) or found
    elif isinstance(obj, list):
        for item in obj:
            found = _strip_llm_timestamps(item) or found
    return found


def _gemini_extract_phrase(text: str) -> Optional[Dict[str, Any]]:
    """Ask Gemini for the time phrase only. ``None`` -> caller falls back to regex."""
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        logger.debug("GEMINI_API_KEY not set; deterministic extraction only")
        return None
    try:
        import urllib.error
        import urllib.request

        from app.core.config import settings
    except Exception:
        return None

    payload = {
        "contents": [{"role": "user",
                      "parts": [{"text": _EXTRACTION_PROMPT.format(text=text[:12000])}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseSchema": _EXTRACTION_SCHEMA,
        },
    }
    models = [settings.gemini_model] + [
        m for m in getattr(settings, "gemini_fallback_models", [])
        if m != settings.gemini_model
    ]
    base = "https://generativelanguage.googleapis.com/v1beta"
    for model in models:
        try:
            req = urllib.request.Request(
                f"{base}/models/{model}:generateContent?key={api_key}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw)
        except Exception as exc:
            logger.debug("Gemini event-time extraction failed on %s: %s", model, exc)
            continue
        if not isinstance(parsed, dict):
            continue
        if _strip_llm_timestamps(parsed):
            logger.warning(
                "Gemini returned a resolved time for the event-time extraction; "
                "discarded. Only the phrase is used."
            )
        return parsed
    return None


def extract_event_time(
    title: Optional[str],
    body: Optional[str],
    published_utc: float,
    *,
    allow_llm: bool = True,
) -> Dict[str, Any]:
    """When did the event happen? Resolved against ``published_utc``, never ``now()``.

    Always returns a dict. ``confidence`` says how much to trust ``event_utc``:

    ``explicit_date``        the article states a calendar date
    ``relative_resolved``    a relative reference ("Sunday morning") resolved against pubDate
    ``detected_unresolvable``a date was found but cannot be converted (Bikram Sambat)
    ``publication_fallback`` nothing found; ``event_utc`` is just the publication time

    A caller must branch on this. ``publication_fallback`` is explicitly NOT an event time,
    and its wide ``ambiguity_days`` is there so a freshness check cannot quietly treat it
    as one.
    """
    title_c, body_c = _clean(title), _clean(body)
    text = f"{title_c}. {body_c}".strip(". ").strip()
    base: Dict[str, Any] = {
        "event_utc": float(published_utc),
        "published_utc": float(published_utc),
        "confidence": "publication_fallback",
        "evidence_phrase": None,
        "evidence_sentence": None,
        "resolution_method": "no time reference found in title or body",
        "ambiguity_days": None,
        "extractor": "none",
        "llm_refers_to_event": None,
    }
    if not text:
        return base

    # 1. Deterministic sweep over the whole text. The title is scanned first: headlines
    #    carry the event date far more reliably than the body, which is thick with
    #    timings for rescue efforts, statements and past disasters.
    best: Optional[Dict[str, Any]] = None
    for source, chunk in (("title", title_c), ("body", body_c)):
        if not chunk:
            continue
        for rule, pattern in _PATTERNS:
            m = pattern.search(chunk)
            if not m:
                continue
            sentence = _sentence_of(chunk, m.start(), m.end())
            resolved = resolve_phrase(m.group(0), published_utc, context=sentence)
            if resolved is None:
                continue
            resolved.update({
                "evidence_phrase": m.group(0).strip(),
                "evidence_sentence": sentence,
                "extractor": f"regex:{source}",
            })
            rank = _CONFIDENCE_RANK.get(resolved["confidence"], 0)
            if best is None or rank > _CONFIDENCE_RANK.get(best["confidence"], 0):
                best = resolved
            break  # first (most specific) matching rule per chunk
        if best is not None and best["confidence"] == "explicit_date":
            break  # a stated calendar date beats anything the body might offer

    # 2. The model only earns a turn when the deterministic pass found nothing solid, there
    #    is real prose for it to read, and even then it only supplies a phrase for the same
    #    resolver to parse. The body check matters: Google News RSS ships the headline as
    #    the "description" and nothing else (verified - across 100 items, not one
    #    description exceeded its title by 60 characters), so without it the pipeline would
    #    spend an API call per item to re-read a headline the regex already swept.
    has_prose = len(body_c) >= len(title_c) + MIN_BODY_CHARS_FOR_LLM
    if allow_llm and has_prose and (
        best is None or _CONFIDENCE_RANK.get(best["confidence"], 0) < 2
    ):
        llm = _gemini_extract_phrase(text)
        if llm and llm.get("phrase"):
            if llm.get("refers_to_event") is False:
                logger.debug("LLM says the phrase %r is not the event's own timing",
                             llm.get("phrase"))
            else:
                resolved = resolve_phrase(
                    str(llm["phrase"]), published_utc,
                    context=str(llm.get("sentence") or llm["phrase"]),
                )
                if resolved is not None:
                    rank = _CONFIDENCE_RANK.get(resolved["confidence"], 0)
                    if best is None or rank > _CONFIDENCE_RANK.get(best["confidence"], 0):
                        resolved.update({
                            "evidence_phrase": str(llm["phrase"]).strip(),
                            "evidence_sentence": llm.get("sentence"),
                            "extractor": "llm_phrase+deterministic_resolve",
                            "llm_confidence": llm.get("confidence"),
                        })
                        best = resolved

    if best is None:
        return base

    out = dict(base)
    out.update(best)
    out["published_utc"] = float(published_utc)
    if out.get("event_utc") is None:
        # Found but unresolvable (Bikram Sambat). Do NOT pretend the pubDate is the event
        # time while also reporting that a date was detected - say both, plainly.
        out["event_utc"] = float(published_utc)
        out["fell_back_to_publication"] = True
    return out


def is_stale_for_alerting(result: Dict[str, Any], max_age_days: float) -> Dict[str, Any]:
    """Is this event too old to raise a 'prediction confirmed' alarm about?

    The guard the 25 October false alarm needed. Uses the lower edge of the uncertainty
    window, so an ambiguous time is judged by the most recent moment it could be - the
    system should be slow to call something stale, and slow to call something fresh.
    """
    event_utc = result.get("event_utc")
    if event_utc is None:
        return {"stale": True, "reason": "no event time could be determined"}
    ambiguity = result.get("ambiguity_days") or 0.0
    now = datetime.now(tz=timezone.utc).timestamp()
    age_days = (now - float(event_utc)) / 86400.0
    youngest = age_days - (ambiguity / 2.0)

    if result.get("confidence") == "publication_fallback":
        return {
            "stale": youngest > max_age_days,
            "age_days": round(age_days, 2),
            "reason": (
                "no event time in the article; judged on publication date alone, which "
                "is not evidence of when the event happened"
            ),
            "weak_basis": True,
        }
    return {
        "stale": youngest > max_age_days,
        "age_days": round(age_days, 2),
        "youngest_possible_age_days": round(youngest, 2),
        "reason": (
            f"event resolved to {age_days:.1f} d before now via "
            f"{result.get('resolution_method')}"
        ),
        "weak_basis": False,
    }
