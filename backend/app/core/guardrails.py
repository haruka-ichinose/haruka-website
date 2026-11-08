"""Local, dependency-free guardrails for /chat.

These run on the Pi before any paid call and work even when Azure AI Content
Safety is unconfigured or down (it fails open):

- ``trim_history`` — the client sends the whole conversation, including
  "assistant" turns it could have forged, so only the last few turns are kept
  and each is length-capped. Bounds token cost and many-shot jailbreaks.
- ``looks_like_jailbreak`` — a short, high-precision pattern list for the most
  common instruction-override phrasings. Prompt Shields is the real detector;
  this is the floor when it isn't available.
- ``RateLimiter`` — in-memory sliding windows per client plus a global daily
  ceiling, so a single visitor (or a script) can't run up the Azure bill.
"""

import re
import time
from collections import deque

from models.chat import ChatMessage

MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_ITEM_CHARS = 2000

_JAILBREAK_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b.{0,20}\b(instructions?|rules|prompts?|guidelines|directives)\b",
    r"\b(reveal|print|show|repeat|output|leak|dump)\b.{0,40}\b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b",
    r"\byou\s+are\s+now\b.{0,40}\b(dan|jailbroken|unfiltered|uncensored|unrestricted|evil)\b",
    r"\b(developer|god|jailbreak|dan)\s+mode\b",
    r"\bdo\s+anything\s+now\b",
    r"</?\s*(system|context|instructions?)\s*>",
]
_JAILBREAK_RE = re.compile("|".join(_JAILBREAK_PATTERNS), re.IGNORECASE | re.DOTALL)


def trim_history(history: list[ChatMessage]) -> list[ChatMessage]:
    """Keep the most recent turns, each truncated to a fixed length."""
    return [
        ChatMessage(role=m.role, content=m.content[:MAX_HISTORY_ITEM_CHARS])
        for m in history[-MAX_HISTORY_MESSAGES:]
    ]


def looks_like_jailbreak(text: str) -> bool:
    return bool(_JAILBREAK_RE.search(text))


class RateLimiter:
    """Sliding-window limits keyed by client, plus one global daily cap.

    In-memory, so it assumes a single process (the Pi runs one uvicorn worker)
    and resets on restart — fine as a cost guard, not as an audit log.
    """

    def __init__(self, *, per_minute: int, per_day: int, global_per_day: int) -> None:
        self._limits = ((60, per_minute), (86400, per_day))
        self._global_per_day = global_per_day
        self._hits: dict[str, deque[float]] = {}
        self._global: deque[float] = deque()

    def allow(self, client: str, now: float | None = None) -> bool:
        """Record and allow a request, or return False if any limit is hit."""
        now = time.monotonic() if now is None else now
        hits = self._hits.setdefault(client, deque())
        for q in (hits, self._global):
            while q and q[0] <= now - 86400:
                q.popleft()

        for window, limit in self._limits:
            if sum(1 for t in hits if t > now - window) >= limit:
                return False
        if len(self._global) >= self._global_per_day:
            return False

        hits.append(now)
        self._global.append(now)
        # Drop idle clients so the dict can't grow without bound.
        if len(self._hits) > 10_000:
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - 86400}
        return True
