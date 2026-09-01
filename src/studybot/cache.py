"""
Lightweight, dependency-free protection for the free Groq quota:

- AnswerCache: identical (or near-identical, after normalization) questions
  are served from memory instead of calling the LLM again. In a 100-student
  class, many questions repeat almost verbatim ("when is the midterm?"),
  so this alone meaningfully cuts real API calls.

- RateLimiter: a simple sliding-window counter that keeps our own outbound
  request rate under Groq's free-tier cap, so we get a clean "please wait"
  message instead of raw 429 errors bubbling up to students.

Both are process-local (in-memory). That's fine for a single Render free-tier
instance. If you ever scale to multiple instances, swap these for a shared
store (e.g. Redis) instead.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from studybot.config import settings


def normalize_question(question: str) -> str:
    q = question.strip().lower()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


@dataclass
class AnswerCache:
    ttl_seconds: int = field(default_factory=lambda: settings.cache_ttl_seconds)
    _store: dict[str, tuple[float, str]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def get(self, question: str) -> str | None:
        key = normalize_question(question)
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            timestamp, answer = entry
            if time.time() - timestamp > self.ttl_seconds:
                del self._store[key]
                return None
            return answer

    def set(self, question: str, answer: str) -> None:
        key = normalize_question(question)
        with self._lock:
            self._store[key] = (time.time(), answer)


class RateLimiter:
    """Sliding-window limiter: allows at most `max_per_minute` calls in any
    trailing 60-second window.
    """

    def __init__(self, max_per_minute: int | None = None) -> None:
        self.max_per_minute = max_per_minute or settings.max_requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_per_minute:
                return False
            self._timestamps.append(now)
            return True

    def seconds_until_available(self) -> float:
        with self._lock:
            if not self._timestamps:
                return 0.0
            return max(0.0, 60 - (time.time() - self._timestamps[0]))


answer_cache = AnswerCache()
rate_limiter = RateLimiter()
