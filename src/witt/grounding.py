"""
witt.grounding — Deterministic fabricated-argument detection.

    "A name means an object. The object is its meaning." (Tractatus 3.203)

In the picture theory, a proposition only has sense if its names refer
to objects in the world. This module catches exactly that failure: an
argument value that appears nowhere in the agent's world — the user's
request, the tool specifications, the initial configuration, any prior
tool result — refers to nothing. A name without a bearer. Where the
engine validates the logical form of a call, grounding validates its
reference.

Principle: every identifier-like argument (string or number) an agent
passes should be *grounded* — it must appear somewhere the agent could
legitimately have gotten it. A value that appears nowhere in that
corpus was fabricated.

This is not semantic validation. Measured against a rule-blind mutation
adversary judged by execution (see examples/grounding_eval.py):

  * catches ~85% of task-breaking value mutations,
    up from 0% for structure-only checking
  * known false-positive classes: semantic *translations* — the user
    says "November 15th" and the agent correctly writes "2026-11-15"
    (skipped by default); the user names a city and the agent correctly
    writes "LAX" — and *computed* values ("fill the tank" -> capacity
    minus current level). Derived values are legitimate yet appear
    nowhere in the corpus; they are the known blind spot.

Because of that boundary, the recommended mode is "warn": surface the
violation in Verdict.feedback so the agent (or a human) confirms the
value, rather than hard-blocking. Use "strict" only where every
legitimate value provably flows through the corpus.
"""

from __future__ import annotations

import json
import re

# Param names that hold free text the agent legitimately composes —
# message bodies, file content, queries. Never grounded.
DEFAULT_FREEFORM = frozenset((
    "content", "message", "body", "text", "description",
    "query", "notes", "comment", "title",
))

# Value shapes that are usually *derived* rather than copied (dates the
# user stated in prose). Skipped by default to avoid known false positives.
DEFAULT_SKIP_PATTERNS = (re.compile(r"^\d{4}-\d{2}-\d{2}([ T].*)?$"),)


class Grounding:
    """Tracks the corpus of legitimately-available text and checks
    argument values against it.

    Usage:
        g = Grounding(user_text=request, tool_specs=tools, config=cfg)
        gate = Supervisor(engine, grounding=g)          # mode="warn"
        ...
        gate.record_success("search_web", result=response)  # feeds corpus

    Corpus matching is case-insensitive substring containment.
    """

    def __init__(self, user_text: str = "", tool_specs: list | None = None,
                 config=None, mode: str = "warn",
                 freeform_params=DEFAULT_FREEFORM,
                 skip_patterns=DEFAULT_SKIP_PATTERNS,
                 min_len: int = 3, max_len: int = 60,
                 check_numbers: bool = True, min_number: float = 10):
        if mode not in ("warn", "strict"):
            raise ValueError("mode must be 'warn' or 'strict'")
        self.mode = mode
        self.freeform_params = {p.lower() for p in freeform_params}
        self.skip_patterns = tuple(skip_patterns)
        self.min_len, self.max_len = min_len, max_len
        # Numeric grounding: booleans never checked; numbers below
        # min_number are ambient (1, 2, page=5) and skipped. Computed
        # values (totals, conversions) are the known FP class — another
        # reason warn is the default mode.
        self.check_numbers = check_numbers
        self.min_number = min_number
        self._corpus = ""
        if user_text:
            self.observe(user_text)
        for spec in tool_specs or []:
            self.observe(json.dumps(spec))
        if config is not None:
            self.observe(config)

    # ── Corpus ───────────────────────────────────────────────────
    def observe(self, value) -> "Grounding":
        """Add text (or the repr of any object — e.g. a tool result) to
        the corpus of legitimately-available values."""
        text = value if isinstance(value, str) else repr(value)
        # Normalize digit-group commas so "10,000" grounds 10000.
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        self._corpus += " " + text.lower()
        return self

    def is_grounded(self, value: str) -> bool:
        return value.lower() in self._corpus

    def _number_grounded(self, value) -> bool:
        """A number is grounded if it appears in the corpus as a standalone
        number (not inside a longer number like 2107)."""
        forms = {repr(value)}
        if isinstance(value, float) and value.is_integer():
            forms.add(repr(int(value)))
        if isinstance(value, int):
            forms.add(repr(float(value)))
        for f in forms:
            if re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d.])",
                         self._corpus):
                return True
        return False

    # ── Checking ─────────────────────────────────────────────────
    def _checkable(self, param: str, value) -> bool:
        """Identifier-like values only: short single-token strings on
        non-freeform params. Everything else is out of scope by design."""
        return (isinstance(value, str)
                and self.min_len <= len(value) <= self.max_len
                and " " not in value
                and param.lower() not in self.freeform_params
                and not any(p.match(value) for p in self.skip_patterns))

    def ungrounded(self, params: dict | None) -> list[tuple[str, object]]:
        """The (param, value) pairs in a proposed call that are checkable
        but appear nowhere in the corpus — i.e. likely fabricated."""
        out = []
        for p, v in (params or {}).items():
            if self._checkable(p, v) and not self.is_grounded(v):
                out.append((p, v))
            elif (self.check_numbers and not isinstance(v, bool)
                  and isinstance(v, (int, float))
                  and abs(v) >= self.min_number
                  and not self._number_grounded(v)):
                out.append((p, v))
        return out
