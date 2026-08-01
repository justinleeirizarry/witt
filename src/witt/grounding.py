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

The corpus is kept as a set of named *sources* rather than one blob, so
grounding answers "where did this value come from?" and not merely
"does it occur anywhere?". Two things follow:

  * Every check reports provenance: the source ids that contain the
    value (Verdict.provenance).
  * A `scopes` declaration can require a parameter to come from a
    *specific* source — `send_email(to=...)` must be grounded in a
    `lookup_contact` result, not in the user's prose or in some
    unrelated tool's output. Without it, any source grounds any value,
    which is the weaker guarantee.

A declared scope is an authored constraint rather than a heuristic, so
scope violations block in either mode (like argument bindings). Plain
ungroundedness stays heuristic and follows `mode`.

This is not semantic validation. Measured against a rule-blind mutation
adversary judged by execution (see examples/grounding_eval.py):

  * catches ~85% of task-breaking value mutations,
    up from 0% for structure-only checking
  * a *real* value drawn from the wrong source is the case flat grounding
    is weakest on — ~0.24 recall, since the value does occur in the corpus.
    Scopes are the answer; mined ones lift that to ~0.33 at the permissive
    default and ~0.49 when tightened, trading false positives as they go
    (examples/scoped_grounding_eval.py). Declared scopes make no such
    trade.
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
from collections import defaultdict
from dataclasses import dataclass

# Param names that hold free text the agent legitimately composes —
# message bodies, file content, queries. Never grounded.
DEFAULT_FREEFORM = frozenset((
    "content", "message", "body", "text", "description",
    "query", "notes", "comment", "title",
))

# Value shapes that are usually *derived* rather than copied (dates the
# user stated in prose). Skipped by default to avoid known false positives.
DEFAULT_SKIP_PATTERNS = (re.compile(r"^\d{4}-\d{2}-\d{2}([ T].*)?$"),)

# Reserved source ids for the corpus available before any tool has run.
# Specs and config are subdivided — "specs:get_order", "config:filesystem" —
# so provenance names the individual schema or config section.
SOURCE_USER = "user"
SOURCE_SPECS = "specs"
SOURCE_CONFIG = "config"

# Default id for observe() calls that don't name a source.
SOURCE_OBSERVED = "observed"

# Separates a tool source from its call ordinal: "get_order#2". Scope
# matching and mining both work on the base name, so a scope naming
# "get_order" is satisfied by any of its calls.
CALL_SEP = "#"


def base_source(source_id: str) -> str:
    """The source id without its call ordinal — 'get_order#2' -> 'get_order'."""
    return source_id.split(CALL_SEP, 1)[0]


@dataclass(frozen=True)
class GroundingViolation:
    """A parameter value that fails grounding.

    kind is one of:
      "ungrounded"   — the value appears in no source at all (heuristic;
                       follows Grounding.mode)
      "out_of_scope" — the value appears, but only in sources this
                       parameter is not allowed to draw from (authored
                       constraint; always blocks)
    """
    param: str
    value: object
    kind: str
    found_in: tuple = ()
    expected: tuple = ()

    def message(self, tool: str) -> str:
        if self.kind == "out_of_scope":
            return (f"{tool}({self.param}={self.value!r}): value comes from "
                    f"{list(self.found_in)}, but must be grounded in "
                    f"{list(self.expected)}")
        return (f"{tool}({self.param}={self.value!r}): value appears nowhere "
                f"in user request, tool specs, or prior results — possibly "
                f"fabricated")


class Grounding:
    """Tracks the corpus of legitimately-available text and checks
    argument values against it.

    Usage:
        g = Grounding(user_text=request, tool_specs=tools, config=cfg)
        gate = Supervisor(engine, grounding=g)          # mode="warn"
        ...
        gate.record_success("search_web", result=response)  # feeds corpus

    Scoped usage — the recipient must come from a contact lookup rather
    than from anywhere the string happens to appear:

        g = Grounding(user_text=request,
                      scopes={"send_email": {"to": ["lookup_contact"]}})

    Corpus matching is case-insensitive substring containment.
    """

    def __init__(self, user_text: str = "", tool_specs: list | None = None,
                 config=None, mode: str = "warn",
                 freeform_params=DEFAULT_FREEFORM,
                 skip_patterns=DEFAULT_SKIP_PATTERNS,
                 min_len: int = 3, max_len: int = 60,
                 check_numbers: bool = True, min_number: float = 10,
                 scopes: dict | None = None,
                 track_provenance: bool = True):
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
        # {tool: {param: (source, ...)}}; tool "*" applies to every tool.
        self.scopes = normalize_scopes(scopes)
        self.track_provenance = track_provenance
        # source id -> accumulated lowercased text. Insertion-ordered, so
        # provenance reads in the order the agent acquired the material.
        self._sources: dict[str, str] = {}
        # tool -> how many results it has contributed, for call ordinals.
        self._call_counts: dict[str, int] = defaultdict(int)
        if user_text:
            self.observe(user_text, source=SOURCE_USER)
        # One source per tool spec and per config section rather than two
        # pooled blobs: an enum value belongs to the tool that declares it,
        # and a scope of "specs:get_order" should not be satisfied by a
        # value that only appears in some other tool's schema.
        for spec in tool_specs or []:
            name = spec.get("name") if isinstance(spec, dict) else None
            self.observe(json.dumps(spec),
                         source=f"{SOURCE_SPECS}:{name}" if name
                         else SOURCE_SPECS)
        if isinstance(config, dict) and config:
            for section, value in config.items():
                self.observe(value, source=f"{SOURCE_CONFIG}:{section}")
        elif config is not None:
            self.observe(config, source=SOURCE_CONFIG)

    # ── Corpus ───────────────────────────────────────────────────
    def observe(self, value, source: str | None = None) -> "Grounding":
        """Add text (or the repr of any object — e.g. a tool result) to
        the corpus of legitimately-available values.

        `source` names where the material came from; it is what provenance
        reports and what `scopes` constrains against. Repeated observations
        under one id accumulate into that source."""
        text = value if isinstance(value, str) else repr(value)
        # Normalize digit-group commas so "10,000" grounds 10000.
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        sid = source or SOURCE_OBSERVED
        self._sources[sid] = self._sources.get(sid, "") + " " + text.lower()
        return self

    def observe_result(self, tool: str, result) -> str:
        """Record a tool result as its own source, returning the source id.

        Each call gets an ordinal (`get_order#1`, `get_order#2`) so
        provenance identifies which call supplied a value; scope matching
        and mining both fold ordinals back to the base tool name."""
        self._call_counts[tool] += 1
        sid = f"{tool}{CALL_SEP}{self._call_counts[tool]}"
        self.observe(result, source=sid)
        return sid

    def sources(self) -> list[str]:
        """Source ids currently in the corpus, in acquisition order."""
        return list(self._sources)

    # ── Matching ─────────────────────────────────────────────────
    def _match(self, value, text: str) -> bool:
        """Does `value` occur in this text segment?"""
        if isinstance(value, str):
            return value.lower() in text
        # A number is grounded if it appears as a standalone number (not
        # inside a longer number like 2107).
        forms = {repr(value)}
        if isinstance(value, float) and value.is_integer():
            forms.add(repr(int(value)))
        if isinstance(value, int):
            forms.add(repr(float(value)))
        return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d.])", text)
                   for f in forms)

    def is_grounded(self, value, sources=None) -> bool:
        """Does the value appear in the corpus (optionally restricted to
        `sources`, matched by base name)? Short-circuits on the first hit."""
        allowed = None if sources is None else tuple(sources)
        for sid, text in self._sources.items():
            if allowed is not None and not _in_scope(sid, allowed):
                continue
            if self._match(value, text):
                return True
        return False

    def where_grounded(self, value) -> list[str]:
        """Every source id whose text contains the value — the provenance
        of an argument. Empty means fabricated."""
        return [sid for sid, text in self._sources.items()
                if self._match(value, text)]

    # ── Checking ─────────────────────────────────────────────────
    def _checkable(self, param: str, value) -> bool:
        """Identifier-like values only: short single-token strings on
        non-freeform params, and numbers large enough not to be ambient.
        Everything else is out of scope by design."""
        if isinstance(value, str):
            return (self.min_len <= len(value) <= self.max_len
                    and " " not in value
                    and param.lower() not in self.freeform_params
                    and not any(p.match(value) for p in self.skip_patterns))
        return (self.check_numbers and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and abs(value) >= self.min_number)

    def scope_for(self, tool: str | None, param: str) -> tuple | None:
        """The sources `param` is allowed to draw from on this tool, or
        None when unconstrained. A tool-specific entry wins over "*"."""
        for key in (tool, "*"):
            if key is not None and key in self.scopes:
                scope = self.scopes[key].get(param)
                if scope is not None:
                    return scope
        return None

    def check(self, tool: str | None, params: dict | None
              ) -> tuple[list[GroundingViolation], dict[str, list[str]]]:
        """Check a proposed call's arguments.

        Returns (violations, provenance) where provenance maps each
        checkable param to the source ids that ground it. Provenance is
        skipped (left empty) when track_provenance is off, which restores
        the short-circuiting single-pass check for latency-critical use."""
        violations: list[GroundingViolation] = []
        provenance: dict[str, list[str]] = {}
        for param, value in (params or {}).items():
            if not self._checkable(param, value):
                continue
            scope = self.scope_for(tool, param)
            if scope is None and not self.track_provenance:
                if not self.is_grounded(value):
                    violations.append(
                        GroundingViolation(param, value, "ungrounded"))
                continue
            found = self.where_grounded(value)
            provenance[param] = found
            if not found:
                violations.append(
                    GroundingViolation(param, value, "ungrounded"))
            elif scope is not None and not any(_in_scope(s, scope)
                                               for s in found):
                violations.append(GroundingViolation(
                    param, value, "out_of_scope",
                    found_in=tuple(found), expected=tuple(scope)))
        return violations, provenance

    def ungrounded(self, params: dict | None, tool: str | None = None
                   ) -> list[tuple[str, object]]:
        """The (param, value) pairs in a proposed call that fail grounding
        — checkable but appearing nowhere they legitimately could have come
        from. See `check` for the reason each one failed."""
        return [(v.param, v.value) for v in self.check(tool, params)[0]]


def _in_scope(source_id: str, allowed) -> bool:
    """Scope entries name a source without its call ordinal, so
    'get_order' admits 'get_order#1' and 'get_order#2' alike."""
    return source_id in allowed or base_source(source_id) in allowed


def normalize_scopes(scopes: dict | None) -> dict:
    """Canonicalize the `scopes` argument to {tool: {param: (source, ...)}}.

    Accepts a bare string in place of a one-element list:
        {"send_email": {"to": "lookup_contact"}}
        {"send_email": {"to": ["lookup_contact", "user"]}}
        {"*":          {"account_id": ["get_accounts"]}}
    """
    out: dict[str, dict[str, tuple]] = {}
    for tool, params in (scopes or {}).items():
        norm = {p: (s,) if isinstance(s, str) else tuple(s)
                for p, s in (params or {}).items()}
        if norm:
            out[tool] = norm
    return out


def infer_scopes_from_logs(logs, min_support: float = 0.0,
                           min_occurrences: int = 3) -> dict:
    """Mine parameter scopes from the provenance recorded on *valid* runs.

    Each argument is scoped to the sources that were ever seen to ground
    it legitimately, so a later value drawn from a source that never
    supplies this parameter is flagged — the fabrication case where the
    value happens to occur *somewhere* in the corpus and so survives a
    plain grounding check.

    CAUTION — like infer_dependencies_from_traces, this is correlation.
    A source that simply never came up in the sample is excluded from the
    scope and will produce false positives when it legitimately appears.
    Mine from runs you know to be correct, over enough of them to have
    seen the normal provenance of each argument, and review the output.

    Args:
        logs: an iterable of Supervisor.log lists (one per run), or a
            single flat log. Entries need the "provenance" key, so the
            runs must have been gated with track_provenance on.
        min_support: keep only sources grounding the param in at least
            this fraction of its occurrences. 0.0 (default) keeps every
            source ever observed — the permissive, lowest-FP choice.
            Raising it tightens the scope and trades FPs for recall.
        min_occurrences: ignore params seen fewer than this many times.

    Returns:
        {tool: {param: [source, ...]}} — pass to Grounding(scopes=...).
    """
    runs = list(logs)
    if runs and isinstance(runs[0], dict):  # a single flat log
        runs = [runs]

    counts: dict[tuple, int] = defaultdict(int)
    per_source: dict[tuple, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    for run in runs:
        for entry in run:
            tool = entry.get("tool")
            for param, found in (entry.get("provenance") or {}).items():
                if not found:
                    continue  # ungrounded here; nothing to learn about source
                counts[(tool, param)] += 1
                for sid in {base_source(s) for s in found}:
                    per_source[(tool, param)][sid] += 1

    out: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for (tool, param), total in counts.items():
        if total < min_occurrences:
            continue
        keep = sorted(s for s, n in per_source[(tool, param)].items()
                      if n / total >= min_support)
        if keep:
            out[tool][param] = keep
    return dict(out)
