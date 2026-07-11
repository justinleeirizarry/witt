"""
witt.autogen — Generate validation rules from tool specifications.

The key insight from BFCL testing: tool specs already contain the
constraints. Required parameters, dependencies, destructive flags —
all of it converts mechanically into truth table rules. You don't
write rules by hand; you generate them from the tool definitions
your agent already has.

Supports the JSON-schema function format used by OpenAI, Anthropic,
and BFCL:

    {
        "name": "get_weather",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
"""

from __future__ import annotations

import re

from .engine import TruthTableEngine

# Verbs that indicate a destructive/irreversible operation.
DESTRUCTIVE_HINTS = frozenset((
    "delete", "remove", "drop", "destroy", "purge", "erase",
    "send", "post", "publish", "pay", "purchase", "buy", "book",
    "transfer", "cancel", "terminate", "overwrite", "truncate",
))

# camelCase / snake_case / kebab boundaries → individual tokens.
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def _name_tokens(name: str) -> set:
    """Split a tool name into lowercase word tokens (handles snake_case,
    camelCase, kebab-case)."""
    return {t.lower() for t in _TOKEN_SPLIT.split(name) if t}


def is_destructive(name: str, hints=DESTRUCTIVE_HINTS) -> bool:
    """Whole-token match on the tool NAME only. Avoids the substring false
    positives of the old approach — 'pay' no longer fires on 'payload' or
    'payment_status', 'book' not on 'bookmark', 'post' not on 'postal_code'
    — and ignores the description, so 'does not delete anything' is safe."""
    return bool(_name_tokens(name) & set(hints))


def action_prop(tool_name: str) -> str:
    return f"Call_{tool_name}"


def param_prop(tool_name: str, param: str) -> str:
    # '::' separates tool from param so names can't alias — Has_a::b_c and
    # Has_a_b::c are distinct, where the old Has_a_b_c form collided. '::'
    # can't occur in a normal tool/param identifier.
    return f"Has_{tool_name}::{param}"


def done_prop(tool_name: str) -> str:
    return f"Done_{tool_name}"


def result_prop(tool_name: str) -> str:
    # Distinct 'Result_' prefix so a param literally named 'result'
    # (Has_{tool}::result) can't collide with the completion prop.
    return f"Result_{tool_name}"


def confirm_prop(tool_name: str) -> str:
    # Per-tool confirmation — confirming one destructive action does not
    # authorize a different one.
    return f"Confirmed_{tool_name}"


def normalize_bindings(bindings: dict | None) -> dict:
    """Normalize the `bindings` argument into a canonical form.

    A binding says: calling `tool` with argument `param` requires that a
    prior `dep` call *completed with the matching value*. This is object
    identity rather than logical structure — so it is enforced by the
    Supervisor layer instead of the boolean engine (which reasons about
    structure only).

    `param` names the argument on the *calling* tool; `dep_param` names the
    argument the dependency must have matched (defaults to `param` when the
    two tools use the same name for the shared value).

    Accepted input forms per entry (values of the top-level dict are lists):
        {"tool": "tool_a", "param": "id"}            # dep_param defaults to param
        {"tool": "tool_a", "param": "id", "dep_param": "key"}
        ("tool_a", "id")                             # tuple shorthand
        ("tool_a", "id", "key")                      # (dep, param, dep_param)

    Returns: {caller_tool: [{"dep", "param", "dep_param"}, ...]}
    """
    out: dict[str, list[dict]] = {}
    for tool, entries in (bindings or {}).items():
        norm: list[dict] = []
        for e in entries:
            if isinstance(e, dict):
                dep = e["tool"]
                param = e["param"]
                dep_param = e.get("dep_param", param)
            else:  # tuple/list form
                dep, param = e[0], e[1]
                dep_param = e[2] if len(e) > 2 else param
            norm.append({"dep": dep, "param": param, "dep_param": dep_param})
        if norm:
            out[tool] = norm
    return out


def generate_rules(
    tools: list[dict],
    dependencies: dict[str, list[str]] | None = None,
    require_confirmation: list[str] | None = None,
    auto_detect_destructive: bool = True,
    model_state_space: bool = True,
    bindings: dict | None = None,
    engine: TruthTableEngine | None = None,
) -> TruthTableEngine:
    """Build a TruthTableEngine from tool specifications.

    Args:
        tools: list of tool specs (OpenAI/Anthropic/BFCL function format)
        dependencies: {"tool_b": ["tool_a"]} — tool_b requires tool_a done first
        bindings: argument-bound dependencies — {"tool_b": [{"tool":
            "tool_a", "param": "id"}]} means tool_b called with id=V requires
            a tool_a that *completed with id=V*, beyond merely requiring
            that some tool_a
            ran. This enforces object identity across steps, which the boolean
            engine cannot; it is stored on the engine and enforced by the
            Supervisor. A bound dependency also implies ordering (the dep must
            have completed), so it subsumes the plain `dependencies` form.
            See normalize_bindings for accepted entry shapes.
        require_confirmation: tool names that require confirmation before
            running (each gets a per-tool Confirmed_X gate)
        auto_detect_destructive: add confirmation rules for tools whose
            name contains a destructive verb as a whole token (see
            is_destructive)
        model_state_space: declare the internal relations implied by the
            StateTracker contract — `Done_X` and `Has_X_result` are set
            together, so they are `coupled`. Makes states where only one
            is set impossible (see TruthTableEngine.audit / possibility
            space). Purely additive; leave on unless you set state by hand.
        engine: extend an existing engine instead of creating a new one

    Rule types generated:
        1. Required parameter rules: Call_X → Has_X::param
        2. Dependency rules:         Call_X → Done_Y
        3. Confirmation rules:       Call_X → Confirmed_X  (per-tool)
    Space constraints generated (model_state_space):
        4. Completion coupling:      Done_X ↔ Result_X
    """
    e = engine or TruthTableEngine()

    confirm_set = set(require_confirmation or [])

    for tool in tools:
        name = tool.get("name")
        if not name:
            continue

        e.prop(action_prop(name), f"Proposed: call {name}", "action")
        e.prop(done_prop(name), f"{name} has completed", "state")

        # A completed tool always has a result and vice versa — the
        # StateTracker sets both on success (colour-exclusion, grounded).
        if model_state_space:
            rp = result_prop(name)
            e.prop(rp, f"{name} produced a result", "state")
            e.coupled(done_prop(name), rp,
                      name=f"{name}: completed iff has result")

        # 1. Required parameters
        params = tool.get("parameters", {}) or {}
        for param in params.get("required", []) or []:
            pp = param_prop(name, param)
            e.prop(pp, f"{name} has required param '{param}'", "state")
            e.rule(
                f"{name} requires param '{param}'",
                e.IMPLIES(action_prop(name), pp),
            )

        # 2. Auto-detect destructive tools (whole-token match on the name)
        if auto_detect_destructive and is_destructive(name):
            confirm_set.add(name)

    # 3. Confirmation rules — per-tool, so confirming one destructive
    #    action never authorizes another.
    for name in confirm_set:
        cp = confirm_prop(name)
        e.prop(cp, f"user confirmed {name}", "state")
        e.rule(
            f"{name} requires user confirmation",
            e.IMPLIES(action_prop(name), cp),
        )

    # 4. Explicit dependencies
    for tool_name, deps in (dependencies or {}).items():
        for dep in deps:
            e.rule(
                f"{tool_name} requires {dep} first",
                e.IMPLIES(action_prop(tool_name), done_prop(dep)),
            )

    # 5. Argument-bound dependencies (object identity across steps).
    #    Not expressible in the boolean engine — the required value is only
    #    known at call time — so they ride along on the engine as metadata
    #    and are enforced by the Supervisor. Stored merged so repeated
    #    generate_rules(engine=e, ...) calls accumulate rather than clobber.
    merged = dict(getattr(e, "bindings", {}) or {})
    merged.update(normalize_bindings(bindings))
    e.bindings = merged

    return e


def infer_dependencies_from_traces(traces: list[list[str]], min_support: float = 0.95) -> dict:
    """Mine ordering dependencies from execution traces.

    If tool B is preceded by tool A in at least `min_support` of B's
    occurrences, propose B depends on A.

    CAUTION — this captures correlation rather than causation. "A usually precedes B"
    does not mean B *requires* A; habitual sequencing and common causes
    produce spurious dependencies (e.g. `cp requires cd`). Dependencies
    mined this way can introduce false positives when used as hard gates
    (see examples/oracle_eval.py). Review before trusting them, and note
    the default threshold is 0.95 rather than 1.0 — a proposed dependency may be
    violated by up to 1 in 20 traces.

    Args:
        traces: list of tool-name sequences, e.g. [["auth","query"],["auth","query","report"]]
        min_support: fraction of B-occurrences where A precedes it (default 0.95)

    Returns:
        {"tool_b": ["tool_a", ...]}
    """
    from collections import defaultdict

    occurrences: dict[str, int] = defaultdict(int)
    preceded_by: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for trace in traces:
        seen: set[str] = set()
        for tool in trace:
            occurrences[tool] += 1
            for prior in seen:
                preceded_by[tool][prior] += 1
            seen.add(tool)

    deps: dict[str, list[str]] = {}
    for tool, priors in preceded_by.items():
        strong = [
            p for p, count in priors.items()
            if count / occurrences[tool] >= min_support and occurrences[tool] >= 3
        ]
        if strong:
            deps[tool] = sorted(strong)
    return deps
