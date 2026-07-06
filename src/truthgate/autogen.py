"""
truthgate.autogen — Generate validation rules from tool specifications.

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

from .engine import TruthTableEngine

# Verbs that indicate a destructive/irreversible operation.
DESTRUCTIVE_HINTS = (
    "delete", "remove", "drop", "destroy", "purge", "erase",
    "send", "post", "publish", "pay", "purchase", "buy", "book",
    "transfer", "cancel", "terminate", "overwrite", "truncate",
)


def action_prop(tool_name: str) -> str:
    return f"Call_{tool_name}"


def param_prop(tool_name: str, param: str) -> str:
    return f"Has_{tool_name}_{param}"


def done_prop(tool_name: str) -> str:
    return f"Done_{tool_name}"


def generate_rules(
    tools: list[dict],
    dependencies: dict[str, list[str]] | None = None,
    require_confirmation: list[str] | None = None,
    auto_detect_destructive: bool = True,
    model_state_space: bool = True,
    engine: TruthTableEngine | None = None,
) -> TruthTableEngine:
    """Build a TruthTableEngine from tool specifications.

    Args:
        tools: list of tool specs (OpenAI/Anthropic/BFCL function format)
        dependencies: {"tool_b": ["tool_a"]} — tool_b requires tool_a done first
        require_confirmation: tool names that need UserConfirmed=True
        auto_detect_destructive: add confirmation rules for tools whose
            name/description contains a destructive verb
        model_state_space: declare the internal relations implied by the
            StateTracker contract — `Done_X` and `Has_X_result` are set
            together, so they are `coupled`. Makes states where only one
            is set impossible (see TruthTableEngine.audit / possibility
            space). Purely additive; leave on unless you set state by hand.
        engine: extend an existing engine instead of creating a new one

    Rule types generated:
        1. Required parameter rules: Call_X → Has_X_param
        2. Dependency rules:         Call_X → Done_Y
        3. Confirmation rules:       Call_X → UserConfirmed
    Space constraints generated (model_state_space):
        4. Completion coupling:      Done_X ↔ Has_X_result
    """
    e = engine or TruthTableEngine()
    e.prop("UserConfirmed", "User confirmed the pending action", "state")

    confirm_set = set(require_confirmation or [])

    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        desc = tool.get("description", "")

        e.prop(action_prop(name), f"Proposed: call {name}", "action")
        e.prop(done_prop(name), f"{name} has completed", "state")

        # A completed tool always has a result and vice versa — the
        # StateTracker sets both on success (colour-exclusion, grounded).
        if model_state_space:
            result_prop = f"Has_{name}_result"
            e.prop(result_prop, f"{name} produced a result", "state")
            e.coupled(done_prop(name), result_prop,
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

        # 2. Auto-detect destructive tools
        if auto_detect_destructive:
            haystack = f"{name} {desc}".lower()
            if any(hint in haystack for hint in DESTRUCTIVE_HINTS):
                confirm_set.add(name)

    # 3. Confirmation rules
    for name in confirm_set:
        e.rule(
            f"{name} requires user confirmation",
            e.IMPLIES(action_prop(name), "UserConfirmed"),
        )

    # 4. Explicit dependencies
    for tool_name, deps in (dependencies or {}).items():
        for dep in deps:
            e.rule(
                f"{tool_name} requires {dep} first",
                e.IMPLIES(action_prop(tool_name), done_prop(dep)),
            )

    return e


def infer_dependencies_from_traces(traces: list[list[str]], min_support: float = 0.95) -> dict:
    """Mine ordering dependencies from execution traces.

    If tool B appears in traces and is *always* (>= min_support of the
    time) preceded by tool A, propose B depends on A.

    Args:
        traces: list of tool-name sequences, e.g. [["auth","query"],["auth","query","report"]]
        min_support: fraction of B-occurrences where A precedes it

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
