"""
witt — Deterministic logic validation for AI agents.

A truth table engine that sits between an agent's decision and
execution. Catches invalid tool calls (missing prerequisites, missing
params, unconfirmed destructive actions) before they run.

    from witt import Supervisor, generate_rules

    engine = generate_rules(tools, dependencies={"summarize": ["search"]})
    gate = Supervisor(engine)

    verdict = gate.check("summarize")
    if verdict:
        result = run_tool("summarize")
        gate.record_success("summarize")
    else:
        agent.retry(verdict.feedback)

Validated: 200/200 valid BFCL multi-turn sequences allowed, 0 false
positives; engine cross-checked against z3 on 1,500 random formulas.
"""

from .engine import TruthTableEngine
from .state import StateTracker
from .supervisor import Supervisor, Verdict, ContradictoryRuleset
from .autogen import (
    generate_rules,
    infer_dependencies_from_traces,
    normalize_bindings,
    action_prop,
    param_prop,
    done_prop,
)

__version__ = "0.1.0"

__all__ = [
    "TruthTableEngine",
    "StateTracker",
    "Supervisor",
    "Verdict",
    "ContradictoryRuleset",
    "generate_rules",
    "infer_dependencies_from_traces",
    "normalize_bindings",
    "action_prop",
    "param_prop",
    "done_prop",
]
