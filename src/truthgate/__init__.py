"""
truthgate — Deterministic logic validation for AI agents.

A truth table engine that sits between an agent's decision and
execution. Catches invalid tool calls (missing prerequisites, missing
params, unconfirmed destructive actions) before they run.

    from truthgate import Supervisor, generate_rules

    engine = generate_rules(tools, dependencies={"summarize": ["search"]})
    gate = Supervisor(engine)

    verdict = gate.check("summarize")
    if verdict:
        result = run_tool("summarize")
        gate.record_success("summarize")
    else:
        agent.retry(verdict.feedback)

Validated: 435/435 on BFCL multi-turn benchmark data, 0 false positives.
"""

from .engine import TruthTableEngine
from .state import StateTracker
from .supervisor import Supervisor, Verdict, ContradictoryRuleset
from .autogen import (
    generate_rules,
    infer_dependencies_from_traces,
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
    "action_prop",
    "param_prop",
    "done_prop",
]
