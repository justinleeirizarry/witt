"""
witt.supervisor — The gate between agent decision and execution.

One call before every tool execution:

    verdict = gate.check("send_email", params={"to": "...", "body": "..."})
    if verdict.allowed:
        execute()
        gate.record_success("send_email")
    else:
        agent.retry(verdict.feedback)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

from .engine import TruthTableEngine
from .state import StateTracker
from .autogen import action_prop, param_prop, confirm_prop


class ContradictoryRuleset(ValueError):
    """Raised when a Supervisor is built in strict mode over a ruleset
    that no possible world satisfies — every call would be blocked, so
    the ruleset is almost certainly misauthored."""


@dataclass
class Verdict:
    allowed: bool
    violations: list[str] = field(default_factory=list)
    space_violations: list[str] = field(default_factory=list)
    feedback: str = ""
    latency_us: float = 0.0

    def __bool__(self):
        return self.allowed


class Supervisor:
    """Validates proposed tool calls against logical rules + current state.

    Usage:
        from witt import Supervisor, generate_rules

        engine = generate_rules(tools, dependencies={"summarize": ["search"]})
        gate = Supervisor(engine)

        verdict = gate.check("summarize")
        if not verdict:
            print(verdict.feedback)   # "Blocked: summarize requires search first"
    """

    def __init__(self, engine: TruthTableEngine, state: StateTracker | None = None,
                 strict: bool = False):
        self.engine = engine
        self.state = state or StateTracker()
        self.log: list[dict] = []
        self._last_tool: str | None = None  # most recent tool passed to check()
        if strict:
            self._check_ruleset_health()

    def _check_ruleset_health(self):
        """Audit the ruleset at construction. Raise on the fatal case (a
        ruleset no world can satisfy); warn on quality smells that don't
        threaten correctness but signal an authoring mistake."""
        report = self.engine.audit()
        fatal = []
        for core in report["conflicts"]:
            fatal.append("contradictory rules {" + ", ".join(core) + "}")
        for space in report["impossible_space"]:
            fatal.append("self-contradictory space {" + ", ".join(space) + "}")
        if fatal:
            raise ContradictoryRuleset("; ".join(fatal))

        smells = []
        if report["vacuous"]:
            smells.append(f"vacuous (say nothing): {report['vacuous']}")
        if report["redundant"]:
            smells.append(f"redundant: {[r['rule'] for r in report['redundant']]}")
        if report["equivalent"]:
            smells.append(f"duplicate sense: {report['equivalent']}")
        if smells:
            warnings.warn("Ruleset audit found " + "; ".join(smells),
                          stacklevel=3)

    def audit(self) -> dict:
        """Health report on the underlying ruleset — vacuous, redundant,
        equivalent, and contradictory rules. See TruthTableEngine.audit."""
        return self.engine.audit()

    # ── The gate ─────────────────────────────────────────────────
    def check(self, tool_name: str, params: dict | None = None) -> Verdict:
        """Validate a proposed tool call against rules and current state.

        Args:
            tool_name: the tool the agent wants to call
            params: the parameters it's passing (presence of a required
                param sets its Has_ prop automatically)

        Note: a param is treated as supplied when its value is not None. A
        None value counts as "not supplied" (a required param passed as
        None will still be flagged); other falsy values ("", 0, False) do
        count as supplied.
        """
        t0 = time.perf_counter()
        self._last_tool = tool_name

        props = self.state.snapshot()
        props[action_prop(tool_name)] = True

        # Presence of params satisfies the Has_ props for this call
        for p, v in (params or {}).items():
            if v is not None:
                props[param_prop(tool_name, p)] = True

        result = self.engine.validate_closed(props)
        latency = (time.perf_counter() - t0) * 1_000_000

        space_violations = result.get("space_violations", [])
        msgs = []
        if result["violations"]:
            msgs.append("Blocked: " + "; ".join(result["violations"]))
        if space_violations:
            # The state itself is impossible — not a forbidden action, an
            # incoherent world. Distinct signal, distinct fix.
            msgs.append("Incoherent state: " + "; ".join(space_violations))

        verdict = Verdict(
            allowed=result["valid"],
            violations=result["violations"],
            space_violations=space_violations,
            feedback=" | ".join(msgs),
            latency_us=latency,
        )

        self.log.append({
            "tool": tool_name,
            "allowed": verdict.allowed,
            "violations": verdict.violations,
            "latency_us": latency,
        })
        return verdict

    # ── State updates ────────────────────────────────────────────
    def record_success(self, tool_name: str, extra_state: dict | None = None):
        """Call after a tool executes successfully. Also *consumes* any
        confirmation for this tool, so the next destructive call must be
        confirmed afresh rather than riding a stale approval."""
        self.state.on_tool_success(tool_name, extra_state)
        self.state.set(confirm_prop(tool_name), False)
        return self

    def record_failure(self, tool_name: str):
        self.state.on_tool_failure(tool_name)
        return self

    def _confirm_target(self, tool_name: str | None) -> str:
        target = tool_name or self._last_tool
        if target is None:
            raise ValueError(
                "confirm() needs a tool name — nothing has been checked yet")
        return target

    def confirm(self, tool_name: str | None = None):
        """Record that the user confirmed a specific destructive action.

        Confirmation is per-tool: confirming `delete_record` does not
        authorize `send_email`. With no argument, confirms the most
        recently checked tool."""
        self.state.set(confirm_prop(self._confirm_target(tool_name)), True)
        return self

    def unconfirm(self, tool_name: str | None = None):
        """Clear a tool's confirmation. record_success already consumes it
        after a successful call; use this to revoke a pending one."""
        self.state.set(confirm_prop(self._confirm_target(tool_name)), False)
        return self

    # ── Introspection ────────────────────────────────────────────
    def stats(self) -> dict:
        if not self.log:
            return {"checks": 0}
        blocked = sum(1 for x in self.log if not x["allowed"])
        latencies = [x["latency_us"] for x in self.log]
        return {
            "checks": len(self.log),
            "allowed": len(self.log) - blocked,
            "blocked": blocked,
            "avg_latency_us": sum(latencies) / len(latencies),
            "max_latency_us": max(latencies),
        }
