"""
witt MCP server — a live playground for the gate.

Exposes witt as MCP tools so an agent (or a human via an MCP client) can
walk a real multi-step scenario and watch the gate accept or block each
call, with the feedback that would go back into an LLM's context.

This is a DEMO surface. In a real deployment witt is not a tool the model
chooses to call; it's middleware the harness runs automatically before
every tool execution. Here we expose it as callable tools so the loop can
be exercised and the feedback messages inspected.

Run directly (stdio transport):
    .venv/bin/python mcp_server.py

Register with Claude Code (example .mcp.json / settings):
    {
      "mcpServers": {
        "witt": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["/absolute/path/to/mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from witt import Supervisor, generate_rules, ContradictoryRuleset

mcp = FastMCP("witt")

# ── Session state (single stateful gate for the playground) ──────────
_STATE: dict = {"gate": None, "tools": []}


def _require_gate() -> Supervisor:
    if _STATE["gate"] is None:
        raise ValueError(
            "No engine configured. Call `configure` first with your tool specs.")
    return _STATE["gate"]


@mcp.tool()
def configure(
    tools: list[dict],
    dependencies: dict[str, list[str]] | None = None,
    require_confirmation: list[str] | None = None,
    bindings: dict | None = None,
    strict: bool = False,
) -> dict:
    """Build a witt engine from tool specs and start a fresh gate session.

    Args:
        tools: JSON-schema tool specs (the same ones you give your agent),
            e.g. [{"name": "send_email", "description": "...",
                   "parameters": {"type": "object",
                                  "properties": {"to": {"type": "string"}},
                                  "required": ["to"]}}]
        dependencies: {"tool_b": ["tool_a"]} — tool_b needs tool_a done first.
        bindings: argument-bound dependencies — object identity across steps.
            {"tool_b": [{"tool": "tool_a", "param": "id"}]} means tool_b with
            id=V requires a tool_a that *completed with id=V*, not merely that
            some tool_a ran. Enforced by the gate (not the boolean engine). A
            bound dependency also implies ordering, so you don't also need it
            in `dependencies`.
        require_confirmation: tool names that require confirmation before use
            (destructive tools are also auto-detected from their names).
        strict: if true, refuse to build a contradictory / vacuous-heavy
            ruleset and report why.

    Returns a summary of the generated rules and the possibility space.
    """
    try:
        engine = generate_rules(
            tools,
            dependencies=dependencies,
            require_confirmation=require_confirmation,
            bindings=bindings,
        )
        gate = Supervisor(engine, strict=strict)
    except ContradictoryRuleset as ex:
        return {"ok": False, "error": f"Contradictory ruleset: {ex}"}

    _STATE["gate"] = gate
    _STATE["tools"] = [t.get("name") for t in tools if t.get("name")]
    return {
        "ok": True,
        "tools": _STATE["tools"],
        "rules": [r["name"] for r in engine.rules],
        "constraints": [c["name"] for c in engine.constraints],
        "bindings": engine.bindings,
        "rule_count": len(engine.rules),
    }


@mcp.tool()
def check(tool: str, params: dict | None = None) -> dict:
    """Validate a proposed tool call against the rules and current state,
    WITHOUT executing it. This is the gate.

    Returns allowed (bool), the feedback string an LLM would receive on a
    block, any violated rules, any incoherent-state (space) violations, and
    the latency in microseconds. Call `record_success` after you actually
    run an allowed tool.
    """
    gate = _require_gate()
    v = gate.check(tool, params=params or {})
    return {
        "allowed": v.allowed,
        "feedback": v.feedback,
        "violations": v.violations,
        "binding_violations": v.binding_violations,
        "space_violations": v.space_violations,
        "latency_us": round(v.latency_us, 1),
    }


@mcp.tool()
def record_success(tool: str, extra_state: dict | None = None,
                   params: dict | None = None) -> dict:
    """Record that a tool executed successfully. Sets Done_<tool> and its
    result, and consumes any confirmation for that tool. Optional
    extra_state sets additional facts (e.g. {"HasUserLocation": true}).

    Optional `params` records the argument values the call ran with, so
    argument-bound dependencies can match on them later. If omitted, the
    params from this tool's most recent check() are reused."""
    gate = _require_gate()
    gate.record_success(tool, extra_state or None, params)
    return {"ok": True, "state": gate.state.snapshot()}


@mcp.tool()
def confirm(tool: str | None = None) -> dict:
    """Record a user confirmation for a destructive tool (per-tool). With no
    argument, confirms the most recently checked tool."""
    gate = _require_gate()
    gate.confirm(tool)
    return {"ok": True, "state": gate.state.snapshot()}


@mcp.tool()
def audit() -> dict:
    """Audit the configured rulebook itself: vacuous (say-nothing) rules,
    redundant rules, equivalent rules, and minimal contradictory cores."""
    gate = _require_gate()
    return gate.engine.audit()


@mcp.tool()
def state() -> dict:
    """Return the current execution state (the facts the gate is validating
    against) and the running check stats."""
    gate = _require_gate()
    return {"facts": gate.state.snapshot(), "stats": gate.stats()}


@mcp.tool()
def reset() -> dict:
    """Clear all recorded state but keep the configured rules, so you can
    re-run a scenario from a clean slate."""
    gate = _require_gate()
    gate.state.reset()
    gate.log.clear()
    return {"ok": True}


if __name__ == "__main__":
    mcp.run()
