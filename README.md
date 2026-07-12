# witt

A deterministic logic validator for AI agent tool use, loosely based on the early philosophy of Ludwig Wittgenstein.

Conceived as an alternative to human-in-the-loop or LLM-as-judge evaluators, witt is a truth table engine for catching invalid AI agent tool use.

## What?

In _Tractatus Logico-Philosophicus_, Wittgenstein argues that a statement's content is the set of possibilities it rules out. witt turns that premise into a logic gate for AI agent tool use.

## Install

```bash
pip install -e .
pip install -e ".[dev]" && pytest    # dev + tests
pip install -e ".[mcp]"              # the demo MCP server
```

## Quickstart

```python
from witt import Supervisor, generate_rules

# Your tools: the same JSON schema you already give your agent
tools = [
    {"name": "search_web",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "summarize", "parameters": {"type": "object", "properties": {}}},
    {"name": "send_email", "description": "Send an email",
     "parameters": {"type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"]}},
]

# Rules are generated from the specs.
engine = generate_rules(tools, dependencies={"summarize": ["search_web"]})
gate = Supervisor(engine)

verdict = gate.check("summarize")
# verdict.allowed == False
# verdict.feedback == "Blocked: summarize requires search_web first"

gate.check("search_web", params={"query": "x"})
gate.record_success("search_web")
gate.check("summarize").allowed   # True
```

`generate_rules` produces three rule types automatically:

1. **Required parameters** — `Call_X → Has_X::param`, from the schema's `required` list.
2. **Confirmations** — tools whose _name_ contains a destructive verb (`delete`, `send`, `pay`, `book`, …) require **per-tool** confirmation. Confirming one action never authorizes another, and `record_success` consumes the confirmation so the next call must re-confirm. Confirm via `gate.confirm("delete_record")` (or `gate.confirm()` for the tool just checked).
3. **Dependencies** — from the `dependencies` argument, or mined from traces with `infer_dependencies_from_traces()` (correlation-based; review before using as hard gates).

It also couples `Done_X ↔ Result_X` (the `StateTracker` sets both on success), so a state with one but not the other is flagged as _impossible_ — a stronger signal than an ordinary rule violation. Free at runtime, and disable-able with `model_state_space=False`.

### Argument binding

A plain dependency is satisfied by _any_ completed prerequisite: `tool_b requires tool_a` lets the agent do `tool_a` on one object and `tool_b` on a different one. When the two calls must concern the **same object**, pass `bindings`:

```python
engine = generate_rules(
    tools,
    require_confirmation=["issue_refund"],
    # issue_refund(order_id=V) requires a get_order that completed with order_id=V
    bindings={"issue_refund": [{"tool": "get_order", "param": "order_id"}]},
)
gate = Supervisor(engine)

gate.check("get_order", params={"order_id": "A-4471"})
gate.record_success("get_order")            # reuses the params from the check
gate.confirm("issue_refund")

gate.check("issue_refund", params={"order_id": "Z-0000", "amount": 10}).allowed
# → False: "Argument mismatch: issue_refund(order_id='Z-0000') requires get_order
#           to have completed with order_id='Z-0000' (get_order completed with order_id=['A-4471'])"
```

This checks object _identity_ — that `tool_b` acts on the same object `tool_a` produced, not whether the value is _correct_. It runs on the actual argument values in the `Supervisor`, reports on `verdict.binding_violations`, and implies ordering, so it subsumes a plain `dependencies` entry.

### Grounding: fabricated-argument detection

Every identifier an agent passes, a symbol, a file name, an ID, must trace back to somewhere it could have come from: the user's request, the tool specs, the config, or a prior result. A value that appears in none of them is almost certainly hallucinated.

```python
from witt import Grounding, Supervisor, generate_rules

g = Grounding(user_text=request, tool_specs=tools, mode="warn")
gate = Supervisor(generate_rules(tools), grounding=g)

verdict = gate.check("place_order", params={"symbol": "TSLA"})
# verdict.grounding_violations →
#   ["place_order(symbol='TSLA'): value appears nowhere in user request,
#     tool specs, or prior results — possibly fabricated"]

gate.record_success("get_stock_info", result=response)  # results feed the corpus
```

The default `warn` mode surfaces ungrounded values in the feedback without blocking; use `strict` where every legitimate value provably comes from the corpus.

### The agent loop

```python
verdict = gate.check(tool_name, params)
if verdict:
    result = execute(tool_name, params)
    gate.record_success(tool_name)
else:
    # violation text goes straight back into the LLM's context
    response = llm(f"{verdict.feedback}. Choose a different action.")
```

See `examples/agent_loop.py` for a runnable version.

## Try it live (MCP server)

`mcp_server.py` exposes the gate as a set of MCP tools.

```bash
pip install -e ".[mcp]"
python mcp_server.py     # stdio transport
```

Register with any MCP client (`.mcp.json` for Claude Code):

```json
{
  "mcpServers": {
    "witt": {
      "command": "/abs/path/.venv/bin/python",
      "args": ["/abs/path/mcp_server.py"]
    }
  }
}
```

Tools: `configure` (build rules from tool specs), `check` (the gate), `record_success`, `confirm`, `audit`, `state`, `reset`.

## Beyond generated rules

The engine is general, so you're not tied to `generate_rules`:

- **Write rules by hand** on `TruthTableEngine` — e.g. `e.rule("no read after delete", e.IMPLIES("RecordDeleted", e.NOT("Call_read_record")))`.
- **Declare the possibility space** with `incompatible` / `coupled` / `one_of` / `constrain`, so impossible worlds (a session both live and expired) never produce phantom counterexamples, and incoherent state is reported separately from a forbidden action.
- **Audit the rules themselves** with `engine.audit()` — it flags vacuous, redundant, equivalent, and contradictory rules (the minimal conflict core), and scales to the 159-rule BFCL engine in milliseconds.

See `examples/possibility_space.py` and `examples/rule_audit.py`.

## Results

**The engine computes logic correctly.** 1,500 random formulas are checked against z3; evaluation, entailment, conflict detection, vacuity, and possibility-space filtering agree on every one. (`tests/test_differential.py`)

**The rules catch structural tool-call errors with zero false positives.** On BFCL's executable multi-turn tasks, the harness runs each call sequence and decides whether a mutation broke it by its _actual effect_ (final state and return values vs. ground truth), then checks whether the engine caught it. The errors are generated blind to the rules, so an independent oracle, not witt, decides what counts as broken. (`examples/oracle_eval.py`, `tests/test_oracle.py`)

Recall per error class (train/test split; some simulators are stochastic, so run the script for current values):

| Error class                            | spec-only rules | + mined dependencies | + grounding     |
| -------------------------------------- | --------------- | -------------------- | --------------- |
| Missing required argument (structural) | **~0.96**       | ~0.96                | **~0.98**       |
| Wrong-but-valid value (semantic)       | ~0.00           | ~0.20                | **~0.85**       |
| Reordered / missing prerequisite       | ~0.00           | ~0.28                | ~0.29           |
| Swapped tool                           | ~0.7            | ~0.9                 | ~0.78           |
| **Valid ground-truth runs flagged**    | **0**           | ~15-20%              | ~20% (warnings) |

The competence boundary: structural errors are the defensible guarantee — every valid sequence allowed, every dropped required argument blocked. Grounding lifts fabricated-value recall from ~0 to ~0.85 (derived values — translations, computations — are the blind spot, surfaced as warnings). And the "0 false positives" claim is only as safe as your rules: required-param rules never misfire, but _mined_ dependencies trade recall for some false positives (correlation, not causation), so add them with eyes open.

## API surface

|                                  |                                                                                                                                                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TruthTableEngine`               | core engine: `prop`, `rule`, `validate`, `validate_closed`, `check_entailment`, `find_conflicts`, `truth_table`, `to_json`/`from_json`                                                                                                           |
| possibility space                | `incompatible`, `coupled`, `one_of`, `constrain`: declare which worlds are genuinely possible                                                                                                                                                    |
| rule auditing                    | `audit`, `is_vacuous`, `minimal_conflicts`                                                                                                                                                                                                       |
| `Supervisor`                     | the gate: `check`, `record_success`, `record_failure`, `confirm`, `unconfirm`, `stats`, `audit`; enforces argument `bindings` (`verdict.binding_violations`); `Supervisor(engine, strict=True)` fails at construction on a contradictory ruleset |
| `StateTracker`                   | execution state: `set`, `on_tool_success`, `completed_with`, `snapshot`, `history`                                                                                                                                                               |
| `generate_rules`                 | tool specs → engine; accepts `dependencies`, `require_confirmation`, `bindings`                                                                                                                                                                  |
| `infer_dependencies_from_traces` | mine ordering constraints from logs                                                                                                                                                                                                              |
| `normalize_bindings`             | canonicalize the `bindings` argument                                                                                                                                                                                                             |

`validate_closed` (closed-world: absent fact = false) is what agent validation uses. `validate` (open-world: absent fact = free variable) is for pure logic checking; e.g. `check_entailment` correctly flags affirming-the-consequent and other fallacies.

## License

MIT
