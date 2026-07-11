# witt

Deterministic logic validation for AI agent tool calls.

Witt is a truth table engine that sits between an agent's decision and execution, catching invalid tool calls (missing prerequisites, missing parameters, unconfirmed destructive actions, wrong ordering, acting on the wrong object, fabricated argument values) before they run.

Every verdict is **a matter of fact**: it follows from your recorded execution state and your rules (generated from the tool specs you already have), so it's explainable and only ever says _no_ for a reason you can point to. The core is a general propositional-logic engine (cross-checked against [z3](https://github.com/Z3Prover/z3)), so it isn't limited to agents; see [Beyond tool calling](#beyond-tool-calling).

## The idea

In his *Tractatus* (1921), Wittgenstein argues that a statement's content is the set of possibilities it rules out. witt is that idea as a gate: your rules mark certain situations as off-limits, and before any action it checks whether the move you're about to make lands in one of them. The gate checks the rules themselves, too, so it can also flag a rule that guards against nothing or a rulebook that quietly contradicts itself.

## Why

Agents pick the wrong tools: they query databases before authenticating, send emails before composing them, read records they just deleted. These aren't hallucinations, they're logic errors, and logic errors are exactly what a truth table catches deterministically. And because the check reads the execution log directly rather than classifying intent, there's no feature-extraction step to misfire: the false-positive rate is zero by construction.

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

# Rules are generated from the specs; you don't write them by hand
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

This checks object _identity_: whether `tool_b` acts on the same object `tool_a` produced. Whether the value itself is _correct_ stays outside its scope. It compares runtime argument values (which the boolean engine never sees), so it's enforced in the `Supervisor` and reported on `verdict.binding_violations`. A bound dependency implies ordering, so it subsumes the plain `dependencies` entry.

### Grounding: fabricated-argument detection

> "A name means an object. The object is its meaning." (*Tractatus* 3.203)

Every identifier an agent passes — a symbol, a file name, an ID — must appear somewhere it could have legitimately come from: the user's request, the tool specs, the config, or a prior result. A value found nowhere in that corpus is a name without a bearer, and almost certainly hallucinated. Where the engine validates the logical form of a call, grounding validates its reference.

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

Derived values are the known blind spot: a translation (city → airport code) or a computation ("fill the tank" → capacity − current) is legitimate yet appears nowhere in the corpus. The default `warn` mode absorbs this — an ungrounded value surfaces in the feedback for confirmation while the run continues. Reserve `strict` for settings where every legitimate value provably flows through the corpus. Free-text params, booleans, small numbers, and ISO dates are skipped by design.

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

`mcp_server.py` exposes the gate as MCP tools, so you or an agent can walk a real scenario and watch each call get allowed or blocked with the exact feedback an LLM would receive. It's a demo surface: in production witt is middleware the harness runs before every tool call, rather than a tool the model chooses.

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

## Writing rules directly

You don't need `generate_rules`; the engine is general.

```python
from witt import TruthTableEngine

e = TruthTableEngine()
e.rule("no read after delete",
       e.IMPLIES("RecordDeleted", e.NOT("Call_read_record")))
e.rule("ambiguity blocks execution",
       e.IMPLIES(e.AND("ClarificationNeeded", e.NOT("ClarificationResolved")),
                 e.NOT("Call_execute")))
```

### The possibility space

Rules say what the agent may _do_. The **space** says what is even _possible_: the internal relations between facts. A session can't be both live and expired; a completed search always has a result. Declare these or the truth table enumerates impossible worlds, and a "counterexample" or "conflict" can be pure fantasy.

```python
e.incompatible("SessionLive", "SessionExpired")   # at most one
e.coupled("Done_search", "Has_search_result")     # all-or-none
e.one_of("StatusOpen", "StatusClosed")            # exactly one
e.constrain("premise", e.IMPLIES("A", "B"))        # any raw axiom
```

The payoff: incoherent state is caught distinctly from a forbidden action (the `Supervisor` reports `"Incoherent state: …"` separately from `"Blocked: …"`), and `check_entailment`/`find_conflicts` range only over genuine possibilities. With no space declared, behaviour is identical to plain rule validation. See `examples/possibility_space.py`.

### Auditing the rules

The gate validates tool calls against the rules; `audit()` validates the _rules themselves_. Rule quality is the whole ceiling of this approach, so this is where you keep it honest.

```python
report = engine.audit()
# report["vacuous"]          rules true in every possible world (dead weight)
# report["redundant"]        [{rule, entailed_by}] implied by the other rules
# report["equivalent"]       groups of rules with identical sense
# report["conflicts"]        minimal jointly-unsatisfiable rule sets
# report["impossible_space"] constraint sets that admit no world at all
```

It reports the **minimal conflict core** — the smallest set of rules that actually contradict, sparing you an unhelpful "all your rules conflict" — and decomposes the ruleset into variable-disjoint components so it scales: it audits the 159-rule / 385-proposition BFCL engine (a naive table would be `2^240`) in a few milliseconds. See `examples/rule_audit.py`.

## Results

Two claims, tested separately, because they're different claims.

**1. The engine is logically correct.** 1,500 randomized formulas are cross-checked against [z3](https://github.com/Z3Prover/z3), an independent SMT solver sharing no code with witt: evaluation, entailment (including every classical fallacy), conflict detection, vacuity, and possibility-space filtering all agree, on every case. Soundness in the only sense that matters, it computes the same function classical logic does. (`tests/test_differential.py`)

**2. The rules catch structural tool-call errors at zero false positives, judged by an independent oracle.** BFCL multi-turn ships executable stateful classes; the harness executes each call sequence, labels a mutation "broke the task" by its _actual effect_ (final state + return values vs. ground truth), and only then gates it with a frozen, spec-derived engine. Mutations are generated blind to the rules. (`examples/oracle_eval.py`, `tests/test_oracle.py`)

Recall per error class (train/test split; some simulators are stochastic, so run the script for current values):

| Error class                               | spec-only rules | + mined dependencies | + grounding |
| ----------------------------------------- | --------------- | -------------------- | ----------- |
| Missing required argument (structural)    | **~0.96**       | ~0.96                | **~0.98**   |
| Wrong-but-valid value (semantic)          | ~0.00           | ~0.20                | **~0.85**   |
| Reordered / missing prerequisite          | ~0.00           | ~0.28                | ~0.29       |
| Swapped tool                              | ~0.7            | ~0.9                 | ~0.78       |
| **Valid ground-truth runs flagged**       | **0**           | ~15-20%              | ~20% (warnings) |

Read this as a map of the competence boundary:

- **Structural errors: caught near-perfectly, at zero false positives.** Every valid ground-truth sequence is allowed; every dropped required argument is blocked. This is the defensible guarantee.
- **Fabricated values: now mostly caught.** Grounding flags values that appear nowhere in the request, specs, or prior results (~0.85 recall, from ~0). Derived values — translations and computations — account for the flagged valid runs, so violations surface as warnings and the run continues. (`examples/grounding_eval.py`)
- **The "0 false positives" guarantee is only as safe as your rules.** Required-param rules are mechanically certain, so they never misfire. _Mined_ dependencies raise recall on ordering but reintroduce false positives (they capture correlation rather than causation), so add them with eyes open.

Latency ~370 μs on the full 159-rule engine. Full suite: 1,624 tests.

## Beyond tool calling

The core is a general propositional-logic engine, so it fits **anywhere you express state as boolean facts and want a fast, deterministic, explainable "is this allowed / consistent?" check**, with the rules auditable for contradictions before you ship.

```python
from witt import TruthTableEngine

e = TruthTableEngine()
e.rule("dark mode needs the new UI", e.IMPLIES("dark_mode", "new_ui"))
e.rule("SSO needs an org plan",       e.IMPLIES("sso", "org_plan"))
e.incompatible("cache", "debug")      # can't ship both at once

e.validate_closed({"dark_mode": True, "new_ui": False})["violations"]
# → ["dark mode needs the new UI"]
e.audit()["conflicts"]                # → []  (or the minimal conflicting rule set)
```

The same shape covers workflow/state-machine guards ("no refund after archive"), access-control policy (`delete → admin ∧ confirmed`), eligibility rules (`loan → income_verified ∧ ¬flagged`), CI/CD gates (`prod_deploy → tests_passed ∧ approved ∧ ¬freeze`), and cross-field form validation (`country=US → state_required`).

## Limits

- **It sees structure.** It enforces "an amount is present and confirmed"; it can't know `amount=1000` should have been `100`. Ground comparisons to booleans (`amount > 100` is a fact something else computes and feeds in).
- **Propositional only.** No quantifiers ("all suppliers must…"); ground them into per-instance propositions.
- **Exponential in free variables.** The engine caps at 22 free variables per expression; in closed-world agent validation everything is pinned, so this never bites.
- **The rules are the ceiling.** The gate catches what the rules describe. Rule quality is the product; `generate_rules` + trace mining get you most of the way.

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

## Why "witt"?

Short for Wittgenstein. The three features above are one move from his _Tractatus_ — a proposition's content is the set of possibilities it excludes — applied three times:

- **The possibility space** is his _logical space_ (2.11); the colour-exclusion problem (6.3751) is exactly the bug it fixes.
- **Rule auditing** is his two degenerate truth-functions (4.46): a tautology "says nothing" (a vacuous rule), a contradiction can't be satisfied (a conflicting ruleset).
- **Rule identity** is _sense = truth-conditions_ (4.431): two rules are the same when they permit the same worlds.

You don't need any of this to use the tool; it's just why it's shaped the way it is.

## Running the benchmarks

```bash
# Engine correctness vs. z3 (no data needed)
pip install z3-solver && pytest tests/test_differential.py

# Structural benchmark + the independent execution oracle
git clone --depth 1 https://github.com/ShishirPatil/gorilla
pip install mpmath   # some BFCL backend classes need it
export GORILLA_ROOT=gorilla/berkeley-function-call-leaderboard
export BFCL_DATA_DIR=$GORILLA_ROOT/bfcl_eval/data
pytest tests/test_bfcl.py tests/test_oracle.py -v

# The full confusion matrix (per-category recall, false positives, mined-deps tradeoff)
python examples/oracle_eval.py
```

## License

MIT
