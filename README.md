# truthgate

Deterministic logic validation for AI agent tool calls.

A truth table engine that sits between an agent's decision and execution. It catches invalid tool calls — missing prerequisites, missing parameters, unconfirmed destructive actions, wrong ordering — before they run. No model, no API, no network. Microseconds per check.

## Why

Agents pick tools wrong. They query databases before authenticating, send emails before composing them, act on ambiguous input, and read records they just deleted. These aren't hallucinations — they're logic errors, and logic errors are exactly what a truth table catches deterministically.

The inputs are facts from the execution log (was auth called? did the user confirm?), not fuzzy NLP extractions. That's why this works where LLM-checks-LLM approaches struggle: there's no feature-extraction bottleneck and no judgment calls, so the false-positive rate is zero by construction.

## Results

Two claims, tested separately — because they're different claims.

**1. The engine is logically correct.** 1,500 randomized formulas are cross-checked against [z3](https://github.com/Z3Prover/z3), an independent SMT solver that shares no code with truthgate: evaluation, entailment (including every classical fallacy), conflict detection, vacuity, and possibility-space filtering all agree, on every case. This is soundness in the only sense that matters — it computes the same function classical logic does, not "passes the tests we wrote." (`tests/test_differential.py`)

**2. The rules catch structural tool-call errors at zero false positives — judged by an independent oracle.** BFCL multi-turn ships executable stateful classes (a real file system, trading bot, …). The harness executes each call sequence, labels a mutation "broke the task" by its *actual effect* (final state + return values vs. ground truth), and only then gates it with a purely spec-derived, frozen engine. Mutations are generated blind to the rules, and the false-negative cell — what the tool *misses* — is counted. (`examples/oracle_eval.py`, `tests/test_oracle.py`)

Recall per error class, on independently-executed BFCL sequences (train/test split; numbers vary slightly because some simulators are stochastic — run the script for current values):

| Error class | spec-only rules | + mined dependencies |
|---|---|---|
| Missing required argument (structural) | **~0.96** | ~0.96 |
| Wrong-but-valid value (semantic) | ~0.00 | ~0.20 |
| Reordered / missing prerequisite | ~0.00 | ~0.28 |
| Swapped tool | ~0.7 | ~0.9 |
| **False positives on valid ground truth** | **0** | **~15–20%** |

Read this as the competence boundary, not a grade:

- **Structural errors: caught near-perfectly, at zero false positives.** Every valid ground-truth sequence (run through the real interpreter) is allowed; every dropped required argument is blocked. This is the genuine, defensible guarantee.
- **Value and semantic errors: invisible.** Presence-checking cannot see `mv(source='wrong.txt')` when the argument is present but wrong. If that's your failure mode, this tool won't help.
- **The "0 false positives" guarantee is only as safe as your rules.** Required-param rules are mechanically certain, so they never misfire. *Mined* dependencies raise recall on ordering errors but reintroduce false positives — trace mining is correlation, not causation, so it infers spurious rules like `cp requires cd first`. Add heuristic rules with eyes open.

The original structural benchmark still holds: 200/200 valid BFCL sequences allowed, 0 false positives, 229 injected errors caught — though the injected-error count is coverage-by-construction (the injector falsifies the exact premise a rule checks), so claim 2 above is the stronger evidence. Latency is ~370 μs on the full 159-rule engine. Full suite: 1,585 tests.

## Install

```bash
pip install -e .
# dev
pip install -e ".[dev]" && pytest
```

## Quickstart

```python
from truthgate import Supervisor, generate_rules

# Your tools — the same JSON schema you already give your agent
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

# Rules are generated from the specs — you don't write them by hand
engine = generate_rules(tools, dependencies={"summarize": ["search_web"]})
gate = Supervisor(engine)

verdict = gate.check("summarize")
# verdict.allowed == False
# verdict.feedback == "Blocked: summarize requires search_web first"

gate.check("search_web", params={"query": "x"})
gate.record_success("search_web")
gate.check("summarize").allowed   # True
```

Three rule types are generated automatically:

1. **Required parameters** — `Call_X → Has_X_param` from the JSON schema's `required` list
2. **Confirmations** — tools whose name/description contains a destructive verb (`delete`, `send`, `pay`, `book`, …) require `UserConfirmed`
3. **Dependencies** — from the `dependencies` argument, or mined from execution traces with `infer_dependencies_from_traces()`

It also declares the one internal relation implied by the `StateTracker` contract: `Done_X` and `Has_X_result` are always set together, so they're `coupled` (`Done_X ↔ Has_X_result`) — a state with one but not the other is flagged as *impossible*, not merely invalid. Pass `model_state_space=False` to skip this if you set state by hand. It's free at runtime: with everything pinned, closed-world validation short-circuits the constraint check, so per-check latency is unchanged.

You can also write rules directly:

```python
from truthgate import TruthTableEngine

e = TruthTableEngine()
e.rule("no read after delete",
       e.IMPLIES("RecordDeleted", e.NOT("Call_read_record")))
e.rule("ambiguity blocks execution",
       e.IMPLIES(e.AND("ClarificationNeeded", e.NOT("ClarificationResolved")),
                 e.NOT("Call_execute")))
```

## The possibility space

Rules say what the agent may *do*. The **space** says what is even *possible* — the internal relations between facts. A session can't be both live and expired; a completed search always has a result. Wittgenstein assumed elementary propositions were logically independent (*Tractatus* 2.061) and then hit the wall himself — the colour-exclusion problem (6.3751). If you don't declare these relations, the truth table enumerates impossible worlds and a "counterexample" or "conflict" can be pure fantasy.

```python
e.incompatible("SessionLive", "SessionExpired")   # at most one (colour-exclusion)
e.coupled("Done_search", "Has_search_result")     # all-or-none
e.one_of("StatusOpen", "StatusClosed")            # exactly one
e.constrain("premise", e.IMPLIES("A", "B"))        # any raw axiom
```

Two payoffs:

1. **Incoherent state is caught, distinctly from a forbidden action.** `validate_closed` now returns `possible` and `space_violations`; the `Supervisor` reports `"Incoherent state: …"` separately from `"Blocked: …"`. A state that can't exist is a different bug from an action that isn't allowed.
2. **Reasoning ranges only over genuine possibilities.** `check_entailment` and `find_conflicts` skip impossible worlds — so a claim that's merely contingent in full logical space can become *necessary* once the space excludes the bad combinations (5.101), and rules jointly satisfiable only in impossible worlds are correctly flagged as conflicts.

With no space declared, behaviour is identical to plain rule validation. See `examples/possibility_space.py`.

## Auditing the rules

The gate validates tool calls against the rules. `audit()` validates the *rules themselves*, via the two degenerate truth-functions Wittgenstein singled out (4.46): the **tautology** (true in every world — "says nothing") and the **contradiction** (unsatisfiable). Rule quality is the whole ceiling of this approach, so this is where you keep it honest.

```python
report = engine.audit()
# report["vacuous"]     — rules true in every possible world (dead weight)
# report["redundant"]   — [{rule, entailed_by}] implied by the other rules
# report["equivalent"]  — groups of rules with identical sense (4.431)
# report["conflicts"]   — minimal jointly-unsatisfiable rule sets
```

It reports the **minimal conflict core** (the smallest set of rules that actually contradict), not "all your rules conflict." The trick that makes it usable at scale: rules with no shared variables are logically independent, so `audit` decomposes the ruleset into connected components and checks each one on its own — it audits the 159-rule / 385-proposition BFCL engine (a naive single table would be `2^240`) in a few milliseconds. See `examples/rule_audit.py`.

## The agent loop

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

## API surface

| | |
|---|---|
| `TruthTableEngine` | core engine: `prop`, `rule`, `validate`, `validate_closed`, `check_entailment`, `find_conflicts`, `truth_table`, `to_json`/`from_json` |
| — possibility space | `incompatible`, `coupled`, `one_of`, `constrain` — declare which worlds are genuinely possible |
| — rule auditing | `audit`, `is_vacuous`, `minimal_conflicts` — find vacuous, redundant, equivalent, and contradictory rules |
| `Supervisor` | the gate: `check`, `record_success`, `record_failure`, `confirm`, `stats`, `audit`; `Supervisor(engine, strict=True)` fails at construction on a contradictory ruleset |
| `StateTracker` | execution state: `set`, `on_tool_success`, `snapshot`, `history` |
| `generate_rules` | tool specs → engine |
| `infer_dependencies_from_traces` | mine ordering constraints from logs |

`validate_closed` (closed-world: absent fact = false) is what agent validation uses. `validate` (open-world: absent fact = free variable) is for pure logic checking — e.g. `check_entailment` correctly flags affirming-the-consequent and other fallacies.

## Limits

- **Propositional logic only.** No quantifiers ("all suppliers must…"). Ground them into per-instance propositions.
- **Exponential in free variables.** The engine raises above 22 free variables in one expression. In closed-world agent validation this never bites — everything is pinned.
- **The rules are the ceiling.** The gate catches what the rules describe. Rule quality is the product; `generate_rules` + trace mining get you most of the way.

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
