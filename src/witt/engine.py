"""
witt.engine — Deterministic propositional logic validation.

The core truth table engine. Exhaustive, deterministic, dependency-free.

Validated against:
  - 1,500 randomized formulas cross-checked with the z3 SMT solver
  - 200/200 valid BFCL multi-turn sequences allowed, 0 false positives
  - an independent BFCL execution oracle (see examples/oracle_eval.py)
"""

from __future__ import annotations


class TruthTableEngine:
    """Propositional logic engine for validating agent decisions.

    Usage:
        e = TruthTableEngine()
        e.prop("Authenticated", "Agent has authenticated", "state")
        e.prop("QueryDB", "Query the database", "action")
        e.rule("DB needs auth", e.IMPLIES("QueryDB", "Authenticated"))

        result = e.validate_closed({"QueryDB": True, "Authenticated": False})
        # result.valid == False
        # result.violations == ["DB needs auth"]
    """

    # Above this many free variables, exhaustive enumeration (2^n rows) is
    # refused — truth tables are exponential. Closed-world agent validation
    # never hits it (everything is pinned). Single source of truth for the cap.
    MAX_VARS = 22

    def __init__(self):
        self.propositions: dict[str, dict] = {}
        self.rules: list[dict] = []
        # Constraints defining the space of *genuine possibilities*
        # (Wittgenstein's logical space / internal relations, 2.11, 4.122).
        # A world that violates one is not "bad" — it is impossible.
        self.constraints: list[dict] = []

    # ── Propositions ─────────────────────────────────────────────
    def prop(self, name: str, description: str = "", prop_type: str = "feature"):
        """Register a proposition.

        prop_type:
          "state"   — facts about the world (from execution log)
          "action"  — a proposed action
          "feature" — extracted feature (classification pipelines)
          "output"  — an agent's conclusion (eligible for auto-correction)
        """
        if not name or not isinstance(name, str):
            raise ValueError("Proposition name must be a non-empty string")
        self.propositions[name] = {"description": description, "type": prop_type}
        return self

    # ── Expression builders ──────────────────────────────────────
    def AND(self, a, b):
        return {"op": "AND", "args": [a, b]}

    def OR(self, a, b):
        return {"op": "OR", "args": [a, b]}

    def NOT(self, a):
        return {"op": "NOT", "args": [a]}

    def IMPLIES(self, a, b):
        return {"op": "IMPLIES", "args": [a, b]}

    def IFF(self, a, b):
        return {"op": "IFF", "args": [a, b]}

    def XOR(self, a, b):
        return {"op": "XOR", "args": [a, b]}

    def ALL(self, *args):
        return self._reduce(args, "AND")

    def ANY(self, *args):
        return self._reduce(args, "OR")

    @staticmethod
    def _reduce(args, op):
        if not args:
            raise ValueError(f"{op} requires at least one argument")
        result = args[0]
        for a in args[1:]:
            result = {"op": op, "args": [result, a]}
        return result

    # ── Rules ────────────────────────────────────────────────────
    def rule(self, name: str, expr):
        """Add a validation rule. Auto-registers any unregistered variables
        so they participate in closed-world validation (prevents the
        free-variable bug found during BFCL testing)."""
        for var in self._get_vars(expr):
            if var not in self.propositions:
                self.propositions[var] = {"description": "(auto-registered)", "type": "state"}
        self.rules.append({"name": name, "expr": expr})
        return self

    # ── The possibility space (internal relations) ───────────────
    # These declare which combinations of propositions are genuinely
    # possible. Wittgenstein assumed elementary propositions are
    # logically independent (2.061) — but they rarely are (the
    # "colour-exclusion problem", 6.3751). Without this, the truth
    # table enumerates impossible worlds, and every counterexample /
    # conflict may be a fantasy. Declaring the space fixes that.
    def constrain(self, name: str, expr):
        """Add a raw axiom every genuine world must satisfy.

        Like `rule`, but a violating world is *impossible* rather than invalid.
        Auto-registers referenced variables."""
        for var in self._get_vars(expr):
            if var not in self.propositions:
                self.propositions[var] = {"description": "(auto-registered)", "type": "state"}
        self.constraints.append({"name": name, "expr": expr})
        return self

    def incompatible(self, *props, name: str | None = None):
        """At most one of these may hold (the colour-exclusion case).

        e.g. incompatible("Authenticated", "SessionExpired") —
        a session cannot be both. Rules out the row where two are true."""
        if len(props) < 2:
            raise ValueError("incompatible() needs at least two propositions")
        expr = self._pairwise(props, lambda a, b: self.NOT(self.AND(a, b)))
        return self.constrain(name or f"incompatible({', '.join(props)})", expr)

    def coupled(self, *props, name: str | None = None):
        """All-or-none: these stand or fall together.

        e.g. coupled("Done_search", "Has_search_result") — a completed
        search always has a result and vice versa."""
        if len(props) < 2:
            raise ValueError("coupled() needs at least two propositions")
        expr = self._reduce([self.IFF(props[0], p) for p in props[1:]], "AND")
        return self.constrain(name or f"coupled({', '.join(props)})", expr)

    def one_of(self, *props, name: str | None = None):
        """Exactly one holds — a partition of the space (mutex + total)."""
        if len(props) < 2:
            raise ValueError("one_of() needs at least two propositions")
        mutex = self._pairwise(props, lambda a, b: self.NOT(self.AND(a, b)))
        expr = self.AND(mutex, self._reduce(list(props), "OR"))
        return self.constrain(name or f"one_of({', '.join(props)})", expr)

    def _pairwise(self, props, build):
        terms = [build(props[i], props[j])
                 for i in range(len(props)) for j in range(i + 1, len(props))]
        return self._reduce(terms, "AND")

    def _world_possible(self, assignment: dict) -> bool:
        """True unless the assignment fully determines a constraint and
        falsifies it. Partially-covered constraints are left open — the
        undetermined variables could still be assigned to satisfy them."""
        keys = assignment.keys()
        for c in self.constraints:
            cv = self._get_vars(c["expr"])
            if cv <= keys and not self._eval(c["expr"], assignment):
                return False
        return True

    def _constraint_closure(self, seed: set) -> set:
        """Expand a variable set to include every constraint that touches
        it (transitively), so those constraints become fully evaluable
        during enumeration."""
        universe = set(seed)
        changed = True
        while changed:
            changed = False
            for c in self.constraints:
                cv = self._get_vars(c["expr"])
                if (cv & universe) and not (cv <= universe):
                    universe |= cv
                    changed = True
        return universe

    def _space_violations(self, props: dict) -> list:
        """Constraints the given (fully-determined) props falsify — i.e.
        the props describe an impossible world."""
        keys = props.keys()
        return [c["name"] for c in self.constraints
                if self._get_vars(c["expr"]) <= keys and not self._eval(c["expr"], props)]

    # ── Evaluation ───────────────────────────────────────────────
    def _eval(self, expr, assignment) -> bool:
        if isinstance(expr, str):
            return assignment.get(expr, False)
        op, args = expr["op"], expr["args"]
        if op == "NOT":
            return not self._eval(args[0], assignment)
        if op == "AND":
            return self._eval(args[0], assignment) and self._eval(args[1], assignment)
        if op == "OR":
            return self._eval(args[0], assignment) or self._eval(args[1], assignment)
        if op == "IMPLIES":
            return (not self._eval(args[0], assignment)) or self._eval(args[1], assignment)
        if op == "IFF":
            return self._eval(args[0], assignment) == self._eval(args[1], assignment)
        if op == "XOR":
            return self._eval(args[0], assignment) != self._eval(args[1], assignment)
        raise ValueError(f"Unknown operator: {op}")

    def _get_vars(self, expr) -> set:
        if isinstance(expr, str):
            return {expr}
        s = set()
        for a in expr.get("args", []):
            s |= self._get_vars(a)
        return s

    def _all_assignments(self, variables):
        vlist = sorted(variables)
        n = len(vlist)
        if n > self.MAX_VARS:
            raise ValueError(
                f"Expression has {n} free variables (2^{n} rows). "
                "Truth tables are exponential; pin more variables or split the rule."
            )
        for i in range(1 << n):
            yield {v: bool(i & (1 << (n - 1 - j))) for j, v in enumerate(vlist)}

    # ── Validation ───────────────────────────────────────────────
    def validate(self, props: dict) -> dict:
        """Open-world validation: propositions not in `props` are free
        variables — a rule is violated if any *genuinely possible*
        assignment of the free variables falsifies it. Counterexamples
        drawn from worlds the possibility space rules out don't count.
        Use for pure logic checking."""
        space_violations = self._space_violations(props)
        violations = []
        details = []

        # Fast path for fully-pinned worlds (the closed-world agent case):
        # if every constraint variable is already determined by `props`,
        # a rule whose own variables are all pinned has no free variables
        # to enumerate — evaluate it directly. This keeps validation at
        # microsecond latency no matter how many constraints are declared.
        pk = props.keys()
        all_cvars = set()
        for c in self.constraints:
            all_cvars |= self._get_vars(c["expr"])
        closed = all_cvars <= pk
        world_ok = not space_violations  # props-as-world possibility

        for r in self.rules:
            rule_vars = self._get_vars(r["expr"])
            violated = False
            counterexample = None

            if closed and rule_vars <= pk:
                # No free variables and the space is fully determined.
                if world_ok and not self._eval(r["expr"], props):
                    violated = True
                    counterexample = dict(props)
            else:
                # Open completion: enumerate free variables (plus any the
                # space links in) over genuine possibilities only.
                universe = self._constraint_closure(rule_vars)
                free = {v for v in universe if v not in props}
                for fa in self._all_assignments(free):
                    merged = {**fa, **props}
                    if not self._world_possible(merged):
                        continue  # not a genuine possibility — skip
                    if not self._eval(r["expr"], merged):
                        violated = True
                        counterexample = merged
                        break

            details.append({"rule": r["name"], "passed": not violated})
            if violated:
                violations.append({"rule": r["name"], "counterexample": counterexample})

        return {
            "valid": len(violations) == 0 and len(space_violations) == 0,
            "violations": [v["rule"] for v in violations],
            "violation_details": violations,
            "space_violations": space_violations,
            "possible": len(space_violations) == 0,
            "details": details,
            "corrections": self._suggest_corrections(props, violations),
        }

    def validate_closed(self, props: dict) -> dict:
        """Closed-world validation: any proposition not in `props` is False.
        This covers every variable referenced by any rule, so unregistered
        variables cannot silently become free variables.

        Use this for agent validation, where the execution state is fully
        known and absence of a fact means the fact is false."""
        full = {}
        for r in self.rules:
            for v in self._get_vars(r["expr"]):
                full[v] = False
        for c in self.constraints:
            for v in self._get_vars(c["expr"]):
                full[v] = False
        for k in self.propositions:
            full[k] = False
        full.update(props)
        return self.validate(full)

    def _suggest_corrections(self, props, violations):
        if not violations:
            return None
        corrections = {}
        for v in violations:
            rule = next((r for r in self.rules if r["name"] == v["rule"]), None)
            if not rule:
                continue
            for var in self._get_vars(rule["expr"]):
                p = self.propositions.get(var)
                if p and p["type"] == "output" and var in props:
                    corrections[var] = not props[var]
        return corrections or None

    # ── Analysis utilities ───────────────────────────────────────
    def truth_table(self, expr):
        """Full truth table for an expression. Each row is flagged with
        whether it is a genuine possibility under the declared space."""
        return [
            {"assignment": a, "result": self._eval(expr, a),
             "possible": self._world_possible(a)}
            for a in self._all_assignments(self._get_vars(expr))
        ]

    def check_entailment(self, premises: list, conclusion) -> dict:
        """Do the premises logically entail the conclusion — across the
        space of *genuine possibilities*? A world the space rules out is
        not a counterexample. (So a claim that is merely contingent in
        full logical space can become necessary once the impossible
        combinations are excluded — 5.101, 6.3751.)"""
        seed = set()
        for p in premises:
            seed |= self._get_vars(p)
        seed |= self._get_vars(conclusion)
        all_vars = self._constraint_closure(seed)

        counterexamples = []
        for a in self._all_assignments(all_vars):
            if not self._world_possible(a):
                continue
            if all(self._eval(p, a) for p in premises) and not self._eval(conclusion, a):
                counterexamples.append(a)

        return {"valid": len(counterexamples) == 0, "counterexamples": counterexamples}

    def find_conflicts(self) -> list:
        """Rule names participating in some minimal contradictory core,
        or [] if the whole ruleset is jointly satisfiable over the space.
        Component-decomposed, so it scales to large rule sets that a
        single 2^n table could never enumerate."""
        seen: list = []
        for core in self.minimal_conflicts():
            for n in core:
                if n not in seen:
                    seen.append(n)
        return seen

    # ── Rule auditing (the two limiting cases, 4.46) ─────────────
    # Wittgenstein: among all truth-functions exactly two are degenerate
    # — the tautology (true in every row, "says nothing") and the
    # contradiction (false in every row, unsatisfiable). Those are the
    # two failure modes of a *ruleset*. audit() reports both, plus
    # redundancy and sameness-of-sense (4.431), all as set operations
    # over the possible worlds each rule permits.

    def _components(self, max_vars: int | None = None) -> list:
        """Partition rules + constraints into connected components by
        shared variables. Variable-disjoint rules are logically
        independent — they can't conflict with, entail, or subsume one
        another — so each component can be audited on its own. This is
        what keeps auditing tractable: 2^(small component) instead of 2^(all).
        """
        if max_vars is None:
            max_vars = self.MAX_VARS
        parent: dict = {}

        def find(x):
            parent.setdefault(x, x)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a, b):
            parent[find(a)] = find(b)

        items = [("rule", r, self._get_vars(r["expr"])) for r in self.rules]
        items += [("con", c, self._get_vars(c["expr"])) for c in self.constraints]
        for _, _, vs in items:
            vs = list(vs)
            for v in vs:
                find(v)
            for v in vs[1:]:
                union(vs[0], v)

        comps: dict = {}
        for kind, obj, vs in items:
            root = find(next(iter(vs)))
            c = comps.setdefault(root, {"rules": [], "constraints": [], "vars": set()})
            (c["rules"] if kind == "rule" else c["constraints"]).append(obj)
            c["vars"] |= vs

        out = []
        for c in comps.values():
            c["too_big"] = len(c["vars"]) > max_vars
            if c["too_big"]:
                c["space_empty"] = False           # unknown; component is skipped
            elif c["constraints"]:
                c["space_empty"] = next(self._possible_worlds(c["vars"]), None) is None
            else:
                c["space_empty"] = False           # no constraints → every world possible
            out.append(c)
        return out

    def _possible_worlds(self, variables):
        """Enumerate only the genuinely possible assignments — lazily, so
        callers can early-exit instead of materializing the whole table."""
        for a in self._all_assignments(variables):
            if self._world_possible(a):
                yield a

    @staticmethod
    def _minimal_core(rules, worlds, eval_fn) -> list:
        """Shrink an unsatisfiable rule set to a minimal still-unsatisfiable
        subset by greedy deletion."""
        def sat(subset):
            return any(all(eval_fn(r["expr"], w) for r in subset) for w in worlds)

        core = list(rules)
        for r in list(core):
            trial = [x for x in core if x is not r]
            if trial and not sat(trial):
                core = trial
        return [r["name"] for r in core]

    def minimal_conflicts(self) -> list:
        """List of minimal contradictory cores — each a smallest set of
        rules that is already jointly unsatisfiable over the space."""
        cores = []
        for comp in self._components():
            rules = comp["rules"]
            if comp["too_big"] or comp["space_empty"] or not rules:
                continue
            if any(all(self._eval(r["expr"], w) for r in rules)
                   for w in self._possible_worlds(comp["vars"])):
                continue  # satisfiable — early-exits on the first model
            worlds = list(self._possible_worlds(comp["vars"]))
            cores.append(self._minimal_core(rules, worlds, self._eval))
        return cores

    def is_vacuous(self, rule) -> bool:
        """Does this rule (name or expression) hold in *every* possible
        world? Then it excludes nothing — it says nothing (4.461)."""
        expr = next((r["expr"] for r in self.rules if r["name"] == rule), rule) \
            if isinstance(rule, str) else rule
        return self.check_entailment([], expr)["valid"]

    def audit(self, detail_max_vars: int = 16) -> dict:
        """Health report on the ruleset itself.

        Returns:
            vacuous:     rules true in every possible world (say nothing)
            redundant:   [{rule, entailed_by}] — implied by the other rules.
                         `entailed_by` names single rules that alone imply
                         it; it is empty when only the *conjunction* of the
                         others does (the rule is still genuinely redundant).
            equivalent:  groups of rules with identical sense (4.431)
            conflicts:   minimal jointly-unsatisfiable rule sets
            impossible_space: constraint sets that admit no world at all
            partial:     components audited only for conflicts + vacuity
                         (too large for the O(worlds×rules²) redundancy pass)
            unauditable: components too large to enumerate (with var count)

        Cheap checks (satisfiability, vacuity) early-exit and run on any
        component up to the enumeration cap. The quadratic redundancy /
        equivalence pass is reserved for components with <= detail_max_vars
        variables; larger ones are reported under `partial`.
        """
        report = {"vacuous": [], "redundant": [], "equivalent": [],
                  "conflicts": [], "impossible_space": [], "partial": [],
                  "unauditable": []}

        # ── Vacuity is a per-rule property (depends only on that rule's
        # own variables rather than its whole component), so check it cheaply
        # and independently of the component join. ──
        for r in self.rules:
            vs = self._constraint_closure(self._get_vars(r["expr"]))
            if len(vs) <= self.MAX_VARS and self.check_entailment([], r["expr"])["valid"]:
                report["vacuous"].append(r["name"])
        report["vacuous"].sort()
        vac_all = set(report["vacuous"])

        for comp in self._components():
            rules = comp["rules"]
            if comp["too_big"]:
                if rules:
                    report["unauditable"].append(
                        {"vars": len(comp["vars"]), "rules": [r["name"] for r in rules]})
                continue
            if comp["space_empty"]:
                if rules:
                    report["impossible_space"].append(
                        sorted(c["name"] for c in comp["constraints"]))
                continue
            if not rules:
                continue

            # Contradiction: no possible world satisfies every rule.
            satisfiable = any(all(self._eval(r["expr"], w) for r in rules)
                              for w in self._possible_worlds(comp["vars"]))
            if not satisfiable:
                worlds = list(self._possible_worlds(comp["vars"]))
                report["conflicts"].append(self._minimal_core(rules, worlds, self._eval))
                continue  # redundancy is ill-defined once the set is contradictory

            if len(comp["vars"]) > detail_max_vars:
                report["partial"].append(
                    {"vars": len(comp["vars"]), "rules": [r["name"] for r in rules]})
                continue

            # ── Detailed pass: each rule's "sense" = permitted-world set ──
            worlds = list(self._possible_worlds(comp["vars"]))
            all_w = frozenset(range(len(worlds)))
            T = {r["name"]: frozenset(i for i, w in enumerate(worlds)
                                      if self._eval(r["expr"], w))
                 for r in rules}
            vac = {n for n in T if n in vac_all}

            # Same sense = identical permitted-world set (4.431).
            by_sense: dict = {}
            for n, t in T.items():
                if n not in vac:
                    by_sense.setdefault(t, []).append(n)
            equiv_members = set()
            for grp in by_sense.values():
                if len(grp) > 1:
                    report["equivalent"].append(sorted(grp))
                    equiv_members.update(grp)

            # Redundant: entailed by the conjunction of the other rules.
            names = [r["name"] for r in rules]
            for n in names:
                if n in vac or n in equiv_members:
                    continue
                others = [m for m in names if m != n]
                if not others:
                    continue
                conj = all_w
                for m in others:
                    conj &= T[m]
                if conj <= T[n]:
                    singles = sorted(m for m in others if T[m] <= T[n])
                    report["redundant"].append({"rule": n, "entailed_by": singles})

        return report

    # ── Serialization ────────────────────────────────────────────
    def to_json(self) -> dict:
        return {
            "version": "1.1",
            "propositions": self.propositions,
            "rules": [{"name": r["name"], "expr": r["expr"]} for r in self.rules],
            "constraints": [{"name": c["name"], "expr": c["expr"]} for c in self.constraints],
        }

    @classmethod
    def from_json(cls, data: dict) -> "TruthTableEngine":
        e = cls()
        for k, v in data.get("propositions", {}).items():
            if isinstance(v, str):
                e.prop(k, v)
            else:
                e.prop(k, v.get("description", ""), v.get("type", "feature"))
        for r in data.get("rules", []):
            e.rule(r["name"], r["expr"])
        for c in data.get("constraints", []):
            e.constrain(c["name"], c["expr"])
        return e

    def expr_to_string(self, expr) -> str:
        if isinstance(expr, str):
            return expr
        op, args = expr["op"], expr["args"]
        if op == "NOT":
            return f"¬{self.expr_to_string(args[0])}"
        symbols = {"AND": "∧", "OR": "∨", "IMPLIES": "→", "IFF": "↔", "XOR": "⊕"}
        return f"({self.expr_to_string(args[0])} {symbols.get(op, op)} {self.expr_to_string(args[1])})"
