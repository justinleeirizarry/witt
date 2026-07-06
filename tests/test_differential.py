"""Differential test: witt vs. an independent SAT solver (z3).

witt reasons by exhaustive truth-table enumeration. z3 reasons by
DPLL/SMT search. They share no code. If they agree on thousands of random
formulas, witt's engine is correct in the only sense that matters —
not "passes the tests we wrote" but "computes the same function classical
logic does."

This proves *Claim A* (the engine is logically sound). It says nothing
about whether the rules catch real agent errors — that is Claim B, tested
in test_oracle.py.

Skipped automatically if z3 is not installed (`pip install z3-solver`).
"""

import random

import pytest

from witt import TruthTableEngine

z3 = pytest.importorskip("z3")

OPS2 = ["AND", "OR", "IMPLIES", "IFF", "XOR"]
SEEDS = list(range(300))


def to_z3(expr, vs):
    """Translate a witt expression into a z3 boolean expression."""
    if isinstance(expr, str):
        return vs[expr]
    op, args = expr["op"], expr["args"]
    if op == "NOT":
        return z3.Not(to_z3(args[0], vs))
    a, b = to_z3(args[0], vs), to_z3(args[1], vs)
    return {
        "AND": z3.And(a, b), "OR": z3.Or(a, b),
        "IMPLIES": z3.Implies(a, b), "IFF": a == b, "XOR": z3.Xor(a, b),
    }[op]


def random_expr(e, names, rng, depth=3):
    if depth == 0 or (rng.random() < 0.3 and depth < 3):
        v = rng.choice(names)
        return e.NOT(v) if rng.random() < 0.3 else v
    if rng.random() < 0.25:
        return e.NOT(random_expr(e, names, rng, depth - 1))
    op = rng.choice(OPS2)
    a = random_expr(e, names, rng, depth - 1)
    b = random_expr(e, names, rng, depth - 1)
    return {"op": op, "args": [a, b]}


def is_valid(z3_expr):
    """True iff z3_expr holds in every assignment (tautology)."""
    s = z3.Solver()
    s.add(z3.Not(z3_expr))
    return s.check() == z3.unsat


def is_unsat(z3_expr):
    s = z3.Solver()
    s.add(z3_expr)
    return s.check() == z3.unsat


class TestEvaluationMatchesZ3:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_eval_agrees_on_every_assignment(self, seed):
        rng = random.Random(seed)
        e = TruthTableEngine()
        names = [f"v{i}" for i in range(rng.randint(2, 4))]
        expr = random_expr(e, names, rng)
        vs = {n: z3.Bool(n) for n in names}
        z3e = to_z3(expr, vs)
        for i in range(1 << len(names)):
            assign = {n: bool(i >> j & 1) for j, n in enumerate(names)}
            tg = e._eval(expr, assign)
            zres = z3.is_true(z3.simplify(z3.substitute(
                z3e, *[(vs[n], z3.BoolVal(v)) for n, v in assign.items()])))
            assert tg == zres, (expr, assign, tg, zres)


class TestEntailmentMatchesZ3:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_entailment_agrees(self, seed):
        rng = random.Random(seed)
        e = TruthTableEngine()
        names = [f"v{i}" for i in range(rng.randint(2, 4))]
        premises = [random_expr(e, names, rng, 2) for _ in range(rng.randint(0, 3))]
        conclusion = random_expr(e, names, rng, 2)
        tg_valid = e.check_entailment(premises, conclusion)["valid"]
        vs = {n: z3.Bool(n) for n in names}
        z3_premises = [to_z3(p, vs) for p in premises]
        z3_goal = z3.Implies(z3.And(*z3_premises) if z3_premises else z3.BoolVal(True),
                             to_z3(conclusion, vs))
        assert tg_valid == is_valid(z3_goal)


class TestConflictsMatchZ3:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_conflicts_agree(self, seed):
        rng = random.Random(seed)
        e = TruthTableEngine()
        names = [f"v{i}" for i in range(rng.randint(2, 4))]
        rule_exprs = [random_expr(e, names, rng, 2) for _ in range(rng.randint(1, 4))]
        for i, x in enumerate(rule_exprs):
            e.rule(f"r{i}", x)
        tg_conflict = bool(e.find_conflicts())
        vs = {n: z3.Bool(n) for n in names}
        z3_unsat = is_unsat(z3.And(*[to_z3(x, vs) for x in rule_exprs]))
        assert tg_conflict == z3_unsat


class TestVacuityMatchesZ3:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_vacuity_agrees(self, seed):
        rng = random.Random(seed)
        e = TruthTableEngine()
        names = [f"v{i}" for i in range(rng.randint(2, 4))]
        expr = random_expr(e, names, rng, 2)
        e.rule("r", expr)
        vs = {n: z3.Bool(n) for n in names}
        assert e.is_vacuous("r") == is_valid(to_z3(expr, vs))


class TestPossibilitySpaceMatchesZ3:
    """The subtle one: witt's space filtering must agree with z3 when
    the space axioms are added to the solver as hard constraints."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_entailment_under_space_agrees(self, seed):
        rng = random.Random(seed)
        e = TruthTableEngine()
        names = [f"v{i}" for i in range(rng.randint(2, 4))]
        # random axioms defining the possibility space
        axioms = []
        if rng.random() < 0.7:
            picks = rng.sample(names, 2)
            kind = rng.choice(["incompatible", "coupled"])
            getattr(e, kind)(*picks, name=f"ax_{kind}")
            axioms.append((kind, picks))
        premises = [random_expr(e, names, rng, 2) for _ in range(rng.randint(0, 2))]
        conclusion = random_expr(e, names, rng, 2)

        tg = e.check_entailment(premises, conclusion)["valid"]

        vs = {n: z3.Bool(n) for n in names}
        z3_ax = []
        for kind, picks in axioms:
            a, b = vs[picks[0]], vs[picks[1]]
            z3_ax.append(z3.Not(z3.And(a, b)) if kind == "incompatible" else (a == b))
        z3_premises = z3_ax + [to_z3(p, vs) for p in premises]
        goal = z3.Implies(z3.And(*z3_premises) if z3_premises else z3.BoolVal(True),
                          to_z3(conclusion, vs))
        assert tg == is_valid(goal)
