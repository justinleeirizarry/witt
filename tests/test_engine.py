"""Engine correctness tests. All must pass 100% — the engine is deterministic."""

import pytest
from truthgate import TruthTableEngine


@pytest.fixture
def e():
    return TruthTableEngine()


# ── Operators ────────────────────────────────────────────────────
class TestOperators:
    def test_and(self, e):
        expr = e.AND("A", "B")
        assert e._eval(expr, {"A": True, "B": True}) is True
        assert e._eval(expr, {"A": True, "B": False}) is False
        assert e._eval(expr, {"A": False, "B": True}) is False
        assert e._eval(expr, {"A": False, "B": False}) is False

    def test_or(self, e):
        expr = e.OR("A", "B")
        assert e._eval(expr, {"A": True, "B": True}) is True
        assert e._eval(expr, {"A": True, "B": False}) is True
        assert e._eval(expr, {"A": False, "B": False}) is False

    def test_not(self, e):
        assert e._eval(e.NOT("A"), {"A": True}) is False
        assert e._eval(e.NOT("A"), {"A": False}) is True

    def test_implies(self, e):
        expr = e.IMPLIES("A", "B")
        assert e._eval(expr, {"A": True, "B": True}) is True
        assert e._eval(expr, {"A": True, "B": False}) is False
        assert e._eval(expr, {"A": False, "B": True}) is True
        assert e._eval(expr, {"A": False, "B": False}) is True

    def test_iff(self, e):
        expr = e.IFF("A", "B")
        assert e._eval(expr, {"A": True, "B": True}) is True
        assert e._eval(expr, {"A": True, "B": False}) is False
        assert e._eval(expr, {"A": False, "B": False}) is True

    def test_xor(self, e):
        expr = e.XOR("A", "B")
        assert e._eval(expr, {"A": True, "B": False}) is True
        assert e._eval(expr, {"A": True, "B": True}) is False

    def test_nested(self, e):
        # NOT(IMPLIES(A,B)) ≡ AND(A, NOT(B))
        lhs = e.NOT(e.IMPLIES("A", "B"))
        rhs = e.AND("A", e.NOT("B"))
        for a in (True, False):
            for b in (True, False):
                assign = {"A": a, "B": b}
                assert e._eval(lhs, assign) == e._eval(rhs, assign)

    def test_all_any(self, e):
        assert e._eval(e.ALL("A", "B", "C"), {"A": True, "B": True, "C": True}) is True
        assert e._eval(e.ALL("A", "B", "C"), {"A": True, "B": False, "C": True}) is False
        assert e._eval(e.ANY("A", "B", "C"), {"A": False, "B": False, "C": True}) is True
        assert e._eval(e.ANY("A", "B", "C"), {"A": False, "B": False, "C": False}) is False


# ── Entailment (classic valid/invalid arguments) ────────────────
class TestEntailment:
    def test_modus_ponens(self, e):
        r = e.check_entailment([e.IMPLIES("P", "Q"), "P"], "Q")
        assert r["valid"] is True

    def test_modus_tollens(self, e):
        r = e.check_entailment([e.IMPLIES("P", "Q"), e.NOT("Q")], e.NOT("P"))
        assert r["valid"] is True

    def test_hypothetical_syllogism(self, e):
        r = e.check_entailment(
            [e.IMPLIES("P", "Q"), e.IMPLIES("Q", "R")], e.IMPLIES("P", "R"))
        assert r["valid"] is True

    def test_disjunctive_syllogism(self, e):
        r = e.check_entailment([e.OR("P", "Q"), e.NOT("P")], "Q")
        assert r["valid"] is True

    def test_constructive_dilemma(self, e):
        r = e.check_entailment(
            [e.IMPLIES("P", "Q"), e.IMPLIES("R", "S"), e.OR("P", "R")],
            e.OR("Q", "S"))
        assert r["valid"] is True

    # Fallacies — must be detected as INVALID
    def test_affirming_consequent(self, e):
        r = e.check_entailment([e.IMPLIES("P", "Q"), "Q"], "P")
        assert r["valid"] is False
        assert len(r["counterexamples"]) > 0

    def test_denying_antecedent(self, e):
        r = e.check_entailment([e.IMPLIES("P", "Q"), e.NOT("P")], e.NOT("Q"))
        assert r["valid"] is False

    def test_illicit_disjunction(self, e):
        r = e.check_entailment([e.OR("P", "Q"), "P"], e.NOT("Q"))
        assert r["valid"] is False

    def test_false_cause(self, e):
        r = e.check_entailment(
            [e.IMPLIES("P", "Q"), e.IMPLIES("R", "Q"), "P"], "R")
        assert r["valid"] is False


# ── Validation ───────────────────────────────────────────────────
class TestValidation:
    def test_valid_passes(self, e):
        e.prop("Auth", "", "state").prop("Query", "", "action")
        e.rule("query needs auth", e.IMPLIES("Query", "Auth"))
        r = e.validate_closed({"Query": True, "Auth": True})
        assert r["valid"] is True

    def test_violation_caught(self, e):
        e.prop("Auth", "", "state").prop("Query", "", "action")
        e.rule("query needs auth", e.IMPLIES("Query", "Auth"))
        r = e.validate_closed({"Query": True, "Auth": False})
        assert r["valid"] is False
        assert "query needs auth" in r["violations"]

    def test_closed_world_defaults_false(self, e):
        e.rule("query needs auth", e.IMPLIES("Query", "Auth"))
        # Query not mentioned → False → rule vacuously true
        r = e.validate_closed({})
        assert r["valid"] is True

    def test_unregistered_var_bug_fixed(self, e):
        """The BFCL bug: rules referencing unregistered props must not
        create free variables in closed-world validation."""
        # Rule mentions props never registered via .prop()
        e.rule("ghost rule", e.IMPLIES("Call_ghost", "Has_ghost_param"))
        e.prop("Call_real", "", "action")
        # Validate an unrelated call — ghost rule must not fire
        r = e.validate_closed({"Call_real": True})
        assert r["valid"] is True

    def test_corrections_for_outputs(self, e):
        e.prop("IsVague", "", "feature")
        e.prop("Relevant", "", "output")
        e.rule("vague not relevant", e.IMPLIES("IsVague", e.NOT("Relevant")))
        r = e.validate_closed({"IsVague": True, "Relevant": True})
        assert r["valid"] is False
        assert r["corrections"] == {"Relevant": False}

    def test_empty_rules(self, e):
        r = e.validate_closed({"Anything": True})
        assert r["valid"] is True

    def test_multiple_violations(self, e):
        e.rule("r1", e.IMPLIES("X", "A"))
        e.rule("r2", e.IMPLIES("X", "B"))
        r = e.validate_closed({"X": True})
        assert set(r["violations"]) == {"r1", "r2"}


# ── Edge cases ───────────────────────────────────────────────────
class TestEdgeCases:
    def test_conflict_detection(self, e):
        e.rule("a true", e.IMPLIES(e.OR("A", e.NOT("A")), "A"))   # forces A
        e.rule("a false", e.IMPLIES(e.OR("A", e.NOT("A")), e.NOT("A")))  # forces ¬A
        assert e.find_conflicts() != []

    def test_no_conflict(self, e):
        e.rule("ok", e.IMPLIES("A", "B"))
        assert e.find_conflicts() == []

    def test_variable_explosion_guard(self, e):
        big = e.ALL(*[f"V{i}" for i in range(25)])
        with pytest.raises(ValueError, match="free variables"):
            e.check_entailment([big], "V0")

    def test_empty_prop_name_rejected(self, e):
        with pytest.raises(ValueError):
            e.prop("")

    def test_unknown_operator(self, e):
        with pytest.raises(ValueError, match="Unknown operator"):
            e._eval({"op": "NAND", "args": ["A", "B"]}, {"A": True, "B": True})

    def test_deep_nesting(self, e):
        expr = "A"
        for _ in range(6):
            expr = e.NOT(expr)
        assert e._eval(expr, {"A": True}) is True  # even number of NOTs


# ── Serialization ────────────────────────────────────────────────
class TestSerialization:
    def test_roundtrip(self, e):
        e.prop("Auth", "auth desc", "state")
        e.prop("Query", "query desc", "action")
        e.rule("query needs auth", e.IMPLIES("Query", "Auth"))

        data = e.to_json()
        e2 = TruthTableEngine.from_json(data)

        r1 = e.validate_closed({"Query": True, "Auth": False})
        r2 = e2.validate_closed({"Query": True, "Auth": False})
        assert r1["valid"] == r2["valid"]
        assert r1["violations"] == r2["violations"]

    def test_import_string_props(self):
        data = {"propositions": {"A": "desc"}, "rules": []}
        e = TruthTableEngine.from_json(data)
        assert "A" in e.propositions

    def test_expr_to_string(self, e):
        s = e.expr_to_string(e.IMPLIES(e.AND("A", "B"), e.NOT("C")))
        assert s == "((A ∧ B) → ¬C)"
