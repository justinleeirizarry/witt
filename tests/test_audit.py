"""Rule auditing — the two limiting cases (Tractatus 4.46) applied to
the ruleset itself, plus redundancy and sameness of sense (4.431).

A tautology "says nothing"; a contradiction is unsatisfiable. Those are
the two ways a *ruleset* can be sick. audit() also finds rules implied
by others (redundant), rules with identical truth-conditions
(equivalent), and — crucially — does it all via connected-component
decomposition so it scales past the 2^22 enumeration wall.
"""

import pytest
from witt import TruthTableEngine


@pytest.fixture
def e():
    return TruthTableEngine()


class TestVacuous:
    def test_tautology_says_nothing(self, e):
        e.rule("empty", e.OR("A", e.NOT("A")))
        assert e.is_vacuous("empty") is True
        assert "empty" in e.audit()["vacuous"]

    def test_impossible_premise_is_vacuous(self, e):
        e.incompatible("A", "B")                       # A∧B can never hold
        e.rule("dead", e.IMPLIES(e.AND("A", "B"), "C"))  # fires only on A∧B
        assert e.is_vacuous("dead") is True
        assert "dead" in e.audit()["vacuous"]

    def test_real_rule_is_not_vacuous(self, e):
        e.rule("live", e.IMPLIES("A", "B"))
        assert e.is_vacuous("live") is False
        assert e.audit()["vacuous"] == []


class TestRedundant:
    def test_hypothetical_syllogism_redundancy(self, e):
        e.rule("a->b", e.IMPLIES("A", "B"))
        e.rule("b->c", e.IMPLIES("B", "C"))
        e.rule("a->c", e.IMPLIES("A", "C"))   # follows from the other two
        red = {r["rule"] for r in e.audit()["redundant"]}
        assert red == {"a->c"}

    def test_subsumption_names_the_stronger_rule(self, e):
        e.rule("weak", e.IMPLIES("A", "B"))
        e.rule("strong", e.IMPLIES("A", e.AND("B", "C")))  # implies weak alone
        red = e.audit()["redundant"]
        assert len(red) == 1
        assert red[0]["rule"] == "weak"
        assert red[0]["entailed_by"] == ["strong"]


class TestEquivalent:
    def test_contrapositive_has_same_sense(self, e):
        e.rule("fwd", e.IMPLIES("A", "B"))
        e.rule("contra", e.IMPLIES(e.NOT("B"), e.NOT("A")))
        groups = e.audit()["equivalent"]
        assert ["contra", "fwd"] in [sorted(g) for g in groups]


class TestConflicts:
    def test_minimal_core_excludes_innocent_rules(self, e):
        e.rule("pos", "A")
        e.rule("neg", e.NOT("A"))
        e.rule("bystander", e.IMPLIES("B", "A"))
        cores = e.audit()["conflicts"]
        assert any(set(c) == {"pos", "neg"} for c in cores)
        # the bystander is not part of the minimal contradiction
        assert all("bystander" not in c for c in cores)

    def test_conflict_only_under_the_space(self, e):
        e.rule("need A", "A")
        e.rule("need B", "B")
        assert e.audit()["conflicts"] == []
        e.incompatible("A", "B")
        assert any(set(c) == {"need A", "need B"} for c in e.audit()["conflicts"])


class TestScaleAndLimits:
    def test_scales_past_enumeration_wall(self, e):
        # 100 variable-disjoint rules → 200 variables total. A single
        # truth table (2^200) is impossible; component decomposition
        # makes each a trivial 2-variable check.
        for i in range(100):
            e.rule(f"r{i}", e.IMPLIES(f"Call_{i}", f"Has_{i}"))
        rep = e.audit()
        assert rep["conflicts"] == []
        assert rep["redundant"] == []
        assert rep["vacuous"] == []
        assert rep["unauditable"] == []

    def test_oversized_component_flagged_not_crashed(self, e):
        e.rule("big", e.ALL(*[f"V{i}" for i in range(25)]))  # 25 vars, one component
        rep = e.audit()
        assert any("big" in u["rules"] for u in rep["unauditable"])

    def test_impossible_space_reported(self, e):
        e.constrain("A holds", "A")
        e.constrain("A fails", e.NOT("A"))   # no world for A
        e.rule("uses A", e.IMPLIES("A", "C"))
        assert e.audit()["impossible_space"]

    def test_midsize_component_reported_partial(self, e):
        # 18 vars: past the detailed-redundancy threshold (16) but under
        # the 22-var enumeration cap. Conflicts + vacuity still run; the
        # quadratic redundancy pass is skipped and the component is 'partial'.
        ante = e.ALL(*[f"v{i}" for i in range(9)])
        cons = e.ANY(*[f"v{i}" for i in range(9, 18)])
        e.rule("mid", e.IMPLIES(ante, cons))
        rep = e.audit()
        assert any("mid" in p["rules"] for p in rep["partial"])
        assert rep["unauditable"] == []      # under the cap, so not unauditable
        assert rep["conflicts"] == []        # cheap checks still ran
        assert rep["vacuous"] == []
