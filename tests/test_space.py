"""The possibility space — internal relations between propositions.

Wittgenstein assumed elementary propositions are logically independent
(Tractatus 2.061). They usually aren't (the colour-exclusion problem,
6.3751). These tests cover the fix: enumeration ranges only over
genuinely possible worlds, so counterexamples and conflicts are never
drawn from impossible ones.
"""

import pytest
from witt import TruthTableEngine, Supervisor


@pytest.fixture
def e():
    return TruthTableEngine()


class TestIncoherentState:
    def test_incompatible_state_is_impossible(self, e):
        e.incompatible("Authenticated", "SessionExpired")
        r = e.validate_closed({"Authenticated": True, "SessionExpired": True})
        assert r["possible"] is False
        assert r["valid"] is False
        assert r["space_violations"]

    def test_incompatible_state_alone_is_possible(self, e):
        e.incompatible("Authenticated", "SessionExpired")
        r = e.validate_closed({"Authenticated": True})
        assert r["possible"] is True
        assert r["valid"] is True

    def test_coupled_props_stand_or_fall_together(self, e):
        e.coupled("Done_search", "Has_search_result")
        # Only one set → coupling broken → impossible world
        assert e.validate_closed({"Done_search": True})["possible"] is False
        # Both set → coherent
        assert e.validate_closed(
            {"Done_search": True, "Has_search_result": True})["possible"] is True

    def test_one_of_requires_exactly_one(self, e):
        e.one_of("StatusOpen", "StatusClosed", "StatusPending")
        assert e.validate_closed({"StatusOpen": True})["possible"] is True
        assert e.validate_closed({})["possible"] is False  # none
        assert e.validate_closed(
            {"StatusOpen": True, "StatusClosed": True})["possible"] is False  # two


class TestCounterexamplesAreGenuine:
    def test_contingent_becomes_necessary_under_the_space(self, e):
        """The colour-exclusion payoff: ¬(A ∧ B) is merely contingent in
        full logical space, but becomes a necessary truth once the space
        excludes the A∧B world."""
        claim = e.NOT(e.AND("A", "B"))
        assert e.check_entailment([], claim)["valid"] is False  # A=B=True refutes
        e.incompatible("A", "B")
        assert e.check_entailment([], claim)["valid"] is True   # that world is gone

    def test_rule_not_flagged_when_only_impossible_worlds_falsify_it(self, e):
        """A rule violated only in an impossible world must pass."""
        # "if A then not B" is falsified only by the A∧B world.
        e.rule("A excludes B", e.IMPLIES("A", e.NOT("B")))
        r_open = e.validate({"A": True})          # B free → B=True falsifies
        assert r_open["valid"] is False
        e.incompatible("A", "B")                   # remove the offending world
        r_closed = e.validate({"A": True})
        assert r_closed["valid"] is True


class TestConflictsRespectSpace:
    def test_conflict_only_under_the_space(self, e):
        e.rule("A required", "A")
        e.rule("B required", "B")
        assert e.find_conflicts() == []            # A=B=True satisfies both
        e.incompatible("A", "B")                   # now no possible world does
        assert e.find_conflicts() != []


class TestSerializationWithConstraints:
    def test_roundtrip_preserves_space(self, e):
        e.incompatible("A", "B")
        e.rule("needs c", e.IMPLIES("Go", "C"))
        e2 = TruthTableEngine.from_json(e.to_json())
        assert e2.validate_closed({"A": True, "B": True})["possible"] is False
        assert e2.validate_closed({"Go": True, "C": False})["valid"] is False


class TestSupervisorReportsIncoherence:
    def test_impossible_state_blocks_with_distinct_feedback(self):
        e = TruthTableEngine()
        e.incompatible("Authenticated", "SessionExpired")
        gate = Supervisor(e)
        gate.state.set("Authenticated", True).set("SessionExpired", True)
        v = gate.check("any_tool")
        assert not v.allowed
        assert v.space_violations
        assert "Incoherent state" in v.feedback
