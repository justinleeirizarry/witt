"""
The possibility space — declaring internal relations between propositions.

Wittgenstein assumed elementary propositions are logically independent
(Tractatus 2.061): "any one can be the case or not the case while
everything else remains the same." Real agent state isn't like that —
a session can't be both live and expired; a completed search always has
a result. He hit the same wall himself (the colour-exclusion problem,
6.3751) and it broke logical atomism.

Without declaring these relations, the truth table enumerates impossible
worlds, and a "counterexample" or "conflict" may be pure fantasy.
Declaring the space restricts reasoning to genuine possibilities.

Run: python examples/possibility_space.py
"""

from witt import TruthTableEngine

e = TruthTableEngine()

# Rules — what the agent may do.
e.rule("query needs a live session", e.IMPLIES("Call_query", "SessionLive"))

# The space — what is even possible. A session is live XOR expired.
e.incompatible("SessionLive", "SessionExpired")
# A completed search and its result stand or fall together.
e.coupled("Done_search", "Has_search_result")

print("1. A coherent, permitted state:")
r = e.validate_closed({"Call_query": True, "SessionLive": True})
print(f"   valid={r['valid']}  possible={r['possible']}\n")

print("2. A forbidden action (querying without a live session):")
r = e.validate_closed({"Call_query": True})
print(f"   valid={r['valid']}  violations={r['violations']}\n")

print("3. An IMPOSSIBLE state (session both live and expired):")
r = e.validate_closed({"SessionLive": True, "SessionExpired": True})
print(f"   valid={r['valid']}  possible={r['possible']}")
print(f"   space_violations={r['space_violations']}")
print("   → not a forbidden action; an incoherent world. Distinct fix.\n")

print("4. A half-set coupling (a 'done' search with no result):")
r = e.validate_closed({"Done_search": True})
print(f"   possible={r['possible']}  space_violations={r['space_violations']}\n")

print("5. Contingent becomes necessary once the space excludes a world:")
claim = e.NOT(e.AND("SessionLive", "SessionExpired"))
fresh = TruthTableEngine()
print(f"   ¬(live ∧ expired) entailed with no space? "
      f"{fresh.check_entailment([], claim)['valid']}")
print(f"   ...once they're declared incompatible?           "
      f"{e.check_entailment([], claim)['valid']}")
