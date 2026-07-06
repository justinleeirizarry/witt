"""
Auditing the rules themselves — Wittgenstein's two limiting cases (4.46).

The gate validates tool calls against the rules. But who validates the
rules? Among all truth-functions exactly two are degenerate: the
tautology (true in every world — "says nothing", 4.461) and the
contradiction (false in every world — unsatisfiable). Those are the two
ways a *ruleset* can be sick. audit() finds both, plus rules implied by
others (redundant) and rules with identical truth-conditions (the same
"sense", 4.431).

Run: python examples/rule_audit.py
"""

from witt import TruthTableEngine

e = TruthTableEngine()

# A tautology — excludes nothing, so it says nothing.
e.rule("always true", e.OR("Ok", e.NOT("Ok")))

# A redundant rule — follows from the other two (hypothetical syllogism).
e.rule("a needs b", e.IMPLIES("A", "B"))
e.rule("b needs c", e.IMPLIES("B", "C"))
e.rule("a needs c", e.IMPLIES("A", "C"))     # implied by the two above

# Two rules with the same sense (a rule and its contrapositive).
e.rule("guard", e.IMPLIES("Go", "Ready"))
e.rule("guard restated", e.IMPLIES(e.NOT("Ready"), e.NOT("Go")))

# A genuine contradiction, plus an innocent bystander sharing a variable.
e.rule("must sign", "Signed")
e.rule("must not sign", e.NOT("Signed"))
e.rule("bystander", e.IMPLIES("Draft", "Signed"))

report = e.audit()

print("Vacuous (say nothing):   ", report["vacuous"])
print("Redundant (implied):     ", report["redundant"])
print("Equivalent (same sense): ", report["equivalent"])
print("Conflicts (minimal core):", report["conflicts"])
print()
print("Note the conflict core is just the two contradictory rules — the")
print("bystander is correctly left out, even though it shares 'Signed'.")
