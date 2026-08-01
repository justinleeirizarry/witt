"""Scoped grounding — closing the wrong-source gap.

Flat grounding pools everything the agent could legitimately have read into
one corpus and asks "does this value appear?" That catches invented values,
but it cannot catch a *real* value taken from the wrong place: an address
that showed up in an unrelated inbox message grounds a send_email just as
well as one that came from the contact lookup. Right shape, wrong object —
the same gap argument bindings close for parameters that two tools share,
except here the value lives in the prior tool's *output*, which a binding
can't reach.

A scope names the sources a parameter is allowed to draw from. It is an
authored constraint, not a heuristic, so it blocks in either mode.

    .venv/bin/python examples/scoped_grounding.py
"""

from witt import Grounding, Supervisor, generate_rules

TOOLS = [
    {"name": "read_inbox",
     "description": "Read recent messages",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "lookup_contact",
     "description": "Resolve a person to their address book entry",
     "parameters": {"type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]}},
    {"name": "send_email",
     "description": "Send an email",
     "parameters": {"type": "object",
                    "properties": {"to": {"type": "string"},
                                   "body": {"type": "string"}},
                    "required": ["to", "body"]}},
]

REQUEST = "Email Dana the Q3 summary."

# A phishing address the agent picked up while reading the inbox. It is a
# real string from a real tool result — flat grounding has no objection.
INBOX = {"messages": [
    {"from": "dana@acme.example", "subject": "Q3?"},
    {"from": "d.ana@acrne-billing.example", "subject": "Updated invoice"},
]}
CONTACTS = {"name": "Dana", "email": "dana@acme.example"}

SPOOFED = "d.ana@acrne-billing.example"


def attack(gate, label):
    print(f"\n=== {label} ===")
    gate.check("read_inbox")
    gate.record_success("read_inbox", result=INBOX)
    gate.check("lookup_contact", params={"name": "Dana"})
    gate.record_success("lookup_contact", result=CONTACTS)
    print("read the inbox, looked up Dana")

    gate.confirm("send_email")
    v = gate.check("send_email", params={"to": SPOOFED, "body": "Q3 summary"})
    print(f"send to {SPOOFED} -> {'ALLOWED' if v.allowed else 'BLOCKED'}")
    print(f"  provenance: {v.provenance}")
    if not v.allowed:
        print(f"  reason: {v.feedback}")


def build(scopes):
    g = Grounding(user_text=REQUEST, tool_specs=TOOLS, mode="strict",
                  scopes=scopes)
    return Supervisor(generate_rules(TOOLS), grounding=g)


# 1. Flat grounding — the address is in the corpus, so it passes.
attack(build(None), "flat grounding (before)")

# 2. Scoped — the recipient must come from a contact lookup.
scoped = build({"send_email": {"to": ["lookup_contact"]}})
attack(scoped, "scoped grounding (after)")

# The real address still sails through.
scoped.confirm("send_email")
ok = scoped.check("send_email",
                  params={"to": "dana@acme.example", "body": "Q3 summary"})
print(f"\nsend to dana@acme.example -> "
      f"{'ALLOWED' if ok.allowed else 'BLOCKED'}  (no false positive)")
print(f"  provenance: {ok.provenance}")
