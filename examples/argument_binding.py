"""Argument-bound dependencies — closing the object-identity gap.

Earlier viability testing found: a plain dependency `issue_refund` requires
`get_order` is satisfied by *any* completed get_order. So an agent can fetch
order A-4471 and then refund a completely different order it never looked at.
The gate saw "a fetch happened," not "you fetched the thing you're refunding."

This demo runs the same attack twice — without and with a binding — driving
the Supervisor exactly as the MCP server does.

    .venv/bin/python examples/argument_binding.py
"""

from witt import Supervisor, generate_rules

GET_ORDER = {
    "name": "get_order",
    "description": "Fetch an order record",
    "parameters": {"type": "object",
                   "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]},
}
ISSUE_REFUND = {
    "name": "issue_refund",
    "description": "Refund money to the customer",
    "parameters": {"type": "object",
                   "properties": {"order_id": {"type": "string"},
                                  "amount": {"type": "number"}},
                   "required": ["order_id", "amount"]},
}


def attack(gate, label):
    print(f"\n=== {label} ===")
    # Agent fetches the order it was asked about.
    gate.check("get_order", params={"order_id": "A-4471"})
    gate.record_success("get_order", params={"order_id": "A-4471"})
    print("fetched order A-4471")

    # ...then refunds a DIFFERENT order (the bug), with a confirmation.
    gate.confirm("issue_refund")
    v = gate.check("issue_refund",
                   params={"order_id": "Z-0000-never-fetched", "amount": 9999999})
    print(f"refund Z-0000-never-fetched, $9,999,999 -> "
          f"{'ALLOWED' if v.allowed else 'BLOCKED'}")
    if not v.allowed:
        print(f"  reason: {v.feedback}")


# 1. Plain dependency — the object-identity gap is open.
plain = Supervisor(generate_rules(
    [GET_ORDER, ISSUE_REFUND],
    dependencies={"issue_refund": ["get_order"]},
    require_confirmation=["issue_refund"],
))
attack(plain, "plain dependency (before)")

# 2. Argument-bound dependency — the refund must target the fetched order.
bound = Supervisor(generate_rules(
    [GET_ORDER, ISSUE_REFUND],
    require_confirmation=["issue_refund"],
    bindings={"issue_refund": [{"tool": "get_order", "param": "order_id"}]},
))
attack(bound, "argument-bound dependency (after)")

# And the legitimate refund still sails through.
bound.confirm("issue_refund")
ok = bound.check("issue_refund", params={"order_id": "A-4471", "amount": 49.99})
print(f"\nlegitimate refund of the fetched order A-4471 -> "
      f"{'ALLOWED' if ok.allowed else 'BLOCKED'}  (no false positive)")
