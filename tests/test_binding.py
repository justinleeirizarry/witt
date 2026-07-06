"""Argument-bound dependency tests.

A plain dependency (`issue_refund` requires `get_order`) is satisfied by
*any* completed get_order — so an agent can fetch order A and refund order
B. A binding tightens that to object identity: issue_refund with order_id=V
requires a get_order that *completed with order_id=V*.
"""

import pytest
from witt import Supervisor, generate_rules, normalize_bindings


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


def _gate():
    engine = generate_rules(
        [GET_ORDER, ISSUE_REFUND],
        require_confirmation=["issue_refund"],
        bindings={"issue_refund": [{"tool": "get_order", "param": "order_id"}]},
    )
    return Supervisor(engine)


class TestNormalizeBindings:
    def test_dict_form(self):
        out = normalize_bindings(
            {"issue_refund": [{"tool": "get_order", "param": "order_id"}]})
        assert out == {"issue_refund":
                       [{"dep": "get_order", "param": "order_id",
                         "dep_param": "order_id"}]}

    def test_tuple_form_and_distinct_dep_param(self):
        out = normalize_bindings({"a": [("b", "x"), ("c", "x", "y")]})
        assert out["a"][0] == {"dep": "b", "param": "x", "dep_param": "x"}
        assert out["a"][1] == {"dep": "c", "param": "x", "dep_param": "y"}

    def test_none(self):
        assert normalize_bindings(None) == {}


class TestBinding:
    def test_bound_dep_blocks_mismatched_object(self):
        gate = _gate()
        # Fetch order A-4471.
        gate.check("get_order", params={"order_id": "A-4471"})
        gate.record_success("get_order", params={"order_id": "A-4471"})
        # Refund a DIFFERENT order — dependency 'get_order done' is satisfied,
        # but the object identity is not.
        gate.confirm("issue_refund")
        v = gate.check("issue_refund",
                       params={"order_id": "Z-0000", "amount": 10})
        assert not v.allowed
        assert v.binding_violations
        assert "Z-0000" in v.binding_violations[0]
        assert "A-4471" in v.binding_violations[0]  # reports what it did see

    def test_bound_dep_allows_matching_object(self):
        gate = _gate()
        gate.check("get_order", params={"order_id": "A-4471"})
        gate.record_success("get_order", params={"order_id": "A-4471"})
        gate.confirm("issue_refund")
        v = gate.check("issue_refund",
                       params={"order_id": "A-4471", "amount": 10})
        assert v.allowed, v.feedback

    def test_binding_reported_before_fetch(self):
        gate = _gate()
        gate.confirm("issue_refund")
        v = gate.check("issue_refund",
                       params={"order_id": "A-4471", "amount": 10})
        assert not v.allowed
        assert any("has not completed" in b for b in v.binding_violations)

    def test_check_then_record_binds_without_explicit_params(self):
        # The common flow: check(tool, params) then record_success(tool).
        # record_success reuses the checked params, so binding still works.
        gate = _gate()
        gate.check("get_order", params={"order_id": "A-4471"})
        gate.record_success("get_order")  # no params — reuse last check()
        gate.confirm("issue_refund")
        v = gate.check("issue_refund",
                       params={"order_id": "A-4471", "amount": 10})
        assert v.allowed, v.feedback

    def test_int_string_value_match(self):
        gate = _gate()
        gate.check("get_order", params={"order_id": 42})
        gate.record_success("get_order", params={"order_id": 42})
        gate.confirm("issue_refund")
        v = gate.check("issue_refund",
                       params={"order_id": "42", "amount": 10})
        assert v.allowed, v.feedback

    def test_missing_bind_param(self):
        gate = _gate()
        gate.confirm("issue_refund")
        v = gate.check("issue_refund", params={"amount": 10})
        assert not v.allowed
        assert v.binding_violations

    def test_no_bindings_is_inert(self):
        # Without bindings, behavior is unchanged: any get_order unlocks refund.
        engine = generate_rules(
            [GET_ORDER, ISSUE_REFUND], require_confirmation=["issue_refund"])
        gate = Supervisor(engine)
        gate.check("get_order", params={"order_id": "A-4471"})
        gate.record_success("get_order")
        gate.confirm("issue_refund")
        v = gate.check("issue_refund",
                       params={"order_id": "Z-9999", "amount": 10})
        assert v.allowed  # the gap this feature closes
        assert v.binding_violations == []
