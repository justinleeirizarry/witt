"""Tests for witt.grounding — fabricated-argument detection."""

import pytest
from witt import Grounding, Supervisor, generate_rules

TOOLS = [
    {"name": "get_stock_info",
     "parameters": {"type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"]}},
    {"name": "place_order",
     "parameters": {"type": "object",
                    "properties": {"symbol": {"type": "string"},
                                   "order_type": {"type": "string"},
                                   "amount": {"type": "integer"}},
                    "required": ["symbol", "order_type", "amount"]}},
    {"name": "send_message",
     "parameters": {"type": "object",
                    "properties": {"receiver_id": {"type": "string"},
                                   "message": {"type": "string"}},
                    "required": ["receiver_id", "message"]}},
]


def make_gate(mode="warn", user_text="Buy 100 shares of NVDA please"):
    g = Grounding(user_text=user_text, tool_specs=TOOLS, mode=mode)
    engine = generate_rules(TOOLS, auto_detect_destructive=False)
    return Supervisor(engine, grounding=g), g


class TestGroundingUnit:
    def test_user_text_grounds_value(self):
        g = Grounding(user_text="check NVDA for me")
        assert g.ungrounded({"symbol": "NVDA"}) == []

    def test_fabricated_value_flagged(self):
        g = Grounding(user_text="check NVDA for me")
        assert g.ungrounded({"symbol": "TSLA"}) == [("symbol", "TSLA")]

    def test_case_insensitive(self):
        g = Grounding(user_text="check nvda for me")
        assert g.ungrounded({"symbol": "NVDA"}) == []

    def test_tool_spec_grounds_enums(self):
        # 'order_type' values often come from the schema, not the user.
        spec = [{"name": "place_order",
                 "parameters": {"properties": {
                     "order_type": {"enum": ["Buy", "Sell"]}}}}]
        g = Grounding(tool_specs=spec)
        assert g.ungrounded({"order_type": "Buy"}) == []

    def test_config_grounds_values(self):
        g = Grounding(config={"files": ["report.txt"]})
        assert g.ungrounded({"file_name": "report.txt"}) == []

    def test_observe_result_grounds_later_values(self):
        g = Grounding(user_text="what's in my portfolio?")
        assert g.ungrounded({"symbol": "AAPL"})  # not yet grounded
        g.observe({"holdings": ["AAPL", "GOOG"]})
        assert g.ungrounded({"symbol": "AAPL"}) == []

    def test_freeform_params_skipped(self):
        g = Grounding(user_text="say hi to bob")
        assert g.ungrounded({"message": "Fabricated_greeting"}) == []

    def test_multiword_values_skipped(self):
        g = Grounding(user_text="irrelevant")
        assert g.ungrounded({"note_field": "two words"}) == []

    def test_dates_skipped_by_default(self):
        # "November 15th" -> "2026-11-15" is translation, not fabrication.
        g = Grounding(user_text="book it for November 15th")
        assert g.ungrounded({"travel_date": "2026-11-15"}) == []

    def test_booleans_never_checked(self):
        g = Grounding(user_text="irrelevant")
        assert g.ungrounded({"flag": True, "enabled": False}) == []

    def test_ungrounded_number_flagged(self):
        g = Grounding(user_text="transfer 100 dollars")
        assert g.ungrounded({"amount": 107}) == [("amount", 107)]

    def test_grounded_number_passes(self):
        g = Grounding(user_text="transfer 100 dollars")
        assert g.ungrounded({"amount": 100}) == []

    def test_number_not_matched_inside_longer_number(self):
        g = Grounding(user_text="order 2107 please")
        assert g.ungrounded({"amount": 107}) == [("amount", 107)]

    def test_int_float_forms_equivalent(self):
        g = Grounding(user_text="price is 99.0 total")
        assert g.ungrounded({"amount": 99}) == []

    def test_small_numbers_skipped(self):
        g = Grounding(user_text="irrelevant")
        assert g.ungrounded({"page": 2, "count": 5}) == []

    def test_numbers_from_results_ground(self):
        g = Grounding(user_text="sell it all")
        assert g.ungrounded({"quantity": 250})
        g.observe({"shares_owned": 250})
        assert g.ungrounded({"quantity": 250}) == []

    def test_check_numbers_off(self):
        g = Grounding(user_text="irrelevant", check_numbers=False)
        assert g.ungrounded({"amount": 107}) == []

    def test_short_values_skipped(self):
        g = Grounding(user_text="irrelevant", min_len=3)
        assert g.ungrounded({"x": "ab"}) == []

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            Grounding(mode="block")


class TestSupervisorIntegration:
    def test_warn_mode_allows_but_reports(self):
        gate, _ = make_gate(mode="warn")
        v = gate.check("get_stock_info", params={"symbol": "TSLA"})
        assert v.allowed  # warn never blocks
        assert v.grounding_violations
        assert "fabricated" in v.feedback

    def test_strict_mode_blocks(self):
        gate, _ = make_gate(mode="strict")
        v = gate.check("get_stock_info", params={"symbol": "TSLA"})
        assert not v.allowed
        assert v.grounding_violations
        # regular rule violations should be empty — this is grounding only
        assert v.violations == []

    def test_grounded_value_passes_strict(self):
        gate, _ = make_gate(mode="strict")
        v = gate.check("get_stock_info", params={"symbol": "NVDA"})
        assert v.allowed
        assert v.grounding_violations == []

    def test_record_success_result_feeds_corpus(self):
        gate, _ = make_gate(mode="strict",
                            user_text="sell my biggest holding")
        v = gate.check("get_stock_info", params={"symbol": "AAPL"})
        assert not v.allowed  # AAPL fabricated so far
        # Suppose a portfolio tool returned it instead:
        gate.check("get_stock_info", params={"symbol": "NVDA"})
        gate.record_success("get_stock_info",
                            result={"top_holding": "AAPL", "price": 190})
        v2 = gate.check("place_order",
                        params={"symbol": "AAPL", "order_type": "Sell",
                                "amount": 5})
        assert v2.grounding_violations == []

    def test_no_grounding_is_default_and_unchanged(self):
        engine = generate_rules(TOOLS, auto_detect_destructive=False)
        gate = Supervisor(engine)
        v = gate.check("get_stock_info", params={"symbol": "ZZZZ"})
        assert v.allowed
        assert v.grounding_violations == []

    def test_grounding_and_rules_compose(self):
        gate, _ = make_gate(mode="strict")
        # missing required param AND fabricated value on another
        v = gate.check("place_order", params={"symbol": "TSLA"})
        assert not v.allowed
        assert v.violations          # missing order_type/amount
        assert v.grounding_violations  # TSLA fabricated
