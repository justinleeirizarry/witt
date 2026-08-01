"""Tests for witt.grounding — fabricated-argument detection."""

import pytest
from witt import (
    Grounding, Supervisor, generate_rules, infer_scopes_from_logs,
    normalize_scopes,
)

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


class TestSources:
    """The corpus is a set of named sources, not one blob."""

    def test_static_sources_are_named(self):
        g = Grounding(user_text="check NVDA", tool_specs=TOOLS,
                      config={"acct": "A-1"})
        assert g.sources() == [
            "user", "specs:get_stock_info", "specs:place_order",
            "specs:send_message", "config:acct"]

    def test_each_tool_spec_is_its_own_source(self):
        g = Grounding(tool_specs=TOOLS)
        assert g.where_grounded("receiver_id") == ["specs:send_message"]

    def test_unnamed_spec_falls_back_to_a_pooled_source(self):
        g = Grounding(tool_specs=[{"parameters": {"properties": {}}}])
        assert g.sources() == ["specs"]

    def test_non_dict_config_stays_one_source(self):
        g = Grounding(config="acct A-1")
        assert g.where_grounded("A-1") == ["config"]

    def test_where_grounded_reports_provenance(self):
        g = Grounding(user_text="check NVDA")
        assert g.where_grounded("NVDA") == ["user"]
        assert g.where_grounded("TSLA") == []

    def test_observe_names_its_source(self):
        g = Grounding()
        g.observe({"holdings": ["AAPL"]}, source="portfolio")
        assert g.where_grounded("AAPL") == ["portfolio"]

    def test_observe_defaults_to_a_generic_source(self):
        g = Grounding()
        g.observe({"holdings": ["AAPL"]})
        assert g.where_grounded("AAPL") == ["observed"]

    def test_observe_result_gets_a_call_ordinal(self):
        g = Grounding()
        assert g.observe_result("get_order", {"id": "A-1"}) == "get_order#1"
        assert g.observe_result("get_order", {"id": "A-2"}) == "get_order#2"
        assert g.where_grounded("A-2") == ["get_order#2"]

    def test_repeat_observations_accumulate_under_one_source(self):
        g = Grounding()
        g.observe("alpha", source="s")
        g.observe("beta", source="s")
        assert g.sources() == ["s"]
        assert g.where_grounded("beta") == ["s"]

    def test_value_in_several_sources_reports_all(self):
        g = Grounding(user_text="order A-1")
        g.observe_result("get_order", {"id": "A-1"})
        assert g.where_grounded("A-1") == ["user", "get_order#1"]

    def test_is_grounded_can_restrict_to_sources(self):
        g = Grounding(user_text="check NVDA")
        assert g.is_grounded("NVDA")
        assert not g.is_grounded("NVDA", sources=["portfolio"])

    def test_numbers_get_provenance_too(self):
        g = Grounding(user_text="transfer 100 dollars")
        assert g.where_grounded(100) == ["user"]

    def test_supervisor_records_provenance_on_the_verdict(self):
        gate, _ = make_gate(mode="warn")
        v = gate.check("get_stock_info", params={"symbol": "NVDA"})
        assert v.provenance == {"symbol": ["user"]}

    def test_provenance_names_the_producing_call(self):
        gate, _ = make_gate(mode="warn", user_text="sell my biggest holding")
        gate.check("get_stock_info", params={"symbol": "NVDA"})
        gate.record_success("get_stock_info", result={"top_holding": "AAPL"})
        v = gate.check("place_order",
                       params={"symbol": "AAPL", "order_type": "Sell",
                               "amount": 5})
        assert v.provenance["symbol"] == ["get_stock_info#1"]

    def test_provenance_empty_for_fabricated_value(self):
        gate, _ = make_gate(mode="warn")
        v = gate.check("get_stock_info", params={"symbol": "TSLA"})
        assert v.provenance == {"symbol": []}

    def test_track_provenance_off_keeps_detection(self):
        g = Grounding(user_text="check NVDA", track_provenance=False)
        assert g.ungrounded({"symbol": "TSLA"}) == [("symbol", "TSLA")]
        assert g.ungrounded({"symbol": "NVDA"}) == []
        assert g.check(None, {"symbol": "NVDA"})[1] == {}


class TestScopes:
    """A scope requires a param to come from a *specific* source, not
    merely from somewhere in the corpus."""

    def test_value_from_the_declared_source_passes(self):
        g = Grounding(scopes={"send_message": {"receiver_id": ["lookup"]}})
        g.observe_result("lookup", {"id": "USR-7"})
        assert g.ungrounded({"receiver_id": "USR-7"}, tool="send_message") == []

    def test_value_from_the_wrong_source_is_flagged(self):
        # The classic hole: grounded by an unrelated tool's output.
        g = Grounding(scopes={"send_message": {"receiver_id": ["lookup"]}})
        g.observe_result("read_inbox", {"sender": "USR-7"})
        vs, _ = g.check("send_message", {"receiver_id": "USR-7"})
        assert [v.kind for v in vs] == ["out_of_scope"]
        assert vs[0].found_in == ("read_inbox#1",)
        assert vs[0].expected == ("lookup",)

    def test_user_text_does_not_satisfy_a_result_scope(self):
        g = Grounding(user_text="message USR-7 for me",
                      scopes={"send_message": {"receiver_id": ["lookup"]}})
        vs, _ = g.check("send_message", {"receiver_id": "USR-7"})
        assert [v.kind for v in vs] == ["out_of_scope"]

    def test_scope_may_name_static_sources(self):
        g = Grounding(user_text="message USR-7 for me",
                      scopes={"send_message": {"receiver_id": ["user"]}})
        assert g.ungrounded({"receiver_id": "USR-7"}, tool="send_message") == []

    def test_any_call_of_the_scoped_tool_satisfies_it(self):
        g = Grounding(scopes={"send_message": {"receiver_id": ["lookup"]}})
        g.observe_result("lookup", {"id": "USR-1"})
        g.observe_result("lookup", {"id": "USR-7"})
        vs, _ = g.check("send_message", {"receiver_id": "USR-7"})
        assert vs == []

    def test_ungrounded_beats_out_of_scope(self):
        # A value in no source at all is fabricated, not misrouted.
        g = Grounding(scopes={"send_message": {"receiver_id": ["lookup"]}})
        vs, _ = g.check("send_message", {"receiver_id": "USR-9"})
        assert [v.kind for v in vs] == ["ungrounded"]

    def test_scope_applies_only_to_the_named_tool(self):
        g = Grounding(scopes={"send_message": {"receiver_id": ["lookup"]}})
        g.observe_result("read_inbox", {"sender": "USR-7"})
        assert g.ungrounded({"receiver_id": "USR-7"}, tool="other") == []

    def test_wildcard_tool_applies_everywhere(self):
        g = Grounding(scopes={"*": {"receiver_id": ["lookup"]}})
        g.observe_result("read_inbox", {"sender": "USR-7"})
        vs, _ = g.check("anything", {"receiver_id": "USR-7"})
        assert [v.kind for v in vs] == ["out_of_scope"]

    def test_tool_specific_scope_overrides_wildcard(self):
        g = Grounding(scopes={"*": {"id": ["a"]}, "t": {"id": ["b"]}})
        assert g.scope_for("t", "id") == ("b",)
        assert g.scope_for("other", "id") == ("a",)

    def test_unscoped_params_are_unaffected(self):
        g = Grounding(user_text="buy NVDA",
                      scopes={"place_order": {"amount": ["get_balance"]}})
        assert g.ungrounded({"symbol": "NVDA"}, tool="place_order") == []

    def test_normalize_scopes_accepts_a_bare_string(self):
        assert normalize_scopes({"t": {"p": "s"}}) == {"t": {"p": ("s",)}}

    def test_normalize_scopes_drops_empty_entries(self):
        assert normalize_scopes({"t": {}}) == {}
        assert normalize_scopes(None) == {}


class TestScopeEnforcement:
    """Scope violations are authored constraints, so they block in either
    mode — unlike heuristic ungroundedness."""

    def scoped_gate(self, mode):
        g = Grounding(user_text="message USR-7", tool_specs=TOOLS, mode=mode,
                      scopes={"send_message": {"receiver_id": ["lookup"]}})
        engine = generate_rules(TOOLS, auto_detect_destructive=False)
        return Supervisor(engine, grounding=g)

    def test_out_of_scope_blocks_in_warn_mode(self):
        gate = self.scoped_gate("warn")
        v = gate.check("send_message",
                       params={"receiver_id": "USR-7", "message": "hi"})
        assert not v.allowed
        assert "out of scope" in v.feedback

    def test_out_of_scope_blocks_in_strict_mode(self):
        gate = self.scoped_gate("strict")
        v = gate.check("send_message",
                       params={"receiver_id": "USR-7", "message": "hi"})
        assert not v.allowed

    def test_plain_ungrounded_still_only_warns(self):
        gate = self.scoped_gate("warn")
        v = gate.check("get_stock_info", params={"symbol": "TSLA"})
        assert v.allowed
        assert "Warning (ungrounded)" in v.feedback

    def test_scope_satisfied_by_a_real_result_allows(self):
        gate = self.scoped_gate("warn")
        gate.record_success("lookup", result={"id": "USR-7"})
        v = gate.check("send_message",
                       params={"receiver_id": "USR-7", "message": "hi"})
        assert v.allowed
        assert v.grounding_violations == []

    def test_scope_message_names_both_ends(self):
        gate = self.scoped_gate("warn")
        v = gate.check("send_message",
                       params={"receiver_id": "USR-7", "message": "hi"})
        assert "comes from ['user']" in v.grounding_violations[0]
        assert "must be grounded in ['lookup']" in v.grounding_violations[0]


class TestScopeMining:
    def run_log(self, pairs):
        """A fake Supervisor.log — [(tool, {param: [sources]}), ...]."""
        return [{"tool": t, "provenance": p} for t, p in pairs]

    def test_mines_the_observed_source(self):
        runs = [self.run_log([("send", {"to": ["lookup#1"]})])
                for _ in range(3)]
        assert infer_scopes_from_logs(runs) == {"send": {"to": ["lookup"]}}

    def test_call_ordinals_are_folded_away(self):
        runs = [self.run_log([("send", {"to": [f"lookup#{i}"]})])
                for i in range(1, 4)]
        assert infer_scopes_from_logs(runs) == {"send": {"to": ["lookup"]}}

    def test_union_of_sources_by_default(self):
        runs = [self.run_log([("send", {"to": ["lookup#1"]})]),
                self.run_log([("send", {"to": ["user"]})]),
                self.run_log([("send", {"to": ["user"]})])]
        assert infer_scopes_from_logs(runs) == {
            "send": {"to": ["lookup", "user"]}}

    def test_min_support_tightens_the_scope(self):
        runs = [self.run_log([("send", {"to": ["lookup#1"]})]),
                self.run_log([("send", {"to": ["user"]})]),
                self.run_log([("send", {"to": ["user"]})])]
        assert infer_scopes_from_logs(runs, min_support=0.5) == {
            "send": {"to": ["user"]}}

    def test_rare_params_are_skipped(self):
        runs = [self.run_log([("send", {"to": ["lookup#1"]})])]
        assert infer_scopes_from_logs(runs) == {}

    def test_ungrounded_observations_teach_nothing(self):
        runs = [self.run_log([("send", {"to": []})]) for _ in range(5)]
        assert infer_scopes_from_logs(runs) == {}

    def test_accepts_a_single_flat_log(self):
        flat = self.run_log([("send", {"to": ["lookup#1"]})] * 3)
        assert infer_scopes_from_logs(flat) == {"send": {"to": ["lookup"]}}

    def test_empty_input(self):
        assert infer_scopes_from_logs([]) == {}

    def test_round_trips_into_a_gate(self):
        runs = [self.run_log([("send_message", {"receiver_id": ["lookup#1"]})])
                for _ in range(3)]
        g = Grounding(user_text="message USR-7",
                      scopes=infer_scopes_from_logs(runs))
        vs, _ = g.check("send_message", {"receiver_id": "USR-7"})
        assert [v.kind for v in vs] == ["out_of_scope"]
