"""Supervisor scenario tests + autogen tests."""

import pytest
from witt import (
    Supervisor, StateTracker, TruthTableEngine,
    generate_rules, infer_dependencies_from_traces, ContradictoryRuleset,
)


# ── Autogen ──────────────────────────────────────────────────────
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["location"],
    },
}
DELETE_TOOL = {
    "name": "delete_record",
    "description": "Delete a record from the database",
    "parameters": {
        "type": "object",
        "properties": {"record_id": {"type": "string"}},
        "required": ["record_id"],
    },
}
SEARCH_TOOL = {
    "name": "search_web",
    "description": "Search the web",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
SUMMARIZE_TOOL = {
    "name": "summarize",
    "description": "Summarize retrieved content",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class TestAutogen:
    def test_required_param_rule(self):
        engine = generate_rules([WEATHER_TOOL])
        gate = Supervisor(engine)
        # Missing required param → blocked
        v = gate.check("get_weather")
        assert not v.allowed
        assert any("location" in viol for viol in v.violations)
        # Param present → allowed
        v2 = gate.check("get_weather", params={"location": "Tokyo"})
        assert v2.allowed

    def test_destructive_autodetect(self):
        engine = generate_rules([DELETE_TOOL])
        gate = Supervisor(engine)
        v = gate.check("delete_record", params={"record_id": "42"})
        assert not v.allowed  # needs confirmation
        gate.confirm()
        v2 = gate.check("delete_record", params={"record_id": "42"})
        assert v2.allowed

    def test_destructive_detection_matches_whole_tokens_only(self):
        from witt.autogen import is_destructive
        # true positives — verb is a whole token in the name
        assert is_destructive("delete_record")
        assert is_destructive("send_email")
        assert is_destructive("bookFlight")      # camelCase split
        assert is_destructive("cancel-order")    # kebab split
        # false positives the old substring match produced — now avoided
        assert not is_destructive("get_payment_status")   # 'pay' not a token
        assert not is_destructive("bookmark_page")        # 'book' not a token
        assert not is_destructive("get_postal_code")      # 'post' not a token
        assert not is_destructive("list_buyers")          # 'buy' not a token

    def test_explicit_dependency(self):
        engine = generate_rules(
            [SEARCH_TOOL, SUMMARIZE_TOOL],
            dependencies={"summarize": ["search_web"]},
        )
        gate = Supervisor(engine)
        v = gate.check("summarize")
        assert not v.allowed
        assert any("search_web" in viol for viol in v.violations)

        gate.check("search_web", params={"query": "x"})
        gate.record_success("search_web")
        v2 = gate.check("summarize")
        assert v2.allowed

    def test_dependency_mining(self):
        traces = [
            ["auth", "query"],
            ["auth", "query", "report"],
            ["auth", "query"],
            ["auth", "browse", "query"],
        ]
        deps = infer_dependencies_from_traces(traces, min_support=0.95)
        assert "auth" in deps.get("query", [])

    def test_engine_extension(self):
        base = TruthTableEngine()
        base.prop("Custom", "custom state", "state")
        engine = generate_rules([SEARCH_TOOL], engine=base)
        assert "Custom" in engine.propositions
        assert "Call_search_web" in engine.propositions

    def test_state_space_couples_completion_and_result(self):
        engine = generate_rules([SEARCH_TOOL])   # model_state_space on by default
        assert engine.constraints, "expected a completion coupling constraint"
        # Half-set completion (done, no result) is an impossible state.
        r = engine.validate_closed({"Done_search_web": True})
        assert r["possible"] is False
        # record_success sets both together → coherent again.
        gate = Supervisor(engine)
        gate.record_success("search_web")
        assert gate.check("search_web", params={"query": "x"}).allowed

    def test_state_space_opt_out(self):
        engine = generate_rules([SEARCH_TOOL], model_state_space=False)
        assert engine.constraints == []


class TestStrictSupervisor:
    def test_strict_raises_on_contradiction(self):
        e = TruthTableEngine()
        e.rule("must", "Go")
        e.rule("must not", e.NOT("Go"))
        with pytest.raises(ContradictoryRuleset):
            Supervisor(e, strict=True)

    def test_strict_raises_on_impossible_space(self):
        e = TruthTableEngine()
        e.constrain("holds", "A")
        e.constrain("fails", e.NOT("A"))
        e.rule("uses", e.IMPLIES("A", "B"))
        with pytest.raises(ContradictoryRuleset):
            Supervisor(e, strict=True)

    def test_strict_warns_on_vacuous_rule(self):
        e = TruthTableEngine()
        e.rule("live", e.IMPLIES("A", "B"))
        e.rule("empty", e.OR("X", e.NOT("X")))
        with pytest.warns(UserWarning, match="vacuous"):
            Supervisor(e, strict=True)

    def test_non_strict_never_raises(self):
        e = TruthTableEngine()
        e.rule("must", "Go")
        e.rule("must not", e.NOT("Go"))
        Supervisor(e)          # no strict → constructs fine

    def test_clean_ruleset_is_silent(self, recwarn):
        engine = generate_rules([SEARCH_TOOL, SUMMARIZE_TOOL],
                                dependencies={"summarize": ["search_web"]})
        Supervisor(engine, strict=True)   # clean → no raise, no warning
        assert len(recwarn) == 0

    def test_audit_delegates(self):
        engine = generate_rules([SEARCH_TOOL])
        report = Supervisor(engine).audit()
        assert set(report) >= {"vacuous", "redundant", "conflicts", "equivalent"}


# ── Full workflow scenarios (from our validated suite) ──────────
@pytest.fixture
def workflow_gate():
    """Email + DB + weather agent with the rules we proved out."""
    e = TruthTableEngine()
    # State
    for p in ["Authenticated", "HasUserLocation", "EmailComposed",
              "UserConfirmed", "RecordDeleted", "SessionExpired",
              "QuotaAvailable", "HasDBConnection", "ClarificationNeeded",
              "ClarificationResolved", "Done_search_web"]:
        e.prop(p, "", "state")
    # Actions
    for a in ["Call_get_weather", "Call_query_db", "Call_send_email",
              "Call_compose_email", "Call_delete_record", "Call_read_record",
              "Call_summarize", "Call_execute"]:
        e.prop(a, "", "action")

    e.rule("weather needs location", e.IMPLIES("Call_get_weather", "HasUserLocation"))
    e.rule("weather needs quota", e.IMPLIES("Call_get_weather", "QuotaAvailable"))
    e.rule("db needs auth", e.IMPLIES("Call_query_db", "Authenticated"))
    e.rule("db needs connection", e.IMPLIES("Call_query_db", "HasDBConnection"))
    e.rule("no db on expired session",
           e.IMPLIES("SessionExpired", e.NOT("Call_query_db")))
    e.rule("send needs compose", e.IMPLIES("Call_send_email", "EmailComposed"))
    e.rule("send needs confirmation", e.IMPLIES("Call_send_email", "UserConfirmed"))
    e.rule("delete needs confirmation", e.IMPLIES("Call_delete_record", "UserConfirmed"))
    e.rule("no read after delete", e.IMPLIES("RecordDeleted", e.NOT("Call_read_record")))
    e.rule("summarize needs search", e.IMPLIES("Call_summarize", "Done_search_web"))
    e.rule("ambiguity blocks execution",
           e.IMPLIES(e.AND("ClarificationNeeded", e.NOT("ClarificationResolved")),
                     e.NOT("Call_execute")))
    return Supervisor(e)


class TestWorkflows:
    def _propose(self, gate, action, **state):
        for k, v in state.items():
            gate.state.set(k, v)
        # bypass Call_ prefixing since fixture uses raw names
        props = gate.state.snapshot()
        props[action] = True
        r = gate.engine.validate_closed(props)
        return r

    def test_weather_blocked_without_location(self, workflow_gate):
        r = self._propose(workflow_gate, "Call_get_weather", QuotaAvailable=True)
        assert not r["valid"]
        assert "weather needs location" in r["violations"]

    def test_weather_allowed_with_location(self, workflow_gate):
        r = self._propose(workflow_gate, "Call_get_weather",
                          HasUserLocation=True, QuotaAvailable=True)
        assert r["valid"]

    def test_email_full_workflow(self, workflow_gate):
        g = workflow_gate
        # Send before compose → blocked (two violations)
        r = self._propose(g, "Call_send_email")
        assert not r["valid"]
        assert set(r["violations"]) >= {"send needs compose", "send needs confirmation"}
        # Compose → allowed
        r = self._propose(g, "Call_compose_email")
        assert r["valid"]
        # Composed but unconfirmed → blocked
        r = self._propose(g, "Call_send_email", EmailComposed=True)
        assert not r["valid"]
        assert r["violations"] == ["send needs confirmation"]
        # Confirmed → allowed
        r = self._propose(g, "Call_send_email", UserConfirmed=True)
        assert r["valid"]

    def test_db_auth_chain(self, workflow_gate):
        r = self._propose(workflow_gate, "Call_query_db", HasDBConnection=True)
        assert not r["valid"]
        r = self._propose(workflow_gate, "Call_query_db", Authenticated=True)
        assert r["valid"]

    def test_expired_session_blocks(self, workflow_gate):
        r = self._propose(workflow_gate, "Call_query_db",
                          Authenticated=True, HasDBConnection=True,
                          SessionExpired=True)
        assert not r["valid"]
        assert "no db on expired session" in r["violations"]

    def test_read_after_delete_blocked(self, workflow_gate):
        r = self._propose(workflow_gate, "Call_read_record", RecordDeleted=True)
        assert not r["valid"]

    def test_summarize_needs_search(self, workflow_gate):
        r = self._propose(workflow_gate, "Call_summarize")
        assert not r["valid"]
        r = self._propose(workflow_gate, "Call_summarize", Done_search_web=True)
        assert r["valid"]

    def test_ambiguity_gate(self, workflow_gate):
        r = self._propose(workflow_gate, "Call_execute", ClarificationNeeded=True)
        assert not r["valid"]
        r = self._propose(workflow_gate, "Call_execute", ClarificationResolved=True)
        assert r["valid"]


# ── Supervisor mechanics ─────────────────────────────────────────
class TestSupervisor:
    def test_stats(self):
        engine = generate_rules([SEARCH_TOOL])
        gate = Supervisor(engine)
        gate.check("search_web")                                  # blocked (no query)
        gate.check("search_web", params={"query": "x"})           # allowed
        s = gate.stats()
        assert s["checks"] == 2
        assert s["blocked"] == 1
        assert s["allowed"] == 1
        assert s["avg_latency_us"] > 0

    def test_verdict_bool(self):
        engine = generate_rules([SEARCH_TOOL])
        gate = Supervisor(engine)
        assert bool(gate.check("search_web", params={"query": "x"})) is True
        assert bool(gate.check("search_web")) is False

    def test_feedback_message(self):
        engine = generate_rules([SEARCH_TOOL])
        gate = Supervisor(engine)
        v = gate.check("search_web")
        assert v.feedback.startswith("Blocked:")
        assert "query" in v.feedback

    def test_confirm_unconfirm_cycle(self):
        engine = generate_rules([DELETE_TOOL])
        gate = Supervisor(engine)
        gate.confirm("delete_record")
        assert gate.check("delete_record", params={"record_id": "1"}).allowed
        gate.unconfirm("delete_record")
        assert not gate.check("delete_record", params={"record_id": "1"}).allowed

    def test_confirm_no_arg_uses_last_checked_tool(self):
        engine = generate_rules([DELETE_TOOL])
        gate = Supervisor(engine)
        assert not gate.check("delete_record", params={"record_id": "1"}).allowed
        gate.confirm()   # no arg → confirms the tool just checked
        assert gate.check("delete_record", params={"record_id": "1"}).allowed

    def test_confirmation_is_scoped_per_tool(self):
        # The #4 fix: confirming one destructive action must not authorize
        # a different one.
        send_tool = {"name": "send_email", "description": "Send an email",
                     "parameters": {"type": "object",
                                    "properties": {"to": {"type": "string"}},
                                    "required": ["to"]}}
        engine = generate_rules([DELETE_TOOL, send_tool])
        gate = Supervisor(engine)
        gate.confirm("delete_record")
        assert gate.check("delete_record", params={"record_id": "1"}).allowed
        assert not gate.check("send_email", params={"to": "a@b.com"}).allowed
        gate.confirm("send_email")
        assert gate.check("send_email", params={"to": "a@b.com"}).allowed

    def test_confirmation_consumed_on_success(self):
        engine = generate_rules([DELETE_TOOL])
        gate = Supervisor(engine)
        gate.confirm("delete_record")
        assert gate.check("delete_record", params={"record_id": "1"}).allowed
        gate.record_success("delete_record")
        # stale approval is consumed → the next destructive call re-confirms
        assert not gate.check("delete_record", params={"record_id": "1"}).allowed

    def test_state_tracker_history(self):
        s = StateTracker()
        s.set("A", True).on_tool_success("search", {"HasResults": True})
        snap = s.snapshot()
        assert snap["A"] and snap["Done_search"] and snap["HasResults"]
        assert len(s.history()) == 4
