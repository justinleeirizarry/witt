"""
witt.state — Execution state tracking for agent pipelines.

Tracks facts about the world as tools execute, so the supervisor
always validates against current reality.
"""

from __future__ import annotations

from .autogen import done_prop, result_prop


class StateTracker:
    """Tracks agent execution state as propositions.

    Facts come from the execution environment — tool results, session
    status, user confirmations — never from the LLM. This is why the
    validator has no feature-extraction bottleneck.

    Usage:
        state = StateTracker()
        state.set("Authenticated", True)
        state.on_tool_success("search_web")   # sets Done_search_web + Result_search_web
        snapshot = state.snapshot()
    """

    def __init__(self, initial: dict | None = None):
        self._state: dict[str, bool] = dict(initial or {})
        self._history: list[tuple[str, bool]] = []

    def set(self, prop: str, value: bool = True):
        self._state[prop] = bool(value)
        self._history.append((prop, bool(value)))
        return self

    def unset(self, prop: str):
        return self.set(prop, False)

    def get(self, prop: str) -> bool:
        return self._state.get(prop, False)

    def on_tool_success(self, tool_name: str, extra: dict | None = None):
        """Record a successful tool execution.

        Sets Done_{tool} and Result_{tool} (kept in lockstep with the
        completion coupling in generate_rules), plus any extra props
        (e.g. a location tool succeeding should set HasUserLocation).
        """
        self.set(done_prop(tool_name), True)
        self.set(result_prop(tool_name), True)
        for k, v in (extra or {}).items():
            self.set(k, v)
        return self

    def on_tool_failure(self, tool_name: str):
        # NOTE: generate_rules emits no rule referencing Failed_ props, so
        # this state is inert unless you author your own failure-aware rules.
        self.set(f"Failed_{tool_name}", True)
        return self

    def snapshot(self) -> dict:
        """Current state as a props dict for the engine."""
        return dict(self._state)

    def history(self) -> list:
        return list(self._history)

    def reset(self):
        self._state.clear()
        self._history.clear()
        return self
