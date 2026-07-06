"""
Agent retry loop — how the gate feeds violations back to the LLM.

The pattern: LLM proposes → gate checks → if blocked, violation text
goes back into the LLM's context and it proposes again. No LLM is
called here (the "agent" is scripted), but the loop is exactly what
you'd wrap around any real model.

Run: python examples/agent_loop.py
"""

from witt import Supervisor, generate_rules

TOOLS = [
    {"name": "get_location", "description": "Get the user's location",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_weather", "description": "Get weather for a location",
     "parameters": {"type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]}},
]

engine = generate_rules(TOOLS, dependencies={"get_weather": ["get_location"]})
gate = Supervisor(engine)


class ScriptedAgent:
    """Stands in for an LLM. First tries the wrong thing (a very common
    real failure: jumping straight to the goal tool), then corrects
    when given the violation feedback."""

    def __init__(self):
        self.feedback = None
        self.location = None

    def next_action(self, goal):
        if self.feedback and "get_location" in self.feedback:
            return ("get_location", {})
        if self.location:
            return ("get_weather", {"location": self.location})
        return ("get_weather", {})  # the naive first attempt


agent = ScriptedAgent()
goal = "What's the weather where I am?"
print(f"Goal: {goal}\n")

for step in range(1, 6):
    tool, params = agent.next_action(goal)
    verdict = gate.check(tool, params=params)
    print(f"Step {step}: agent proposes {tool}({params})")

    if verdict:
        print(f"         ✓ allowed — executing")
        # simulate execution
        if tool == "get_location":
            agent.location = "Tokyo"
            gate.record_success(tool)
            agent.feedback = None
        elif tool == "get_weather":
            gate.record_success(tool)
            print(f"\nResult: Sunny, 22°C in {params['location']}")
            break
    else:
        print(f"         ✗ blocked — {verdict.feedback}")
        agent.feedback = verdict.feedback  # goes back into LLM context

print(f"\nStats: {gate.stats()}")
