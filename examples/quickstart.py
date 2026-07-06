"""
truthgate quickstart — validate agent tool calls in ~10 lines.

Run: python examples/quickstart.py
"""

from truthgate import Supervisor, generate_rules

# 1. Your tools, in the same format you already give your agent
#    (OpenAI / Anthropic / BFCL function-calling schema)
TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "summarize",
        "description": "Summarize retrieved content",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    },
]

# 2. Generate rules automatically from the specs
#    - required params → prerequisite rules
#    - "send" in name → destructive → confirmation rule (auto-detected)
#    - explicit dependency: summarize requires a completed search
engine = generate_rules(TOOLS, dependencies={"summarize": ["search_web"]})
gate = Supervisor(engine)

print("Rules generated:")
for r in engine.rules:
    print(f"  • {r['name']}:  {engine.expr_to_string(r['expr'])}")
print()

# 3. Simulate an agent making decisions — some good, some bad
proposals = [
    ("summarize", {}),                                   # bad: search not done yet
    ("search_web", {}),                                  # bad: missing required 'query'
    ("search_web", {"query": "quarterly results"}),      # good
    ("send_email", {"to": "a@b.com", "body": "hi"}),     # bad: not confirmed
]

for tool, params in proposals:
    verdict = gate.check(tool, params=params)
    status = "ALLOW" if verdict else "BLOCK"
    print(f"[{status}] {tool}({params})")
    if not verdict:
        print(f"        {verdict.feedback}")
    else:
        gate.record_success(tool)

# 4. User confirms → destructive action now allowed
gate.confirm()
verdict = gate.check("send_email", params={"to": "a@b.com", "body": "hi"})
print(f"[{'ALLOW' if verdict else 'BLOCK'}] send_email (after confirmation)")

print()
print("Stats:", gate.stats())
