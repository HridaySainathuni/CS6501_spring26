"""Topic 4: 2-Hour Agent Project - multi-tool agent."""
import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
try:
    from load_secrets import load_secrets
    load_secrets()
except Exception:
    pass
import json
import math
from datetime import datetime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, pi."""
    try:
        ns = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "pi": math.pi}
        return json.dumps({"result": eval(expression, {"__builtins__": {}}, ns)})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def count_letter(text: str, letter: str) -> str:
    """Count how many times a letter appears in text. Example: count_letter('hello','l') -> 2."""
    c = (text or "").lower().count((letter or " ").lower()[:1])
    return json.dumps({"count": c, "letter": letter, "text": text})

@tool
def current_time() -> str:
    """Get the current date and time in ISO format."""
    return json.dumps({"current_time": datetime.now().isoformat()})

TOOLS = [calculator, count_letter, current_time]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY.")
        return
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm, TOOLS)
    print("2-Hour Agent: calculator, count_letter, current_time. Type 'exit' to quit.")
    while True:
        inp = input("You: ").strip()
        if not inp:
            continue
        if inp.lower() == "exit":
            break
        if inp.lower() == "verbose":
            inp = input("Query: ").strip()
        for chunk in agent.stream({"messages": [("user", inp)]}, stream_mode="values"):
            if chunk.get("messages"):
                last = chunk["messages"][-1]
                if hasattr(last, "content") and last.content:
                    print("Agent:", last.content)

if __name__ == "__main__":
    main()
