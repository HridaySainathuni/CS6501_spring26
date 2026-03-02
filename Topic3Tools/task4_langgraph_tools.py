"""Topic 3 Task 4: LangGraph tool handling - calculator, letter count, custom tool."""
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*create_react_agent.*")
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
import re
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- Calculator (with geometric functions) ---
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use for arithmetic and geometry: +, -, *, /, **, sqrt, sin, cos, tan, log, exp, pi, e."""
    safe = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "pi": math.pi, "e": math.e,
        "ceil": math.ceil, "floor": math.floor,
    }
    safe["abs"] = abs
    expr = (expression or "").strip().replace("^", "**")
    if not re.match(r"^[\d\s+\-*/().%\w]+$", expr):
        return json.dumps({"error": "Invalid expression"})
    try:
        result = eval(expr, {"__builtins__": {}}, safe)
        return json.dumps({"result": float(result) if isinstance(result, (int, float)) else result})
    except Exception as e:
        return json.dumps({"error": str(e)})

# --- Letter count tool ---
@tool
def count_letter(text: str, letter: str) -> str:
    """Count how many times a given letter appears in a piece of text. Example: count_letter('Mississippi', 's') -> 4."""
    if not letter or len(letter) != 1:
        return json.dumps({"error": "letter must be a single character"})
    c = (text or "").lower().count(letter.lower())
    return json.dumps({"text": text, "letter": letter, "count": c})

# --- Third tool: word count ---
@tool
def word_count(text: str) -> str:
    """Count the number of words in a piece of text. Splits on whitespace."""
    words = (text or "").split()
    return json.dumps({"text": text, "word_count": len(words)})

TOOLS = [calculator, count_letter, word_count]
TOOL_MAP = {t.name: t for t in TOOLS}

def run_agent(query: str, max_turns: int = 5):
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY.")
        return None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm, TOOLS)
    config = {"configurable": {}}
    state = agent.invoke({"messages": [("user", query)]}, config=config)
    for _ in range(max_turns - 1):
        msgs = state["messages"]
        if not msgs:
            break
        last = msgs[-1]
        if hasattr(last, "content") and last.content and not getattr(last, "tool_calls", None):
            return last.content
        state = agent.invoke(state, config=config)
    return (state["messages"][-1].content if state.get("messages") else None) or "[No response]"

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="How many s are in Mississippi riverboats? Use the tool.")
    args = p.parse_args()
    print("Query:", args.query)
    out = run_agent(args.query)
    print("Response:", out)

if __name__ == "__main__":
    main()
