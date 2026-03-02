"""Topic 3 Task 5: Single long conversation with LangGraph nodes/edges and checkpointing."""
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
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import json

class State(TypedDict):
    messages: Annotated[list, operator.add]

# Simple tools for the conversation agent
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression: +, -, *, /, **, sqrt, sin, cos, pi, e."""
    import math
    ns = {"sin": math.sin, "cos": math.cos, "sqrt": math.sqrt, "pi": math.pi, "e": math.e}
    try:
        return json.dumps({"result": eval(expression, {"__builtins__": {}}, ns)})
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def count_letter(text: str, letter: str) -> str:
    """Count occurrences of a letter in text."""
    return json.dumps({"count": (text or "").lower().count((letter or " ").lower()[:1])})

TOOLS = [calculator, count_letter]
TOOL_MAP = {t.name: t for t in TOOLS}

def build_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)
    memory = MemorySaver()

    def chat_node(state):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def tool_node(state):
        messages = state["messages"]
        last = messages[-1]
        if not getattr(last, "tool_calls", None):
            return state
        from langchain_core.messages import ToolMessage
        to_append = []
        for tc in last.tool_calls:
            name = tc.get("name", getattr(tc, "name", ""))
            args = tc.get("args", getattr(tc, "args", {}))
            if name in TOOL_MAP:
                result = TOOL_MAP[name].invoke(args)
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})
            tid = tc.get("id", getattr(tc, "id", ""))
            to_append.append(ToolMessage(content=result, tool_call_id=tid))
        return {"messages": to_append}

    def should_continue(state):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(State)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", should_continue)
    graph.add_edge("tools", "chat")
    comp = graph.compile(checkpointer=memory)
    return comp

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY.")
        return
    comp = build_graph()
    config = {"configurable": {"thread_id": "main"}}
    # One turn with tool use
    state = comp.invoke(
        {"messages": [HumanMessage(content="What is sin(0) + sqrt(4)? Use the calculator.")]},
        config=config,
    )
    print("Final messages:", len(state["messages"]))
    for m in state["messages"]:
        if hasattr(m, "content") and m.content:
            print("Content:", m.content[:200] if len(str(m.content)) > 200 else m.content)
    print("Conversation with checkpointing ready. Restart with same thread_id to resume.")

if __name__ == "__main__":
    main()
