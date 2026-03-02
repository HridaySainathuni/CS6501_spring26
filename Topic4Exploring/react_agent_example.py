"""Topic 4: ReAct agent example. Run: python react_agent_example.py"""
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
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for '{query}': [simulated]"

TOOLS = [get_weather, search]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY. Exiting.")
        return
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm, TOOLS)
    try:
        g = agent.get_graph()
        if hasattr(g, "draw_mermaid_png"):
            g.draw_mermaid_png(output_file_path="react_agent_graph.png")
            print("Saved react_agent_graph.png")
    except Exception as e:
        print("Graph export:", e)
    while True:
        inp = input("You (verbose/exit to quit): ").strip()
        if inp.lower() == "exit":
            print("Goodbye.")
            break
        if inp.lower() == "verbose":
            print("[TRACE] Verbose mode - next response will show tool calls")
            inp = input("Query: ").strip()
        for chunk in agent.stream({"messages": [("user", inp)]}, stream_mode="values"):
            if "messages" in chunk and chunk["messages"]:
                print("Assistant:", chunk["messages"][-1].content or "(tool call)")

if __name__ == "__main__":
    main()
