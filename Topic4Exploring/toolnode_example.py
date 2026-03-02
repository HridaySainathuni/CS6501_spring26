"""Topic 4: ToolNode example. Run: python toolnode_example.py"""
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
    """Get the current weather for a city. Input: city name."""
    # Simulated
    return f"Weather in {city}: Sunny, 72°F"

@tool
def search(query: str) -> str:
    """Search for information. Input: search query string."""
    return f"Search results for '{query}': [simulated result]"

TOOLS = [get_weather, search]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY. Exiting.")
        return
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(llm, TOOLS)
    # Draw graph (ToolNode is inside create_react_agent)
    try:
        from langchain_core.runnables import RunnableLambda
        graph = agent.get_graph()
        if hasattr(graph, "draw_mermaid_png"):
            graph.draw_mermaid_png(output_file_path="toolnode_graph.png")
            print("Saved toolnode_graph.png")
    except Exception as e:
        print("Graph export:", e)
    # Run
    inp = input("Enter query (or 'verbose' / 'exit'): ").strip()
    if inp.lower() == "exit":
        print("Goodbye.")
        return
    verbose = inp.lower() == "verbose"
    if verbose:
        inp = input("Now enter your query: ").strip()
    for chunk in agent.stream({"messages": [("user", inp)]}, stream_mode="values"):
        if "messages" in chunk and chunk["messages"]:
            last = chunk["messages"][-1]
            if hasattr(last, "content") and last.content:
                print("Assistant:", last.content)
            if verbose and hasattr(last, "tool_calls") and last.tool_calls:
                print("[TRACE] Tool calls:", last.tool_calls)

if __name__ == "__main__":
    main()
