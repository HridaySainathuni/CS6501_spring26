"""Task 2: Handle empty input with 3-way conditional."""
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    llm_response: str

def create_llm():
    device = get_device()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    print("Model loaded successfully!")
    return llm

def create_graph(llm):
    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True
            }

        return {
            "user_input": user_input,
            "should_exit": False
        }

    def call_llm(state: AgentState) -> dict:
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        print("\nProcessing your input...")

        response = llm.invoke(prompt)

        return {"llm_response": response}

    def print_response(state: AgentState) -> dict:
        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(state["llm_response"])

        return {}

    def route_after_input(state: AgentState) -> str:
        """
        3-way conditional routing:
        1. If user wants to exit -> END
        2. If input is empty -> loop back to get_user_input
        3. If input has content -> proceed to call_llm
        """
        if state.get("should_exit", False):
            return END
        
        # Check if input is empty (after stripping whitespace)
        user_input = state.get("user_input", "").strip()
        if not user_input:
            print("[INFO] Empty input detected, prompting again...")
            return "get_user_input"  # Loop back to itself
        
        return "call_llm"

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    # 3-way conditional: END, get_user_input (loop), or call_llm
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",  # Loop back for empty input
            "call_llm": "call_llm",
            END: END
        }
    )

    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    graph = graph_builder.compile()

    return graph

def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

def main():
    print("=" * 50)
    print("LangGraph Simple Agent - Task 2: Empty Input Handling")
    print("=" * 50)
    print()
    print("Note: Try entering empty input to see the 3-way conditional in action.")
    print()

    llm = create_llm()

    print("\nCreating LangGraph...")
    graph = create_graph(llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph, "task2_graph.png")

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llm_response": ""
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
