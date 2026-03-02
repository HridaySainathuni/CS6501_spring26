"""Task 1: Add verbose/quiet tracing."""
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
    verbose: bool  # New field for tracing control

def trace(message: str, verbose: bool):
    """Print tracing information if verbose mode is enabled."""
    if verbose:
        print(f"[TRACE] {message}")

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
        verbose = state.get("verbose", False)
        trace("Entering get_user_input node", verbose)
        
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit, 'verbose' for tracing, 'quiet' to disable):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()
        
        trace(f"User input received: '{user_input}'", verbose)

        if user_input.lower() in ['quit', 'exit', 'q']:
            trace("User requested exit", verbose)
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True
            }

        # Check for verbose/quiet commands
        if user_input.lower() == "verbose":
            trace("Enabling verbose mode", True)  # Always trace this
            return {
                "user_input": "",
                "should_exit": False,
                "verbose": True
            }
        elif user_input.lower() == "quiet":
            trace("Disabling verbose mode", True)  # Always trace this
            return {
                "user_input": "",
                "should_exit": False,
                "verbose": False
            }

        trace("Proceeding to LLM call", verbose)
        return {
            "user_input": user_input,
            "should_exit": False
        }

    def call_llm(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        trace("Entering call_llm node", verbose)
        
        user_input = state["user_input"]
        trace(f"Preparing prompt with input: '{user_input}'", verbose)
        
        prompt = f"User: {user_input}\nAssistant:"
        trace(f"Formatted prompt: '{prompt}'", verbose)

        print("\nProcessing your input...")
        trace("Invoking LLM", verbose)

        response = llm.invoke(prompt)
        
        trace(f"LLM response received: '{response}'", verbose)
        return {"llm_response": response}

    def print_response(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        trace("Entering print_response node", verbose)
        
        trace(f"Printing response: '{state['llm_response']}'", verbose)
        
        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(state["llm_response"])

        trace("Exiting print_response node", verbose)
        return {}

    def route_after_input(state: AgentState) -> str:
        verbose = state.get("verbose", False)
        trace("Entering route_after_input function", verbose)
        
        if state.get("should_exit", False):
            trace("Routing to END", verbose)
            return END
        
        trace("Routing to call_llm", verbose)
        return "call_llm"

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
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
    print("LangGraph Simple Agent - Task 1: Verbose/Quiet Tracing")
    print("=" * 50)
    print()

    llm = create_llm()

    print("\nCreating LangGraph...")
    graph = create_graph(llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph, "task1_graph.png")

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llm_response": "",
        "verbose": False
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
