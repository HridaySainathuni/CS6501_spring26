"""Task 4: Conditional model routing."""
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
    model_used: str  # Track which model was used

def create_llm(model_id: str, model_name: str):
    """Create LLM for a specific model."""
    device = get_device()

    print(f"Loading {model_name}: {model_id}")
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
    print(f"{model_name} loaded successfully!")
    return llm

def create_graph(llama_llm, qwen_llm):
    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)
        print("(Type 'Hey Qwen' at the start to use Qwen, otherwise uses Llama)")

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

    def call_llama(state: AgentState) -> dict:
        """Node that calls Llama model."""
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"

        print("\n[Llama] Processing your input...")
        response = llama_llm.invoke(prompt)

        return {
            "llm_response": response,
            "model_used": "Llama"
        }

    def call_qwen(state: AgentState) -> dict:
        """Node that calls Qwen model."""
        user_input = state["user_input"]
        # Remove "Hey Qwen" prefix from the input
        if user_input.lower().startswith("hey qwen"):
            user_input = user_input[9:].strip()
        
        prompt = f"User: {user_input}\nAssistant:"

        print("\n[Qwen] Processing your input...")
        response = qwen_llm.invoke(prompt)

        return {
            "llm_response": response,
            "model_used": "Qwen"
        }

    def print_response(state: AgentState) -> dict:
        """Node that prints the model response."""
        print("\n" + "=" * 50)
        print(f"{state.get('model_used', 'LLM')} Response:")
        print("-" * 50)
        print(state["llm_response"])

        return {}

    def route_after_input(state: AgentState) -> str:
        """Route to END if user wants to quit."""
        if state.get("should_exit", False):
            return END
        return "route_to_model"

    def route_to_model(state: AgentState) -> str:
        """
        Route to appropriate model based on input.
        If input starts with "Hey Qwen", route to Qwen, otherwise to Llama.
        """
        user_input = state.get("user_input", "")
        if user_input.lower().startswith("hey qwen"):
            return "call_qwen"
        else:
            return "call_llama"

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "route_to_model": "route_to_model",
            END: END
        }
    )

    # Route to appropriate model
    graph_builder.add_conditional_edges(
        "route_to_model",
        route_to_model,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen"
        }
    )

    # Both models feed into print_response
    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    
    # After printing, loop back
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
    print("LangGraph Simple Agent - Task 4: Conditional Model Routing")
    print("=" * 50)
    print()

    # Create both LLMs
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2-1B")
    print()
    qwen_llm = create_llm("Qwen/Qwen2.5-0.5B-Instruct", "Qwen 2.5-0.5B")

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph, "task4_graph.png")

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llm_response": "",
        "model_used": ""
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
