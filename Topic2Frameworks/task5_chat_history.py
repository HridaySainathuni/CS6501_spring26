"""Task 5: Chat history with Message API."""
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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated
from typing_extensions import TypedDict as ExtTypedDict
import operator

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

class AgentState(ExtTypedDict):
    messages: Annotated[list, operator.add]  # Chat history using Message API

def create_llm():
    """Create Llama LLM only."""
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
            return {"messages": []}  # Empty messages to signal exit

        # Add user message to history
        return {"messages": [HumanMessage(content=user_input)]}

    def call_llm(state: AgentState) -> dict:
        """Node that calls LLM with chat history."""
        messages = state["messages"]
        
        # Convert LangChain messages to format LLM expects
        # For instruction-tuned models, format as conversation
        conversation_text = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                conversation_text += f"System: {msg.content}\n"
            elif isinstance(msg, HumanMessage):
                conversation_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_text += f"Assistant: {msg.content}\n"
        
        conversation_text += "Assistant:"
        
        print("\nProcessing your input with chat history...")
        print(f"[DEBUG] Conversation length: {len(messages)} messages")
        
        response = llm.invoke(conversation_text)
        
        # Add AI response to history
        return {"messages": [AIMessage(content=response)]}

    def print_response(state: AgentState) -> dict:
        """Node that prints the latest AI response."""
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            print("\n" + "=" * 50)
            print("LLM Response:")
            print("-" * 50)
            print(messages[-1].content)
        return {}

    def route_after_input(state: AgentState) -> str:
        """Route based on whether user wants to exit."""
        messages = state.get("messages", [])
        if not messages:  # Empty messages means exit
            return END
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
    print("LangGraph Simple Agent - Task 5: Chat History with Message API")
    print("=" * 50)
    print()

    llm = create_llm()

    print("\nCreating LangGraph...")
    graph = create_graph(llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph, "task5_graph.png")

    # Initialize with system message
    initial_state: AgentState = {
        "messages": [SystemMessage(content="You are a helpful assistant.")]
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
