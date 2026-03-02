"""Task 6: Chat history with model switching."""
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

def format_messages_for_llm(messages, target_model="Llama"):
    """
    Format chat history for LLM, handling three entities:
    - Human messages: "Human: <content>"
    - Llama responses: "Llama: <content>" (as user role)
    - Qwen responses: "Qwen: <content>" (as user role)
    """
    conversation_text = ""
    
    # Add system prompt based on target model
    if target_model == "Llama":
        system_prompt = "You are Llama, one of two AI assistants. The other assistant is Qwen. When the human says 'Hey Qwen', they are addressing Qwen. Otherwise, they are addressing you. You can see the full conversation history including responses from both you and Qwen."
    else:  # Qwen
        system_prompt = "You are Qwen, one of two AI assistants. The other assistant is Llama. When the human says 'Hey Qwen', they are addressing you. Otherwise, they are addressing Llama. You can see the full conversation history including responses from both you and Llama."
    
    conversation_text += f"System: {system_prompt}\n"
    
    # Process all messages
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Skip the original system message, we use our own
            continue
        elif isinstance(msg, HumanMessage):
            # Check if it's a human message or a model response stored as human
            content = msg.content
            if content.startswith("Human:") or content.startswith("Llama:") or content.startswith("Qwen:"):
                conversation_text += f"{content}\n"
            else:
                conversation_text += f"Human: {content}\n"
        elif isinstance(msg, AIMessage):
            # AI messages are responses from models
            content = msg.content
            if target_model == "Llama" and content.startswith("Llama:"):
                # This is our own response, format as assistant
                conversation_text += f"Assistant: {content[7:]}\n"  # Remove "Llama: " prefix
            elif target_model == "Qwen" and content.startswith("Qwen:"):
                # This is our own response, format as assistant
                conversation_text += f"Assistant: {content[6:]}\n"  # Remove "Qwen: " prefix
            else:
                # Other model's response, format as user
                conversation_text += f"{content}\n"
    
    conversation_text += "Assistant:"
    return conversation_text

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
            return {"messages": []}  # Empty messages to signal exit

        # Add human message with prefix
        return {"messages": [HumanMessage(content=f"Human: {user_input}")]}

    def call_llama(state: AgentState) -> dict:
        """Node that calls Llama model with chat history."""
        messages = state["messages"]
        
        conversation_text = format_messages_for_llm(messages, target_model="Llama")
        
        print("\n[Llama] Processing with chat history...")
        print(f"[DEBUG] Conversation length: {len(messages)} messages")
        
        response = llama_llm.invoke(conversation_text)
        
        # Add Llama response with prefix
        return {"messages": [AIMessage(content=f"Llama: {response}")]}

    def call_qwen(state: AgentState) -> dict:
        """Node that calls Qwen model with chat history."""
        messages = state["messages"]
        
        # Remove "Hey Qwen" prefix from the latest human message
        if messages and isinstance(messages[-1], HumanMessage):
            content = messages[-1].content
            if content.lower().startswith("human: hey qwen"):
                messages[-1].content = content.replace("hey qwen", "", 1).strip()
                if messages[-1].content.startswith("Human:"):
                    messages[-1].content = "Human: " + messages[-1].content[6:].strip()
        
        conversation_text = format_messages_for_llm(messages, target_model="Qwen")
        
        print("\n[Qwen] Processing with chat history...")
        print(f"[DEBUG] Conversation length: {len(messages)} messages")
        
        response = qwen_llm.invoke(conversation_text)
        
        # Add Qwen response with prefix
        return {"messages": [AIMessage(content=f"Qwen: {response}")]}

    def print_response(state: AgentState) -> dict:
        """Node that prints the latest model response."""
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            print("\n" + "=" * 50)
            content = messages[-1].content
            if content.startswith("Llama:"):
                print("Llama Response:")
            elif content.startswith("Qwen:"):
                print("Qwen Response:")
            else:
                print("LLM Response:")
            print("-" * 50)
            # Print without the prefix
            if ":" in content:
                print(content.split(":", 1)[1].strip())
            else:
                print(content)
        return {}

    def route_after_input(state: AgentState) -> str:
        """Route based on whether user wants to exit."""
        messages = state.get("messages", [])
        if not messages:  # Empty messages means exit
            return END
        return "route_to_model"

    def route_to_model(state: AgentState) -> str:
        """Route to appropriate model based on input."""
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            user_input = messages[-1].content.lower()
            if user_input.startswith("human: hey qwen"):
                return "call_qwen"
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

    graph_builder.add_conditional_edges(
        "route_to_model",
        route_to_model,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen"
        }
    )

    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
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
    print("LangGraph Simple Agent - Task 6: Chat History with Model Switching")
    print("=" * 50)
    print()

    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2-1B")
    print()
    qwen_llm = create_llm("Qwen/Qwen2.5-0.5B-Instruct", "Qwen 2.5-0.5B")

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph, "task6_graph.png")

    initial_state: AgentState = {
        "messages": []
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
