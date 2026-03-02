# Topic 2: Frameworks - LangGraph Implementation

This directory contains implementations of LangGraph agents with progressive enhancements.

## Table of Contents

1. [Base Implementation](#base-implementation)
2. [Task 1: Verbose/Quiet Tracing](#task-1)
3. [Task 2: Empty Input Handling](#task-2)
4. [Task 3: Parallel Models](#task-3)
5. [Task 4: Conditional Model Routing](#task-4)
6. [Task 5: Chat History with Message API](#task-5)
7. [Task 6: Chat History with Model Switching](#task-6)
8. [Task 7: Checkpointing and Crash Recovery](#task-7)

## Base Implementation

**File**: `langgraph_simple_agent.py`

Simple LangGraph agent that:
- Uses Llama 3.2-1B-Instruct from Hugging Face
- Reads input from stdin
- Prints LLM responses to stdout
- Generates graph visualization (Mermaid PNG)

**Usage**:
```bash
python langgraph_simple_agent.py
```

## Task 1: Verbose/Quiet Tracing

**File**: `task1_verbose_quiet_tracing.py`

Adds tracing functionality:
- Type "verbose" to enable detailed tracing for each node
- Type "quiet" to disable tracing
- Each node prints trace messages when verbose mode is enabled

**Features**:
- Tracing control via user input
- Detailed node execution information
- State tracking for verbose mode

**Usage**:
```bash
python task1_verbose_quiet_tracing.py
# Type "verbose" to enable tracing
# Type "quiet" to disable tracing
```

## Task 2: Empty Input Handling

**File**: `task2_empty_input_handling.py`

Handles empty input using LangGraph conditional routing:
- 3-way conditional branch from `get_user_input` node
- Empty input loops back to `get_user_input` (never sent to LLM)
- Demonstrates LangGraph routing patterns

**Features**:
- Prevents empty input from reaching LLM
- Uses graph structure (not loops) for input validation
- Shows how small LLMs handle empty input

**Usage**:
```bash
python task2_empty_input_handling.py
# Try entering empty input to see the loop behavior
```

**Observations**:
- First empty input: Prompts again
- Second empty input: Same behavior
- Small LLMs like Llama 3.2-1B may produce inconsistent responses to empty input
- This reveals limitations of smaller models

## Task 3: Parallel Models

**File**: `task3_parallel_models.py`

Runs both Llama and Qwen models in parallel:
- User input sent to both models simultaneously
- Both responses printed together
- Demonstrates parallel node execution in LangGraph

**Models Used**:
- Llama 3.2-1B-Instruct
- Qwen 2.5-0.5B-Instruct

**Features**:
- Parallel model execution
- Combined output display
- Graph structure for parallel processing

**Usage**:
```bash
python task3_parallel_models.py
```

## Task 4: Conditional Model Routing

**File**: `task4_conditional_model_routing.py`

Routes to appropriate model based on input:
- Input starting with "Hey Qwen" → routes to Qwen
- All other input → routes to Llama
- Only one model runs per turn

**Features**:
- Conditional routing based on input prefix
- Single model execution per turn
- Model selection via natural language

**Usage**:
```bash
python task4_conditional_model_routing.py
# Regular input uses Llama
# "Hey Qwen, what is AI?" uses Qwen
```

## Task 5: Chat History with Message API

**File**: `task5_chat_history.py`

Implements chat history using LangChain Message API:
- Uses `HumanMessage`, `AIMessage`, `SystemMessage`
- Maintains conversation context
- Qwen disabled (Llama only)

**Features**:
- Message API integration
- Conversation context preservation
- System prompts support

**Usage**:
```bash
python task5_chat_history.py
```

## Task 6: Chat History with Model Switching

**File**: `task6_chat_history_with_model_switching.py`

Integrates chat history with model switching:
- Handles three entities: Human, Llama, Qwen
- Uses Message API with name prefixes
- Maintains full conversation history across model switches

**Features**:
- Multi-entity conversation tracking
- Model-specific system prompts
- Full history visible to both models

**Usage**:
```bash
python task6_chat_history_with_model_switching.py
# Have a conversation, switch between models
# Example:
# > What is machine learning?
# > Hey Qwen, what do you think?
# > I agree with Qwen
```

**Example Conversation**:
```
Human: What is the best ice cream flavor?
Llama: There is no one best flavor, but the most popular is vanilla.
Human: Hey Qwen, what do you think?
Qwen: No way, chocolate is the best!
Human: I agree.
```

## Task 7: Checkpointing and Crash Recovery

**File**: `task7_checkpointing_crash_recovery.py`

Adds checkpointing for crash recovery:
- Uses LangGraph MemorySaver for state persistence
- Can kill process and resume from saved state
- Preserves full conversation history

**Features**:
- Automatic state checkpointing
- Crash recovery on restart
- Thread-based state management

**Usage**:
```bash
python task7_checkpointing_crash_recovery.py
# Have a conversation
# Kill the process (Ctrl+C)
# Restart - conversation will resume
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- transformers
- torch
- langchain
- langchain-core
- langchain-huggingface
- langgraph
- grandalf (for graph visualization)

## Graph Visualizations

Each task generates a graph visualization:
- `task1_graph.png` - Task 1 graph structure
- `task2_graph.png` - Task 2 with 3-way conditional
- `task3_graph.png` - Task 3 parallel execution
- `task4_graph.png` - Task 4 conditional routing
- `task5_graph.png` - Task 5 with Message API
- `task6_graph.png` - Task 6 multi-model with history
- `task7_graph.png` - Task 7 with checkpointing

## Terminal Output Files (Logs)

Terminal session outputs are saved as:
- `log_langgraph_simple_agent.txt` – Base agent run
- `log_task1_verbose_quiet_tracing.txt` – Task 1 (verbose/quiet)
- `log_task2_empty_input_handling.txt` – Task 2 empty input behavior
- `task2_empty_input_observations.txt` – Notes on empty input behavior
- `log_task3_parallel_models.txt` – Task 3 parallel Llama + Qwen
- `log_task4_conditional_model_routing.txt` – Task 4 conditional routing
- `log_task5_chat_history.txt` – Task 5 chat history
- `log_task6_chat_history_with_model_switching.txt` – Task 6 model switching
- `task6_conversation_example.txt` – Example multi-model conversation
- `log_task7_checkpointing_crash_recovery.txt` – Task 7 checkpointing

Generate fresh logs from repo root: `py -3 run_all_logs.py --topic 2`

## Notes

### Empty Input Behavior
When testing empty input (Task 2):
- First empty input: System prompts again
- Subsequent empty inputs: Same behavior
- Small LLMs (like Llama 3.2-1B) may produce inconsistent or nonsensical responses when given empty input
- This reveals that smaller models lack the sophistication to handle edge cases gracefully

### Model Selection
- **Llama 3.2-1B-Instruct**: Default model, instruction-tuned
- **Qwen 2.5-0.5B-Instruct**: Alternative model, activated with "Hey Qwen" prefix

### Checkpointing
- State is saved after each node execution
- Uses in-memory checkpointing (MemorySaver)
- Thread ID: "main_conversation" (fixed for session)
- Can be extended to use file-based checkpointing for persistence across sessions

## Development Notes

Each task builds upon the previous one:
1. Base → Simple agent
2. Task 1 → Adds tracing
3. Task 2 → Handles edge cases
4. Task 3 → Parallel execution
5. Task 4 → Conditional routing
6. Task 5 → Chat history
7. Task 6 → History + model switching
8. Task 7 → Crash recovery

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Message API](https://python.langchain.com/docs/modules/messages/)
- [LangGraph Crash Recovery](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
