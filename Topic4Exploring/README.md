# Topic 4: Exploring Tools

## Table of Contents

1. [ToolNode example](#toolnode_example)
2. [ReAct agent example](#react_agent_example)
3. [Portfolio questions and answers](#portfolio-questions)
4. [2-Hour Agent Project](#two-hour-project)

## Files

| File | Purpose |
|------|--------|
| toolnode_example.py | ToolNode-style agent (create_react_agent with tools) |
| react_agent_example.py | ReAct agent, handles verbose/exit |
| two_hour_agent_project.py | Multi-tool agent: calculator, count_letter, current_time |
| educational_analyzer_agent.py | 2-hour project: YouTube educational video analyzer (transcript tools) |
| README.md | This file |

## Terminal output (logs)

- `log_toolnode_example.txt` – ToolNode example run
- `log_react_agent_example.txt` – ReAct agent example run
- `log_two_hour_agent_project.txt` – 2-hour agent (calculator, time) run
- `log_educational_analyzer_agent.txt` – Educational analyzer run (video analysis)

From repo root: `py -3 run_all_logs.py --topic 4`

## Portfolio questions and answers

### What features of Python does ToolNode use to dispatch tools in parallel? What kinds of tools would most benefit from parallel dispatch?

**ToolNode** (inside LangGraph’s tool execution step) can run multiple tool calls in one step by dispatching each call to a worker (e.g. `concurrent.futures.ThreadPoolExecutor`) and waiting for all to finish. So it uses **threads or async** to run independent tool invocations in parallel. Tools that **benefit most** are I/O-bound and independent: e.g. multiple API calls (weather, search, different URLs), or several slow external services. CPU-bound or single-call tools gain less.

### How do the two programs handle special inputs such as "verbose" and "exit"?

- **toolnode_example.py:** Reads one line; if it is `"exit"`, the program exits. If it is `"verbose"`, it sets a flag and prompts again for the real query, then prints trace info (e.g. tool_calls) for the response.
- **react_agent_example.py:** In a loop, if the input is `"exit"`, it breaks and exits. If the input is `"verbose"`, it prints a message and asks for the next query, then continues with the agent.

So both handle **exit** by stopping, and **verbose** by asking for another input and optionally printing extra trace output.

### Compare the graph diagrams of the two programs. How do they differ if at all?

Both use `create_react_agent(llm, tools)`, so the **underlying graph is the same**: a ReAct loop with a node that calls the LLM and a node that runs tools (ToolNode). The Mermaid diagrams (e.g. `toolnode_graph.png` vs `react_agent_graph.png`) will look identical: START → agent node → conditional edge (tool_calls vs end) → tools → back to agent → END. The only difference is the surrounding script (e.g. loop vs single run, verbose/exit handling).

### What is an example of a case where the structure imposed by the LangChain ReAct agent is too restrictive and you'd want the ToolNode approach (or a custom graph)?

ReAct forces a strict **reason → act → reason** cycle: one LLM call, then one batch of tool calls, then one LLM call, etc. That is restrictive when you want:

- **Conditional tool chains:** e.g. run tool A, then depending on the result run either tool B or C without another LLM call.
- **Parallel human-in-the-loop:** e.g. run several tools in parallel and then show results to the user before the next LLM turn.
- **Custom routing:** e.g. one node that decides “search” vs “calculator” and different subgraphs for each.

In those cases you’d build a **custom graph** with your own nodes and edges (and optionally use ToolNode only for the “execute these tool calls” node) instead of using the single prebuilt ReAct agent.

## 2-Hour Agent Projects

**two_hour_agent_project.py** – Minimal multi-tool agent with calculator, count_letter, and current_time. Uses `create_react_agent`; type **verbose** or **exit** as in the examples.

**educational_analyzer_agent.py** – Separate 2-hour project: YouTube Educational Video Analyzer. Uses ReAct agent with tools to fetch YouTube transcripts (plain or timestamped). Features: summary with bullet points and key quotes; extract chapter timestamps; answer questions about the video; compare multiple videos. Requires `OPENAI_API_KEY` and `pip install youtube-transcript-api`.

```bash
python two_hour_agent_project.py
python educational_analyzer_agent.py   # Enter YouTube URL or ID, then use menu (1–4)
```

## Running the examples

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python toolnode_example.py
python react_agent_example.py
python two_hour_agent_project.py
```
