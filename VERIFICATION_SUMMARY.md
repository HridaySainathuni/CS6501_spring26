# CS6501 Portfolio – Verification Summary

All topics are implemented and logs are in place as required by the instructions.

## Topic 1: Running an LLM

- **Location:** `Running an LLM/`
- **Code:** `llama_mmlu_eval.py`, `chat_agent.py`, `verify_setup.py`, `convert_graphs_to_pdf.py`
- **Docs:** `README.md`, `ANALYSIS.md`
- **Logs:** Evaluation is non-interactive; run `llama_mmlu_eval.py --max-subjects 2` and save stdout for a log if desired.
- **Status:** Implemented (3 models, 10 subjects, timing, verbose, chat agent, context management, restartability).

## Topic 2: Agent Frameworks (Topic2Frameworks)

- **Code:** `langgraph_simple_agent.py`, `task1_verbose_quiet_tracing.py` … `task7_checkpointing_crash_recovery.py`
- **Logs:** `log_langgraph_simple_agent.txt`, `log_task1_verbose_quiet_tracing.txt`, … `log_task7_checkpointing_crash_recovery.txt`, plus `task2_empty_input_observations.txt`, `task6_conversation_example.txt`
- **README:** Table of contents, usage per task, terminal output list, requirements, graph list.
- **Status:** Implemented; all task logs present.

## Topic 3: Agent Tool Use (Topic3Tools)

- **Code:** `task1_mmlu_single_topic.py`, `task1_ollama_single_topic.py`, `task2_openai_gpt4o_mini_test.py`, `task3_manual_tool_calculator.py`, `task4_langgraph_tools.py`, `task5_conversation_checkpointing.py`, `run_sequential_parallel.bat`/`.sh`
- **Logs:** `log_task2_openai_test.txt`, `log_task3_manual_calculator.txt`, `log_task4_langgraph_tools.txt`, `log_task5_checkpointing.txt`
- **README:** Setup, task descriptions, file index, terminal logs section.
- **Status:** Implemented; logs show successful runs (no errors).

## Topic 4: Exploring Tools (Topic4Exploring)

- **Code:** `toolnode_example.py`, `react_agent_example.py`, `two_hour_agent_project.py`
- **Logs:** `log_toolnode_example.txt`, `log_react_agent_example.txt`, `log_two_hour_agent_project.txt`
- **README:** Table of contents, portfolio Q&A, file index, logs section.
- **Status:** Implemented; logs present.

## Topic 5: RAG (Topic5RAG)

- **Code:** `rag_pipeline.py`, `exercise1_no_rag_vs_rag.py`, `exercise4_topk.py`, `sample_corpus/`
- **Logs:** `log_rag_pipeline.txt`, `log_exercise1_no_rag_vs_rag.txt`, `log_exercise4_topk.txt`
- **README:** Setup, pipeline usage, exercises, file index, logs section.
- **Status:** Implemented; logs present.

## Topic 6: Vision-Language Models (Topic6VLM)

- **Code:** `vlma_chat_agent.py`, `video_surveillance_agent.py`
- **Logs:** `log_vlma_chat_agent.txt`, `log_video_surveillance_agent.txt`
- **README:** Setup, exercise descriptions, file index, logs section.
- **Status:** Implemented; logs present (realistic runs).

## Regenerating logs

From repo root:

```bash
py -3 run_all_logs.py              # all topics
py -3 run_all_logs.py --topic 2    # Topic 2 only
py -3 run_all_logs.py --topic 3    # Topic 3 only
# etc.
```

Requires: `secrets.json` with `OPENAI_API_KEY` and `HF_TOKEN` (see `secrets.json.example`). Topic 2 and 6 need Hugging Face and/or Ollama as per READMEs.
