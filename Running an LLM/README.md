# Running an LLM - Portfolio Submission

This directory contains the code, graphs, and analysis for the "Running an LLM" assignment.

## Contents

### Code Files
- `llama_mmlu_eval.py` - Multi-model MMLU evaluation script
- `chat_agent.py` - Chat agent with context management
- `verify_setup.py` - Environment setup verification

### Documentation
- `ANALYSIS.md` - Analysis and discussion of results (questions from the tasks)

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Verify setup: `python verify_setup.py`
3. Run evaluation: `python llama_mmlu_eval.py --use-gpu`
4. Run chat agent: `python chat_agent.py`

## Running the evaluation

**Quick verification (2 subjects):**
```bash
python llama_mmlu_eval.py --max-subjects 2
```

**Full run (all 10 MMLU subjects, 3 models):**
```bash
# GPU recommended
python llama_mmlu_eval.py --use-gpu

# With verbose (print each question, model answer, right/wrong)
python llama_mmlu_eval.py --use-gpu --verbose
```

Graphs (accuracy, timing) are generated from the evaluation results. Run from this directory; output JSON and plots appear in the current folder.

## Features Implemented

✅ 3 tiny/small models evaluated  
✅ 10 MMLU subjects  
✅ Timing information (real, CPU, GPU time)  
✅ Verbose mode for question-level analysis  
✅ Graph generation  
✅ Chat agent with context management  
✅ History toggle option  
✅ Restartability with pickle  

Chat agent supports context management, history toggle, and restartability (pickle).

