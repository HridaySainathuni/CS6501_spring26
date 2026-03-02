# Topic 5: RAG - Retrieval Augmented Generation

## Table of Contents

1. [Setup (Exercise 0)](#setup)
2. [Exercise 1: No RAG vs RAG](#exercise-1)
3. [Exercise 4: Top-K retrieval](#exercise-4)
4. [Pipeline usage](#pipeline)
5. [Full exercises (Corpora.zip)](#full-exercises)

## Setup (Exercise 0)

- Install: `pip install -r requirements.txt`
- For full corpus: Download **Corpora.zip** (from course materials), unzip, and use the `NewModelT` and Congressional Record `txt/` folders as `--corpus`.
- This repo includes a **sample_corpus** with one excerpt so the pipeline runs without external downloads.

## Pipeline

**rag_pipeline.py** – Single script for chunking, embedding, retrieval, and generation.

```bash
# Default: sample corpus, top_k=5
python rag_pipeline.py --query "What is the correct spark plug gap for a Model T?"

# Custom corpus and top-k
python rag_pipeline.py --corpus /path/to/txt --top-k 10 --chunk-size 512 --overlap 64

# Retrieve only (no LLM)
python rag_pipeline.py --no-llm --query "carburetor adjustment"
```

## Exercise 1: Open Model RAG vs No RAG

**exercise1_no_rag_vs_rag.py** – Asks the same question (e.g. spark plug gap) once without RAG (direct LLM) and once with RAG. Documents whether the model hallucinates without RAG and whether RAG grounds the answer.

```bash
export OPENAI_API_KEY=...
python exercise1_no_rag_vs_rag.py
```

## Exercise 4: Effect of Top-K

**exercise4_topk.py** – Runs the same query with k=1, 3, 5, 10 and prints the generated answer for each. Use to observe when adding more context helps or hurts.

```bash
python exercise4_topk.py
```

## Full exercises

For the complete assignment (Model T + Congressional Record, GPT-4o Mini comparison, unanswerable questions, query phrasing, chunk overlap/size, score analysis, prompt templates, cross-document synthesis):

1. Download **manual_rag_pipeline_universal.ipynb** and **Corpora.zip** from course resources.
2. Run the notebook on Colab or locally with the provided corpora.
3. Use this directory’s scripts as reference or adapt them to the notebook (e.g. same chunk/embed/retrieve logic).

## Terminal output (logs)

- `log_rag_pipeline.txt` – RAG pipeline run (sample query)
- `log_exercise1_no_rag_vs_rag.txt` – No RAG vs RAG comparison
- `log_exercise4_topk.txt` – Top-k retrieval comparison

From repo root: `py -3 run_all_logs.py --topic 5`

## File index

| File | Purpose |
|------|--------|
| rag_pipeline.py | Chunk, embed, retrieve, generate (configurable) |
| exercise1_no_rag_vs_rag.py | Compare no RAG vs RAG |
| exercise4_topk.py | Vary top-k and compare answers |
| sample_corpus/ | Small excerpt for testing |
| README.md | This file |

## Team

List your team members in this README when submitting.
