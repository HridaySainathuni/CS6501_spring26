"""Topic 5 Exercise 4: Effect of top-k retrieval count.
Run: python exercise4_topk.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import load_corpus, chunk_text, get_embeddings, build_vectorstore, retrieve, generate_answer

def main():
    corpus_dir = os.path.join(os.path.dirname(__file__), "sample_corpus")
    text = load_corpus(corpus_dir)
    if not text.strip():
        print("No corpus.")
        return
    chunks = chunk_text(text, 512, 64)
    emb = get_embeddings()
    if not emb:
        print("No embeddings.")
        return
    store = build_vectorstore(chunks, emb)
    query = "What oil should I use in a Model T engine?"
    for k in [1, 3, 5, 10]:
        results = retrieve(store, query, top_k=min(k, len(chunks)))
        answer = generate_answer(query, results, "Answer based only on the context.", use_llm=bool(os.getenv("OPENAI_API_KEY")))
        print(f"--- top_k={k} ---")
        print(answer[:400] if answer else "(no answer)")
        print()

if __name__ == "__main__":
    main()
