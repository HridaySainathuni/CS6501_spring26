"""Topic 5 Exercise 1: Compare LLM answers with and without RAG."""
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
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import load_corpus, chunk_text, get_embeddings, build_vectorstore, retrieve, generate_answer

def main():
    corpus_dir = os.path.join(os.path.dirname(__file__), "sample_corpus")
    text = load_corpus(corpus_dir)
    if not text.strip():
        print("No corpus. Add .txt to sample_corpus/")
        return
    query = "What is the correct spark plug gap for a Model T?"
    # Without RAG: call LLM directly (may hallucinate)
    print("=== Without RAG (direct LLM) ===")
    try:
        from langchain_openai import ChatOpenAI
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            no_rag = llm.invoke(f"Question: {query}").content
            print(no_rag)
        else:
            print("Set OPENAI_API_KEY for direct LLM.")
    except Exception as e:
        print("Error:", e)
    # With RAG
    print("\n=== With RAG ===")
    chunks = chunk_text(text, 512, 64)
    emb = get_embeddings()
    if emb:
        store = build_vectorstore(chunks, emb)
        results = retrieve(store, query, top_k=3)
        answer = generate_answer(query, results, "Answer based only on the context.", use_llm=True)
        print(answer)
    else:
        print("No embeddings (install sentence-transformers or set OPENAI_API_KEY).")

if __name__ == "__main__":
    main()
