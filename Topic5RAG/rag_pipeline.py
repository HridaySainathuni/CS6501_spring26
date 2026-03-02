"""Topic 5: RAG pipeline - chunk, embed, retrieve, generate."""
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
import argparse
from pathlib import Path

def load_corpus(corpus_dir: str):
    """Load all .txt files from directory into one string."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        return ""
    text = []
    for f in corpus_dir.glob("**/*.txt"):
        text.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(text)

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64):
    """Split text into overlapping chunks."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        return splitter.split_text(text)
    except ImportError:
        # Fallback: simple sliding window
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap if overlap < chunk_size else end
        return chunks

def get_embeddings():
    """Use HuggingFace embeddings (no API key). Fallback to fake for testing."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        pass
    try:
        from langchain_openai import OpenAIEmbeddings
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        pass
    return None

def build_vectorstore(chunks, embeddings):
    """Build in-memory vector store from chunks."""
    if embeddings is None:
        return None
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        docs = [Document(page_content=c) for c in chunks]
        return FAISS.from_documents(docs, embeddings)
    except ImportError:
        return None

def retrieve(store, query: str, top_k: int = 5):
    """Return top_k relevant chunks."""
    if store is None:
        return []
    return store.similarity_search(query, k=top_k)

def generate_answer(query: str, context_chunks: list, prompt_template: str, use_llm: bool = True):
    """Generate answer from context. If no LLM, return concatenated context."""
    context = "\n\n".join(c.page_content if hasattr(c, "page_content") else str(c) for c in context_chunks)
    if not use_llm:
        return f"Context:\n{context[:500]}..."
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        if not os.getenv("OPENAI_API_KEY"):
            return f"[Set OPENAI_API_KEY for LLM] Context:\n{context[:300]}..."
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template or "Answer based only on the following context. If the answer is not in the context, say 'I cannot answer from the available documents.'"),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain = prompt | llm
        return chain.invoke({"context": context, "question": query}).content
    except Exception as e:
        return f"[Error: {e}]\nContext:\n{context[:300]}"

def main():
    p = argparse.ArgumentParser(description="RAG pipeline")
    p.add_argument("--corpus", default=os.path.join(os.path.dirname(__file__), "sample_corpus"), help="Directory with .txt files")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--query", default="What is the correct spark plug gap for a Model T?")
    p.add_argument("--no-llm", action="store_true", help="Only retrieve, no generation")
    p.add_argument("--prompt", default="", help="Custom system prompt for generation")
    args = p.parse_args()

    text = load_corpus(args.corpus)
    if not text.strip():
        print("No corpus found. Add .txt files to", args.corpus)
        return
    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Chunks: {len(chunks)} (size={args.chunk_size}, overlap={args.overlap})")

    embeddings = get_embeddings()
    if embeddings is None:
        print("No embeddings available. Install sentence-transformers or set OPENAI_API_KEY for embeddings.")
        print("Corpus loaded successfully; install langchain-community, langchain-huggingface (or langchain-openai), and faiss-cpu for full RAG.")
        return
    store = build_vectorstore(chunks, embeddings)
    if store is None:
        print("Vector store not built (missing langchain_community/FAISS).")
        return
    results = retrieve(store, args.query, top_k=args.top_k)
    print(f"Retrieved {len(results)} chunks for top_k={args.top_k}")
    answer = generate_answer(args.query, results, args.prompt, use_llm=not args.no_llm)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
