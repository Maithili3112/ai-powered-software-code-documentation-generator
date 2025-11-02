"""
Test script: embed sample chunks into a local ChromaDB collection named
`codexglue_chunks` and run RAGContextRetriever to retrieve relevant chunks.

Run: python scripts/test_rag_chroma.py
"""

import logging
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Use local paths inside the repo
RAG_CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'test_rag_chroma')
RAG_CHROMA_PATH = os.path.abspath(RAG_CHROMA_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    print(f"Using RAG ChromaDB path: {RAG_CHROMA_PATH}")

    # Initialize model (small one for CPU-friendly testing)
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name, device='cpu')

    # Setup chroma client
    client = chromadb.PersistentClient(path=RAG_CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name='codexglue_chunks', metadata={"description": "Test CodexGLUE chunks"})

    # Sample texts to embed
    texts = [
        "def foo():\n    \"\"\"Simple function that returns 1\"\"\"\n    return 1",
        "def bar(x):\n    \"\"\"Multiply by two\"\"\"\n    return x * 2",
        "def baz(y):\n    \"Example: uses foo() and bar()\"\n    return bar(y) + foo()",
    ]

    ids = [f"test_{i+1}" for i in range(len(texts))]
    metadatas = [{'name': f'snippet_{i+1}'} for i in range(len(texts))]

    print("Computing embeddings...")
    embeddings = model.encode(texts, convert_to_tensor=False)

    # Normalize embeddings to plain lists
    embeddings_list = []
    for emb in embeddings:
        try:
            embeddings_list.append(emb.tolist())
        except Exception:
            embeddings_list.append(list(emb))

    print("Adding documents to ChromaDB collection 'codexglue_chunks'...")
    # Add or upsert - remove existing ids first to avoid duplicates
    try:
        # Attempt to delete existing ids to ensure test is deterministic
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings_list)
    print(f"Added {len(ids)} items to ChromaDB at {RAG_CHROMA_PATH}")

    # Now test retrieval using local RAG retriever from the project
    from src.rag.rag_context import RAGContextRetriever

    retriever = RAGContextRetriever(rag_chroma_path=RAG_CHROMA_PATH, model_name=model_name, device='cpu')
    print("Retrieving context for query: 'def foo'\n")
    results = retriever.retrieve_context("def foo", n_results=3)

    print("Retrieved results:")
    for i, r in enumerate(results, 1):
        print(f"Result {i} - id: {r.get('id')} similarity: {r.get('similarity')}")
        print(r.get('text'))
        print('-' * 40)

    print("Test completed successfully")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Test failed: {e}")

