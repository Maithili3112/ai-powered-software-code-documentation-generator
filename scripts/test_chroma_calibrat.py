"""
Test Chroma pipeline on local project 'calibrat'.

This script will:
- Chunk up to 50 Python files from the target project using the repo's chunker
- Embed the first 10 chunks into a local Chroma collection under ./test_chroma_pipeline_calibrat
- Use the project's RAGContextRetriever to query for similar chunks
- Print results

Run: python scripts/test_chroma_calibrat.py
"""

import sys
from pathlib import Path
import os
import logging

# Ensure repo root is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow overriding target via CLI arg or environment variable for flexibility
if len(sys.argv) > 1:
    TARGET = Path(sys.argv[1])
else:
    TARGET = Path(os.environ.get('TARGET_PATH', r"C:\Users\HARSHA\Desktop\calibrat"))
if not TARGET.exists():
    print(f"Target project not found: {TARGET}")
    sys.exit(1)

# Safe local test paths
TEST_WORKDIR = repo_root / "test_chroma_pipeline_calibrat"
TEST_WORKDIR.mkdir(parents=True, exist_ok=True)
CHROMA_PATH = str(TEST_WORKDIR / "chroma_store")

print(f"Target: {TARGET}")
print(f"Test workdir: {TEST_WORKDIR}")
print(f"Chroma path: {CHROMA_PATH}")

# Use project chunker
from src.chunking.chunk_code import chunk_directory
from src.embedding.embed_chunks import ChunkEmbedder
from src.rag.rag_context import RAGContextRetriever

# Chunk the target (this may take a bit)
print("Running chunker...")
chunks = chunk_directory(str(TARGET), str(TEST_WORKDIR))
print(f"Chunks generated: {len(chunks)}")
if not chunks:
    print("No chunks produced; aborting.")
    sys.exit(1)

# Embed first N chunks
N = min(10, len(chunks))
print(f"Embedding first {N} chunks...")
embedder = ChunkEmbedder(chroma_path=CHROMA_PATH, model_name=os.getenv('SENTENCE_TRANSFORMER_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'), device=os.getenv('MODEL_DEVICE', 'cpu'))
embedder.embed_chunks(chunks[:N])
print("Embedding complete.")

# Retrieval test: use a short query taken from the first chunk's text
query = chunks[0]['text'].split('\n')[0].strip() if chunks else 'def foo'
print(f"Querying for: {query!r}")
retriever = RAGContextRetriever(rag_chroma_path=CHROMA_PATH, model_name=os.getenv('SENTENCE_TRANSFORMER_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'), device=os.getenv('MODEL_DEVICE', 'cpu'))
results = retriever.retrieve_context(query, n_results=5)

print(f"Retrieved {len(results)} results")
for i, r in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print("id:", r.get('id'))
    print("similarity:", r.get('similarity'))
    print(r.get('text')[:1000])
    print()

print('Test finished. Chroma DB stored at', CHROMA_PATH)
print('To clean up: remove', TEST_WORKDIR)
