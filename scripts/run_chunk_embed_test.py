from pathlib import Path
import sys
p=Path(r"C:\Users\HARSHA\Desktop\calibrat")
print("TARGET_EXISTS:", p.exists())
if not p.exists():
    print("Project path does not exist, aborting test.")
    sys.exit(0)
from src.chunking.chunk_code import chunk_directory
from src.embedding.embed_chunks import ChunkEmbedder
out = Path(r"C:\Users\HARSHA\Desktop\mini\test_chroma_run")
out.mkdir(parents=True, exist_ok=True)
chunks = chunk_directory(str(p), str(out))
print("CHUNKS_FOUND:", len(chunks))
if not chunks:
    print("No chunks generated, aborting embedding test")
    sys.exit(0)
# instantiate embedder (CPU)
embedder = ChunkEmbedder(chroma_path=str(out/'chroma_store_test'), model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu')
print('Embedder ready, embedding up to first 10 chunks...')
embedder.embed_chunks(chunks[:10])
print('Embedded first 10 chunks into Chroma at', str(out/'chroma_store_test'))
