#!/usr/bin/env python3
"""
Test script to verify pipeline works without requiring HuggingFace token.
"""

import logging
from pathlib import Path
from src.chunking.chunk_code import PyCodeChunker, chunk_directory
from src.embedding.embed_chunks import ChunkEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the main pipeline steps."""
    
    project_path = "./test_project"
    
    logger.info("=" * 80)
    logger.info("🧪 Testing Documentation Pipeline")
    logger.info("=" * 80)
    
    # STEP 1: Chunking
    logger.info("\n[STEP 1] Chunking code files...")
    chunks = chunk_directory(project_path, "./test_generated_docs")
    logger.info(f"✓ Generated {len(chunks)} chunks")
    
    if not chunks:
        logger.error("No chunks generated!")
        return False
    
    # STEP 2: Embedding
    logger.info("\n[STEP 2] Generating embeddings...")
    try:
        embedder = ChunkEmbedder(chroma_path="./test_chroma_store")
        embedder.embed_chunks(chunks)
        logger.info(f"✓ Successfully embedded {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE TEST COMPLETE!")
    logger.info("=" * 80)
    logger.info("✓ Chunking: Working")
    logger.info("✓ Embedding: Working")
    logger.info(f"✓ Total chunks processed: {len(chunks)}")
    
    return True

if __name__ == "__main__":
    test_pipeline()
