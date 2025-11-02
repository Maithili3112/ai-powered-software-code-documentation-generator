#!/usr/bin/env python3
"""
Automated Documentation Generation Pipeline

Automatically generates comprehensive software-level documentation for Python codebases
using Gemma model, ChromaDB, Neo4j, and RAG pipeline.

Usage:
    python automate_docs.py --project_path "/path/to/project"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip loading .env

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))

from src.chunking.chunk_code import PyCodeChunker, chunk_directory
from src.embedding.embed_chunks import ChunkEmbedder
from src.graph.generate_call_graph import Neo4jGraphBuilder, PyCGCallGraphGenerator
from src.rag.rag_context import RAGContextRetriever
# Replace Hugging Face Gemma with Google Gemini API
import google.generativeai as genai
from src.reassembly.reassemble_docs import DocReassembler, reassemble_docs
from src.web.sphinx_builder import SphinxBuilder
from src.utils.helpers import save_chunks_to_jsonl, load_chunks_from_jsonl, group_chunks_by_file, extract_project_stats

class GeminiDocGenerator:
    """Generates documentation using Google's Gemini API."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def generate_documentation(self, code_chunk: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Build a prompt with code and RAG context and call Gemini to generate docs."""
        try:
            context_texts: List[str] = []
            for ctx in context_chunks or []:
                # ctx may be dicts with 'text' or similar; fall back to str(ctx)
                text = ctx.get("text") if isinstance(ctx, dict) else None
                context_texts.append(text if isinstance(text, str) else str(ctx))

            prompt = (
                "You are an expert software documentation generator.\n"
                "Given a Python code chunk and optional related context, write clear,\n"
                "concise, and comprehensive documentation that explains: purpose, public API,\n"
                "parameters, return values, side effects, exceptions, dependencies, and examples.\n\n"
                "Code chunk:\n" + code_chunk + "\n\n"
                "Related context (may include referenced symbols, neighboring functions, or usage):\n"
                + ("\n\n".join(context_texts) if context_texts else "(none)") + "\n\n"
                "Produce well-structured Markdown suitable for Sphinx."
            )

            model = self._get_model()
            response = model.generate_content(prompt)
            # google-generativeai returns .text for concatenated parts
            return getattr(response, "text", "") or ""
        except Exception as exc:
            logger.warning(f" Gemini generation failed, returning empty doc: {exc}")
            return ""

class DocumentationPipeline:
    """Main pipeline orchestrator for automated documentation generation."""
    
    def __init__(self, project_path: str, 
                 chroma_path: str = "./chroma_store",
                 rag_chroma_path: str = "./rag_chroma",
                 neo4j_uri: Optional[str] = None,
                 output_dir: str = "./generated_docs",
                 docs_output_dir: str = "./docs",
                 google_api_key: Optional[str] = None):
        
        self.project_path = Path(project_path)
        self.chroma_path = chroma_path
        self.rag_chroma_path = rag_chroma_path
        # Prefer explicit parameter, otherwise use environment NEO4J_URI, fallback to localhost
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
        self.output_dir = Path(output_dir)
        self.docs_output_dir = Path(docs_output_dir)
        # Configure Google Gemini API key (CLI arg takes precedence, then env)
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
            except Exception as e:
                logger.warning(f" Failed to configure Gemini API: {e}")
        
        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunks_output = self.output_dir / "chunks.jsonl"
    
    def run(self):
        """Execute the complete documentation pipeline."""
        logger.info("=" * 80)
        logger.info("🚀 Starting Automated Documentation Pipeline")
        logger.info("=" * 80)
        
        try:
            # STEP 1: Chunking
            logger.info("\n[STEP 1/7]  Chunking code files...")
            chunks = chunk_directory(str(self.project_path), str(self.output_dir))
            
            if not chunks:
                logger.error(" No chunks generated. Aborting.")
                return False
            
            logger.info(f"✓ Generated {len(chunks)} chunks")
            save_chunks_to_jsonl(chunks, str(self.chunks_output))
            
            # STEP 2: Embedding
            logger.info("\n[STEP 2/7]  Generating embeddings and storing in ChromaDB...")
            embedder = ChunkEmbedder(chroma_path=self.chroma_path)
            embedder.embed_chunks(chunks)
            logger.info(f"✓ Stored {len(chunks)} chunks in ChromaDB")
            
            # STEP 3: Call Graph Generation
            logger.info("\n[STEP 3/7]  Generating call graph and storing in Neo4j...")
            try:
                generator = PyCGCallGraphGenerator()
                generator.generate_call_graph(str(self.project_path), self.neo4j_uri)
                logger.info("✓ Generated call graph")
            except Exception as e:
                logger.warning(f" Call graph generation failed (continuing): {e}")
            
            # STEP 4: RAG Context Retrieval
            logger.info("\n[STEP 4/7]  Retrieving RAG context from CodexGLUE...")
            rag_retriever = RAGContextRetriever(rag_chroma_path=self.rag_chroma_path)
            logger.info("✓ RAG context retriever ready")
            
            # STEP 5: Generate Documentation
            logger.info("\n[STEP 5/7]   Generating documentation using Gemini API...")
            doc_generator = GeminiDocGenerator(model_name="gemini-2.5-flash")
            
            docs = []
            for i, chunk in enumerate(chunks):
                logger.info(f"   Processing chunk {i+1}/{len(chunks)}: {chunk.get('name', 'unknown')}")
                
                # Retrieve RAG context
                context_chunks = rag_retriever.retrieve_context(chunk['text'], n_results=5)
                
                # Generate documentation
                doc = doc_generator.generate_documentation(chunk['text'], context_chunks)
                docs.append(doc)
            
            logger.info(f"✓ Generated documentation for {len(docs)} chunks")
            
            # STEP 6: Reassemble Documentation
            logger.info("\n[STEP 6/7]  Reassembling documentation by file...")
            file_docs = reassemble_docs(chunks, docs, str(self.output_dir))
            logger.info(f"✓ Reassembled documentation for {len(file_docs)} files")
            
            # STEP 7: Build Sphinx HTML
            logger.info("\n[STEP 7/7]  Building Sphinx HTML documentation...")
            builder = SphinxBuilder(
                source_dir=str(self.output_dir),
                output_dir=str(self.docs_output_dir)
            )
            success = builder.build_all()
            
            if success:
                logger.info(f"✓ Built Sphinx documentation at {builder.build_dir}")
            
            # Print statistics
            self._print_stats(chunks, docs)
            
            logger.info("\n" + "=" * 80)
            logger.info(" PIPELINE COMPLETE!")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"\n Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_stats(self, chunks: List[Dict], docs: List[str]):
        """Print pipeline statistics."""
        logger.info("\n Pipeline Statistics:")
        logger.info(f"   • Total chunks: {len(chunks)}")
        logger.info(f"   • Files processed: {len(set(ch['filepath'] for ch in chunks))}")
        logger.info(f"   • Total lines: {sum(ch['end_line'] - ch['start_line'] + 1 for ch in chunks)}")
        logger.info(f"   • Documentation generated: {len(docs)} chunks")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automated documentation generation pipeline"
    )
    
    parser.add_argument(
        "--project_path",
        required=True,
        help="Path to project directory to document"
    )
    
    parser.add_argument(
        "--chroma-path",
        default="./chroma_store",
        help="Path to ChromaDB storage (default: ./chroma_store)"
    )
    
    parser.add_argument(
        "--rag-chroma-path",
        default="./rag_chroma",
        help="Path to RAG ChromaDB storage (default: ./rag_chroma)"
    )
    
    parser.add_argument(
        "--neo4j-uri",
        default=None,
        help="Neo4j connection URI (overrides NEO4J_URI env var if provided)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./generated_docs",
        help="Output directory for generated documentation (default: ./generated_docs)"
    )
    
    parser.add_argument(
        "--docs-dir",
        default="./docs",
        help="Sphinx docs directory (default: ./docs)"
    )
    
    parser.add_argument(
        "--google-api-key",
        help="Google API key (or use GOOGLE_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate project path
    if not Path(args.project_path).exists():
        logger.error(f"Project path does not exist: {args.project_path}")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = DocumentationPipeline(
        project_path=args.project_path,
        chroma_path=args.chroma_path,
        rag_chroma_path=args.rag_chroma_path,
        neo4j_uri=args.neo4j_uri,
        output_dir=args.output_dir,
        docs_output_dir=args.docs_dir,
        google_api_key=args.google_api_key
    )
    
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
