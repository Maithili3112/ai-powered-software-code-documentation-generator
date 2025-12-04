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
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
import google.generativeai as genai
from src.reassembly.reassemble_docs import DocReassembler
from src.graph.graphviz.graphformmater import build_graphcommons_files
from src.utils.helpers import save_chunks_to_jsonl, load_chunks_from_jsonl, group_chunks_by_file, extract_project_stats

class DocumentationPipeline:
    """Main pipeline orchestrator for automated documentation generation."""
    
    def __init__(self, project_path: str, 
                 chroma_path: str = "./chroma_store",
                 rag_chroma_path: str = "./rag_chroma",
                 neo4j_uri: Optional[str] = None,
                 output_dir: str = "./generated_docs",
                 google_api_key: Optional[str] = None):
        
        self.project_path = Path(project_path)
        self.chroma_path = chroma_path
        self.rag_chroma_path = rag_chroma_path
        # Prefer explicit parameter, otherwise use environment NEO4J_URI, fallback to localhost
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'neo4j://127.0.0.1:7687')
        self.output_dir = Path(output_dir)
        self.gemma_interface_key = google_api_key or os.getenv("GEMMA_INTERFACE_KEY")
        # We keep this for backward compatibility but don't configure genai here
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunks_output = self.output_dir / "chunks.jsonl"
    
    def run(self):
        """Execute the complete documentation pipeline."""
        logger.info("=" * 80)
        logger.info(" Starting Automated Documentation Pipeline")
        logger.info("=" * 80)
        
        try:
            # STEP 1: Chunking
            logger.info("\n[STEP 1/7]  Chunking code files...")
            chunks = chunk_directory(str(self.project_path), str(self.output_dir))
            
            if not chunks:
                logger.error(" No chunks generated. Aborting.")
                return False
            
            logger.info(f"Generated {len(chunks)} chunks")
            save_chunks_to_jsonl(chunks, str(self.chunks_output))
            
            # STEP 2: Embedding
            logger.info("\n[STEP 2/7]  Generating embeddings and storing in ChromaDB...")
            embedder = ChunkEmbedder(chroma_path=self.chroma_path)
            embedder.embed_chunks(chunks)
            logger.info(f" Stored {len(chunks)} chunks in ChromaDB")
            
            # STEP 3: Call Graph Generation
            logger.info("\n[STEP 3/7]  Generating call graph and storing in Neo4j...")
            try:
                generator = PyCGCallGraphGenerator()
                generator.generate_call_graph(str(self.project_path), self.neo4j_uri)
                logger.info(" Generated call graph")
            except Exception as e:
                logger.warning(f" Call graph generation failed (continuing): {e}")
            
            # STEP 4: RAG Context Retrieval
            
            rag_retriever = RAGContextRetriever(rag_chroma_path=self.rag_chroma_path)
           
            
            # STEP 5: Generate Documentation
            logger.info("\n[STEP 5/7]   Generating documentation using LLM...")
            # Collect API keys for DocGenerator
            api_keys = []
            if self.gemma_interface_key:
                api_keys.append(self.gemma_interface_key)
            # Check for second API key
            secondary_key = os.getenv("GEMMA_INTERFACE_KEY_2")
            if secondary_key:
                api_keys.append(secondary_key)
            
            doc_generator = DocGenerator(api_keys=api_keys if api_keys else None)
            dep_resolver = GraphDependencyResolver(self.neo4j_uri)
            
            docs = []
            per_chunk_deps: List[List[Dict[str, Any]]] = []
            dep_csv_rows: List[str] = ["file,chunk_name,chunk_start,chunk_end,dep_name,dep_location,direction,source"]
            for i, chunk in enumerate(chunks):
                logger.info(f"   Processing chunk {i+1}/{len(chunks)}: {chunk.get('name', 'unknown')}")
                
                # Retrieve RAG context
                context_chunks = rag_retriever.retrieve_context(chunk['text'], n_results=5)
                # Resolve dependencies via graph (with AST fallback)
                deps, dep_source = dep_resolver.resolve_for_chunk_with_source(chunk)
                per_chunk_deps.append(deps)
                # record CSV rows
                for d in deps:
                    dep_csv_rows.append(
                        f"{chunk.get('filepath','')},{chunk.get('name','')},{chunk.get('start_line',0)},{chunk.get('end_line',0)},{d.get('name','')},{d.get('location','')},{d.get('direction','outgoing')},{dep_source}"
                    )
                
                # Generate documentation
                doc = doc_generator.generate_chunk_doc(chunk, deps, context_chunks)
                docs.append(doc)
            
            logger.info(f" Generated documentation for {len(docs)} chunks")
            
            # STEP 6: Reassemble Documentation and append file-level summaries
            logger.info("\n[STEP 6/7]  Reassembling documentation by file and generating file summaries...")
            reassembler = DocReassembler(str(self.output_dir))
            file_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
            file_to_docs: Dict[str, List[str]] = {}
            file_to_deps: Dict[str, List[Dict[str, Any]]] = {}
            for idx, ch in enumerate(chunks):
                fp = ch["filepath"]
                file_to_chunks.setdefault(fp, []).append(ch)
                file_to_docs.setdefault(fp, []).append(docs[idx] if idx < len(docs) else "")
                for d in per_chunk_deps[idx] if idx < len(per_chunk_deps) else []:
                    file_to_deps.setdefault(fp, []).append(d)

            # Build combined file docs in original order
            combined_docs: Dict[str, str] = {}
            for fp, ch_list in file_to_chunks.items():
                # sort by line
                pairs = list(zip(ch_list, file_to_docs.get(fp, [])))
                pairs.sort(key=lambda x: x[0]['start_line'])
                combined = [f"# File: {fp}\n"]
                for i, (ch, doc) in enumerate(pairs, 1):
                    combined.append(f"\n## Chunk {i} (lines {ch['start_line']}-{ch['end_line']})\n\n")
                    # Include the exact chunk code for reference
                    combined.append("```python\n" + (ch.get('text','')) + "\n```\n\n")
                    combined.append(doc)
                    combined.append("\n---\n")
                # File summary at end
                file_summary = doc_generator.generate_file_summary(fp, [p[0] for p in pairs], file_to_deps.get(fp, []))
                if file_summary:
                    combined.append("\n\n" + file_summary + "\n")
                combined_docs[fp] = "\n".join(combined)

            # Save to disk and create index
            reassembler.save_file_docs(combined_docs, preserve_structure=True)
            index = reassembler.create_index(combined_docs)
            (reassembler.output_dir / "index.md").write_text(index)
            logger.info(f"Reassembled documentation for {len(combined_docs)} files")

            # Write dependency CSV (especially useful if AST fallback was used)
            dep_csv_path = self.output_dir / "function_dependencies.csv"
            try:
                dep_csv_path.write_text("\n".join(dep_csv_rows), encoding="utf-8")
                logger.info(f"Wrote dependency CSV to {dep_csv_path}")
            except Exception as e:
                logger.warning(f"Failed to write dependency CSV: {e}")

            # Build graph data (nodes/edges CSVs) for visualization
            try:
                logger.info("Building function dependency graph data (nodes/edges CSVs)...")
                build_graphcommons_files(
                    dependencies_csv=str(dep_csv_path),
                    edges_output="./src/graph/graphviz/connections_graphcommons.csv",
                    nodes_output="./src/graph/graphviz/nodes_graphcommons.csv",
                )
                logger.info("Function dependency graph CSVs generated.")
            except Exception as e:
                logger.warning(f"Failed to build dependency graph CSVs (continuing): {e}")

            # Generate README using project structure, summaries, and requirements
            try:
                # Project structure from file_to_chunks keys
                paths = sorted(file_to_chunks.keys())
                structure_lines = []
                for p in paths:
                    structure_lines.append(f"- {p}")
                project_structure_text = "\n".join(structure_lines)

                # Collect file summaries (extract from combined content)
                file_summaries_text = []
                for fp, content in combined_docs.items():
                    # naive extraction: take the last 'File Summary' section if present
                    marker = "File Summary"
                    snippet = content
                    if marker in content:
                        snippet = content[content.rfind(marker):]
                    file_summaries_text.append(f"## {fp}\n\n" + snippet[:1500])
                file_summaries_joined = "\n\n".join(file_summaries_text)

                # Requirements: prefer project requirements.txt, else infer from imports
                requirements_path = Path(self.project_path) / "requirements.txt"
                if requirements_path.exists():
                    requirements_text = requirements_path.read_text(encoding="utf-8")[:4000]
                else:
                    # Infer from imports (very rough)
                    imports = []
                    for chs in file_to_chunks.values():
                        for ch in chs:
                            imports.extend(ch.get("imports", []) or [])
                    # Extract top-level package names heuristically
                    pkgs = []
                    for imp in set(imports):
                        line = imp.strip()
                        if line.startswith("from "):
                            pkg = line.split()[1].split(".")[0]
                            pkgs.append(pkg)
                        elif line.startswith("import "):
                            pkg = line.split()[1].split(",")[0].split(".")[0]
                            pkgs.append(pkg)
                    requirements_text = "\n".join(sorted(set(pkgs)))

                readme = doc_generator.generate_project_readme(project_structure_text, file_summaries_joined, requirements_text)
                if readme:
                    (self.output_dir / "README.md").write_text(readme, encoding="utf-8")
                    logger.info("Generated project README.md in output directory")
            except Exception as e:
                logger.warning(f"Failed to generate README: {e}")

            # Dependency coverage / graph validation
            try:
                total = len(chunks)
                with_deps = sum(1 for deps in per_chunk_deps if deps)
                coverage = (with_deps / total * 100.0) if total else 0.0
                logger.info(f"Dependency coverage: {with_deps}/{total} chunks ({coverage:.1f}%) have dependencies")
                if coverage == 0.0:
                    logger.warning("No dependencies were resolved. Check Neo4j connectivity and AST fallback logic.")
            except Exception:
                pass
            
            # Final statistics
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
class GeminiAPIKeyManager:
    """Manages multiple Gemini API keys with automatic fallback on rate limits."""
    
    def __init__(self, api_keys: List[str], model_name: str = "gemini-2.5-flash"):
        """
        Initialize the API key manager.
        
        Args:
            api_keys: List of API keys to use (at least one required)
            model_name: Gemini model name to use
        """
        if not api_keys or not any(api_keys):
            raise ValueError("At least one API key is required")
        
        # Filter out None/empty keys
        self.api_keys = [key for key in api_keys if key]
        if not self.api_keys:
            raise ValueError("No valid API keys provided")
        
        self.model_name = model_name
        self.current_key_index = 0
        self.models = {}  # Cache models for each key
        self.rate_limited_keys = set()  # Track which keys are currently rate limited
        self.rate_limit_reset_time = {}  # Track when rate limits reset (60 seconds)
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit error."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check for common rate limit indicators
        rate_limit_indicators = [
            "429",
            "quota",
            "rate limit",
            "rate_limit",
            "quota exceeded",
            "resource exhausted",
            "per minute",
            "requests per minute",
            "rpm",
            "quotaexceeded",
        ]
        
        return any(indicator in error_str for indicator in rate_limit_indicators) or "429" in error_type
    
    def _get_current_key(self) -> str:
        """Get the currently active API key."""
        return self.api_keys[self.current_key_index]
    
    def _switch_to_next_key(self):
        """Switch to the next available API key."""
        current_time = time.time()
        
        # Check if any rate-limited keys can be reset (60 seconds passed)
        keys_to_reset = []
        for key_idx, reset_time in self.rate_limit_reset_time.items():
            if current_time >= reset_time:
                keys_to_reset.append(key_idx)
        
        for key_idx in keys_to_reset:
            self.rate_limited_keys.discard(key_idx)
            del self.rate_limit_reset_time[key_idx]
            logger.info(f"API key {key_idx + 1} is now available again after rate limit reset")
        
        # Try to find a non-rate-limited key
        original_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.api_keys):
            if self.current_key_index not in self.rate_limited_keys:
                if self.current_key_index != original_index:
                    logger.info(f"Switched to API key {self.current_key_index + 1}")
                return
            
            # Move to next key
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            attempts += 1
        
        # If all keys are rate limited, use the current one anyway and log warning
        logger.warning(f"All API keys are rate limited. Using key {self.current_key_index + 1} and will retry...")
    
    def _get_model(self, api_key: str):
        """Get or create a model instance for the given API key."""
        if api_key not in self.models:
            genai.configure(api_key=api_key)
            self.models[api_key] = genai.GenerativeModel(self.model_name)
        return self.models[api_key]
    
    def generate_content(self, prompt: str, max_retries: int = 2):
        """
        Generate content using the current API key, with automatic fallback on rate limits.
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Maximum number of retries with different keys
            
        Returns:
            Response from the model
            
        Raises:
            Exception: If all keys fail or non-rate-limit error occurs
        """
        for attempt in range(max_retries + 1):
            try:
                # Get current key and ensure genai is configured with it
                current_key = self._get_current_key()
                # Reconfigure genai with current key (in case we switched)
                genai.configure(api_key=current_key)
                model = self._get_model(current_key)
                
                # Try to generate content
                response = model.generate_content(prompt)
                return response
                
            except Exception as e:
                if self._is_rate_limit_error(e):
                    # Mark current key as rate limited
                    self.rate_limited_keys.add(self.current_key_index)
                    self.rate_limit_reset_time[self.current_key_index] = time.time() + 60  # Reset after 60 seconds
                    # Switch to next key
                    self._switch_to_next_key()
                    
                    # If we have more attempts, continue
                    if attempt < max_retries:
                       
                        continue
                    else:
                        # All keys exhausted, raise the error
                        logger.error("All interface have hit rate limits. Please wait before retrying.")
                        raise
                else:
                    # Non-rate-limit error, raise immediately
                    
                    raise
        
        # Should not reach here, but just in case
        raise Exception("Failed to generate content after all retries")


class DocGenerator:
    

    CHUNK_PROMPT_TEMPLATE = (
        "You are a senior Python engineer writing concise, precise technical docs.\n"
        "Context: The file is processed in chunks; you will document ONLY the given chunk.\n"
        "Rules:\n"
        "- Be clear and simple, avoid verbose theory.\n"
        "- Do not speculate about usage beyond provided dependencies; if unknown, say so.\n"
        "- If purpose is not evident, state 'purpose unclear from this chunk'.\n"
        "- Use Markdown with small sections: Purpose, Parameters, Returns, Exceptions, Notes.\n"
        "- Do NOT include a header like '## Chunk ...'; that header is already provided.\n"
        "- Do NOT repeat the raw code; it will be displayed above your documentation.\n"
        "- Include an explicit 'Dependencies' section listing incoming and outgoing functions using ONLY the provided list.\n"
        "- Include 'Inter-Module Relationships' summarizing how this chunk interacts with others (based on dependencies and imports).\n"
        "- Include 'Maintainability & Scalability' notes (readability, modularity, extensibility).\n"
        "Inputs:\n"
        "- File: {filepath} (lines {start_line}-{end_line})\n"
        "- Dependencies (from call graph):\n{dependencies}\n"
        "- Related context (from RAG):\n{rag_context}\n"
        "Code:\n```python\n{code}\n```\n"
        "Write the documentation now."
    )

    FILE_SUMMARY_PROMPT_TEMPLATE = (
        "You are a senior Python engineer. Provide a short file-level summary.\n"
        "Rules: 1-2 short paragraphs max. Focus on purpose, main responsibilities, and key integration points.\n"
        "Avoid assumptions not supported by code or dependencies.\n"
        "Inputs:\n"
        "- File: {filepath}\n"
        "- Functions/classes: {symbols}\n"
        "- Imports: {imports}\n"
        "- Notable dependencies (from call graph):\n{dependencies}\n"
        "Code (for reference, truncated as needed):\n```python\n{code}\n```\n"
        "Write the summary now in Markdown under a heading 'File Summary'."
    )

    README_PROMPT_TEMPLATE = (
        "You are a senior Python engineer. Generate a high-quality README for the project.\n"
        "Output sections:\n"
        "1. Project Overview (brief)\n"
        "2. Architecture & Structure (summarize directories and key files)\n"
        "3. Installation (list exact requirements; if a requirements.txt is provided, include it clearly; otherwise infer from imports)\n"
        "4. Usage (how to run the pipeline/commands)\n"
        "5. Documentation (where generated docs live)\n"
        "6. Notes on Call Graph & RAG (how dependencies are handled, limitations)\n"
        "Inputs:\n"
        "- Project structure:\n{project_structure}\n"
        "- File summaries:\n{file_summaries}\n"
        "- Proposed requirements:\n{requirements_text}\n"
        "Constraints:\n"
        "- Keep it concise and accurate.\n"
        "- Do not fabricate packages; prefer the provided requirements list.\n"
        "- Use Markdown formatting.\n"
    )

    def __init__(self, model_name: str = "gemini-2.5-flash", api_keys: Optional[List[str]] = None):
        """
        Initialize the DocGenerator.
        
        Args:
            model_name: Gemini model name to use
            api_keys: List of API keys (supports multiple keys for fallback)
        """
        self.model_name = model_name
        
        # Get API keys from parameter or environment
        if api_keys:
            keys = api_keys
        else:
            # Try to get multiple keys from environment
            primary_key = os.getenv("GEMMA_INTERFACE_KEY")
            secondary_key = os.getenv("GEMMA_INTERFACE_KEY_2")
            keys = [k for k in [primary_key, secondary_key] if k]
        
        # Initialize API key manager
        if keys:
            self.api_key_manager = GeminiAPIKeyManager(keys, model_name)
        else:
            self.api_key_manager = None
            logger.warning("No Gemini API keys found. Documentation generation may fail.")

    def _format_dependencies(self, deps: List[Dict[str, Any]]) -> str:
        if not deps:
            return "(none provided)"
        lines: List[str] = []
        for d in deps:
            name = d.get("name", d.get("callee", "?"))
            location = d.get("location", "")
            direction = d.get("direction", "outgoing")
            lines.append(f"- [{direction}] {name} {location}")
        return "\n".join(lines)

    def _format_rag_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return "(none)"
        parts: List[str] = []
        for i, ctx in enumerate(context_chunks[:5], 1):
            text = ctx.get("text", "")
            parts.append(f"Context {i}:\n```python\n{text[:500]}\n```")
        return "\n\n".join(parts)

    def generate_chunk_doc(
        self,
        chunk: Dict[str, Any],
        dependencies: List[Dict[str, Any]],
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        try:
            prompt = self.CHUNK_PROMPT_TEMPLATE.format(
                filepath=chunk.get("filepath", ""),
                start_line=chunk.get("start_line", 0),
                end_line=chunk.get("end_line", 0),
                dependencies=self._format_dependencies(dependencies),
                rag_context=self._format_rag_context(context_chunks),
                code=chunk.get("text", ""),
            )
            if not self.api_key_manager:
                logger.error("No API key manager available. Cannot generate documentation.")
                return ""
            
            response = self.api_key_manager.generate_content(prompt)
            text = getattr(response, "text", "") or ""
            # Remove any model-produced generic chunk headers like "## Chunk X (...)"
            lines = text.splitlines()
            filtered = []
            for line in lines:
                l = line.strip().lower()
                if l.startswith("## chunk "):
                    continue
                filtered.append(line)
            return "\n".join(filtered).strip()
        except Exception as exc:
            logger.warning(f" LLM generation failed for chunk, returning empty doc: {exc}")
            return ""

    def generate_project_readme(self, project_structure: str, file_summaries_text: str, requirements_text: str) -> str:
        try:
            prompt = self.README_PROMPT_TEMPLATE.format(
                project_structure=project_structure,
                file_summaries=file_summaries_text,
                requirements_text=requirements_text,
            )
            if not self.api_key_manager:
                logger.error("No API key manager available. Cannot generate README.")
                return ""
            
            response = self.api_key_manager.generate_content(prompt)
            return getattr(response, "text", "") or ""
        except Exception as exc:
            logger.warning(f" LLM generation failed for README, returning empty: {exc}")
            return ""

    def generate_file_summary(
        self,
        filepath: str,
        file_chunks: List[Dict[str, Any]],
        file_dependencies: List[Dict[str, Any]],
    ) -> str:
        try:
            # Aggregate code (truncate to keep prompt size reasonable)
            code_text = "\n\n".join(ch.get("text", "") for ch in file_chunks)
            if len(code_text) > 6000:
                code_text = code_text[:6000]

            symbols = ", ".join(ch.get("name", "") for ch in file_chunks if ch.get("name"))
            imports = []
            for ch in file_chunks:
                imports.extend(ch.get("imports", []) or [])
            imports_str = ", ".join(sorted(set(imports)))[:500]

            prompt = self.FILE_SUMMARY_PROMPT_TEMPLATE.format(
                filepath=filepath,
                symbols=symbols or "(none)",
                imports=imports_str or "(none)",
                dependencies=self._format_dependencies(file_dependencies),
                code=code_text,
            )
            if not self.api_key_manager:
                logger.error("No API key manager available. Cannot generate file summary.")
                return ""
            
            response = self.api_key_manager.generate_content(prompt)
            return getattr(response, "text", "") or ""
        except Exception as exc:
            logger.warning(f" LLM generation failed for file summary, returning empty: {exc}")
            return ""


class GraphDependencyResolver:
    """Resolve function dependencies for chunks using Neo4j if available; fallback to AST."""

    def __init__(self, neo4j_uri: str):
        self.neo4j_uri = neo4j_uri
        self._driver = None
        self._connect()

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD")
            self._driver = GraphDatabase.driver(self.neo4j_uri, auth=(user, password))
            with self._driver.session() as sess:
                sess.run("RETURN 1 as n")
        except Exception:
            self._driver = None

    def close(self):
        try:
            if self._driver:
                self._driver.close()
        except Exception:
            pass

    def _extract_calls_ast(self, code: str) -> List[str]:
        try:
            import ast
            tree = ast.parse(code)
            calls: List[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = self._name_from_node(node.func)
                    if name:
                        calls.append(name)
            return list(sorted(set(calls)))
        except Exception:
            return []

    def _name_from_node(self, node) -> Optional[str]:
        try:
            import ast
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                parts = []
                cur = node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                parts.reverse()
                return ".".join(parts)
        except Exception:
            return None
        return None

    def resolve_for_chunk_with_source(self, chunk: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        # Try Neo4j first by matching functions in the file and range
        deps: List[Dict[str, Any]] = []
        if self._driver:
            try:
                with self._driver.session() as sess:
                    # Outgoing calls from the chunk's functions
                    q_out = (
                        "MATCH (f:Function) "
                        "WHERE f.filepath = $fp AND f.start_line >= $s AND f.end_line <= $e "
                        "OPTIONAL MATCH (f)-[:CALLS]->(g:Function) "
                        "RETURN g.name as name, g.filepath as gfp, g.start_line as gs, g.end_line as ge"
                    )
                    res_out = sess.run(q_out, fp=chunk.get("filepath"), s=chunk.get("start_line", 0), e=chunk.get("end_line", 0))
                    for r in res_out:
                        if r["name"]:
                            deps.append({
                                "name": r["name"],
                                "location": f"({r['gfp']}:{r['gs']}-{r['ge']})" if r.get("gfp") else "",
                                "direction": "outgoing",
                            })
                    # Incoming calls to the chunk's functions (same file prioritised)
                    q_in = (
                        "MATCH (f:Function) "
                        "WHERE f.filepath = $fp AND f.start_line >= $s AND f.end_line <= $e "
                        "MATCH (h:Function)-[:CALLS]->(f) "
                        "RETURN h.name as name, h.filepath as hfp, h.start_line as hs, h.end_line as he"
                    )
                    res_in = sess.run(q_in, fp=chunk.get("filepath"), s=chunk.get("start_line", 0), e=chunk.get("end_line", 0))
                    for r in res_in:
                        if r["name"]:
                            deps.append({
                                "name": r["name"],
                                "location": f"({r['hfp']}:{r['hs']}-{r['he']})" if r.get("hfp") else "",
                                "direction": "incoming",
                            })
                if deps:
                    return deps, "neo4j"
            except Exception:
                pass
        # Fallback to AST extraction of call names when graph is unavailable or empty
        ast_calls = self._extract_calls_ast(chunk.get("text", ""))
        for name in ast_calls:
            deps.append({"name": name, "location": "", "direction": "outgoing"})
        return deps, "ast"


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
        "--gemma-interface-key",
        dest="gemma_interface_key",
        help="Gemma interface key (or use GEMMA_INTERFACE_KEY env var). For fallback, also set GEMMA_INTERFACE_KEY_2 env var."
    )

    parser.add_argument(
        "--read_from_input",
        action="store_true",
        help="Flag maintained for backward compatibility; input is always read from the provided project_path."
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
        google_api_key=args.gemma_interface_key
    )
    
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
