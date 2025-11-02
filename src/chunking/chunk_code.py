"""
Code chunking module that breaks down Python files into logical chunks.
Uses AST parsing to identify function/class boundaries for intelligent chunking.
"""

import ast
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a single code chunk."""
    id: str
    filepath: str
    start_line: int
    end_line: int
    text: str
    node_type: str
    name: str
    summary: str
    imports: List[str]
    language: str = "python"


class PyCodeChunker:
    """Python code chunker using AST parsing."""
    
    def __init__(self, max_chunk_size: int = 25000, min_chunk_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunks: List[CodeChunk] = []
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Conservative estimate: ~4 chars per token
        return len(text) // 4
    
    def create_chunk_id(self, filepath: str, start_line: int, end_line: int, text: str) -> str:
        """Create unique chunk ID."""
        content = f"{filepath}:{start_line}:{end_line}:{text}"
        return f"sha1:{hashlib.sha1(content.encode()).hexdigest()}"
    
    def extract_imports(self, source: str) -> List[str]:
        """Extract import statements from source code."""
        imports = []
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"from {module} import {alias.name}")
        except SyntaxError:
            # Fallback to line-based extraction
            for line in source.split('\n'):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)
        return imports
    
    def generate_summary(self, node_type: str, name: str) -> str:
        """Generate a summary for a chunk."""
        if node_type == 'function':
            return f"Function: {name}"
        elif node_type == 'class':
            return f"Class: {name}"
        elif node_type == 'module':
            return "Module-level code"
        else:
            return f"{node_type.title()}: {name}"
    
    def chunk_file(self, filepath: str) -> List[CodeChunk]:
        """Chunk a Python file into logical units."""
        logger.info(f"Chunking file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return []
        
        chunks = []
        lines = source.split('\n')
        
        # Parse AST to find logical boundaries
        try:
            tree = ast.parse(source)
            visitor = ChunkVisitor(filepath, lines)
            visitor.visit(tree)
            chunks = visitor.chunks
        except SyntaxError as e:
            logger.warning(f"AST parsing failed for {filepath}: {e}")
            # Fallback: chunk by file
            chunks.append(CodeChunk(
                id=self.create_chunk_id(filepath, 1, len(lines), source),
                filepath=filepath,
                start_line=1,
                end_line=len(lines),
                text=source,
                node_type='file',
                name=Path(filepath).stem,
                summary=f"File: {Path(filepath).name}",
                imports=self.extract_imports(source)
            ))
        
        # Post-process: merge small chunks and check size
        processed_chunks = []
        for chunk in chunks:
            token_count = self.estimate_tokens(chunk.text)
            if token_count < self.min_chunk_size:
                # Try to merge with previous chunk
                if processed_chunks and self.estimate_tokens(processed_chunks[-1].text) < self.max_chunk_size:
                    prev_chunk = processed_chunks[-1]
                    merged_text = prev_chunk.text + '\n\n' + chunk.text
                    if self.estimate_tokens(merged_text) <= self.max_chunk_size:
                        prev_chunk.text = merged_text
                        prev_chunk.end_line = chunk.end_line
                        prev_chunk.summary = f"{prev_chunk.summary}, {chunk.summary}"
                        continue
            
            processed_chunks.append(chunk)
        
        return processed_chunks


class ChunkVisitor(ast.NodeVisitor):
    """AST visitor to extract function and class definitions."""
    
    def __init__(self, filepath: str, lines: List[str]):
        self.filepath = filepath
        self.lines = lines
        self.chunks: List[CodeChunk] = []
        self.current_class = None
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        chunk_name = node.name
        
        if self.current_class:
            chunk_name = f"{self.current_class}.{chunk_name}"
            node_type = 'method'
        else:
            node_type = 'function'
        
        chunk_text = '\n'.join(self.lines[start_line-1:end_line])
        
        chunk = CodeChunk(
            id=f"chunk_{len(self.chunks)}",
            filepath=self.filepath,
            start_line=start_line,
            end_line=end_line,
            text=chunk_text,
            node_type=node_type,
            name=chunk_name,
            summary=f"{node_type.title()}: {chunk_name}",
            imports=self._extract_local_imports(chunk_text)
        )
        self.chunks.append(chunk)
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Visit class definition."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        chunk_text = '\n'.join(self.lines[start_line-1:end_line])
        
        self.current_class = node.name
        
        chunk = CodeChunk(
            id=f"chunk_{len(self.chunks)}",
            filepath=self.filepath,
            start_line=start_line,
            end_line=end_line,
            text=chunk_text,
            node_type='class',
            name=node.name,
            summary=f"Class: {node.name}",
            imports=self._extract_local_imports(chunk_text)
        )
        self.chunks.append(chunk)
        
        # Visit class body
        for child in node.body:
            self.visit(child)
        
        self.current_class = None
    
    def _extract_local_imports(self, text: str) -> List[str]:
        """Extract imports from text snippet."""
        imports = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports


def chunk_directory(root_path: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Chunk all Python files in a directory.
    
    Args:
        root_path: Root directory to process
        output_path: Output directory for chunks
    
    Returns:
        List of chunk dictionaries
    """
    root = Path(root_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    
    chunker = PyCodeChunker()
    all_chunks = []
    
    # Find all Python files
    py_files = list(root.rglob("*.py"))
    logger.info(f"Found {len(py_files)} Python files")
    
    for py_file in py_files:
        # Skip hidden files and common ignore patterns
        if any(part.startswith('.') for part in py_file.parts):
            continue
        if 'venv' in py_file.parts or '__pycache__' in py_file.parts:
            continue
        
        chunks = chunker.chunk_file(str(py_file))
        
        for chunk in chunks:
            # Convert to dictionary format
            chunk_dict = {
                "id": chunker.create_chunk_id(chunk.filepath, chunk.start_line, chunk.end_line, chunk.text),
                "filepath": chunk.filepath,
                "language": chunk.language,
                "node_type": chunk.node_type,
                "name": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
                "summary": chunk.summary,
                "imports": chunk.imports,
                "tokens_estimate": chunker.estimate_tokens(chunk.text),
                "last_modified": str(Path(chunk.filepath).stat().st_mtime) if Path(chunk.filepath).exists() else ""
            }
            all_chunks.append(chunk_dict)
    
    logger.info(f"Generated {len(all_chunks)} chunks from {len(py_files)} files")
    
    return all_chunks
