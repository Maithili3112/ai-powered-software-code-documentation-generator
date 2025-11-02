"""
Documentation reassembly module that reconstructs generated documentation
back into the same structure and order as the original files.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocReassembler:
    """Reassembles documentation chunks back into file-level documentation."""
    
    def __init__(self, output_dir: str = "./generated_docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def reassemble_by_file(self, chunks: List[Dict[str, Any]], 
                          docs: List[str]) -> Dict[str, str]:
        """
        Reassemble documentation by grouping chunks by file.
        
        Args:
            chunks: List of original chunk dictionaries
            docs: List of generated documentation strings
        
        Returns:
            Dictionary mapping file paths to combined documentation
        """
        file_to_chunks = defaultdict(lambda: {'chunks': [], 'docs': []})
        
        # Group chunks by file
        for i, chunk in enumerate(chunks):
            filepath = chunk['filepath']
            file_to_chunks[filepath]['chunks'].append(chunk)
            if i < len(docs):
                file_to_chunks[filepath]['docs'].append(docs[i])
        
        # Reassemble documentation for each file
        file_docs = {}
        for filepath, data in file_to_chunks.items():
            # Sort chunks by start_line to maintain order
            chunk_doc_pairs = list(zip(data['chunks'], data['docs']))
            chunk_doc_pairs.sort(key=lambda x: x[0]['start_line'])
            
            # Combine documentation
            combined_doc = self._combine_chunk_docs(filepath, chunk_doc_pairs)
            file_docs[filepath] = combined_doc
        
        return file_docs
    
    def _combine_chunk_docs(self, filepath: str, 
                           chunk_doc_pairs: List[tuple]) -> str:
        """
        Combine documentation from multiple chunks.
        
        Args:
            filepath: File path
            chunk_doc_pairs: List of (chunk, doc) tuples
        
        Returns:
            Combined documentation
        """
        output = [f"# File: {filepath}\n"]
        
        for i, (chunk, doc) in enumerate(chunk_doc_pairs, 1):
            output.append(f"\n## Chunk {i} (lines {chunk['start_line']}-{chunk['end_line']})\n\n")
            output.append(doc)
            output.append("\n---\n")
        
        return "\n".join(output)
    
    def save_file_docs(self, file_docs: Dict[str, str], preserve_structure: bool = True):
        """
        Save reassembled documentation to disk.
        
        Args:
            file_docs: Dictionary of file paths to documentation
            preserve_structure: Whether to preserve directory structure
        """
        for filepath, doc in file_docs.items():
            if preserve_structure:
                # Create directory structure
                rel_path = Path(filepath)
                output_path = self.output_dir / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Flatten structure
                safe_name = Path(filepath).name.replace('.py', '.md')
                output_path = self.output_dir / safe_name
            
            # Write documentation
            output_path.with_suffix('.md').write_text(doc, encoding='utf-8')
            logger.info(f"Saved documentation to {output_path}")
    
    def create_index(self, file_docs: Dict[str, str]) -> str:
        """
        Create an index page for all documentation.
        
        Args:
            file_docs: Dictionary of file paths to documentation
        
        Returns:
            Index page content
        """
        index_content = ["# Documentation Index\n"]
        
        # Group by directory
        dir_files = defaultdict(list)
        for filepath in sorted(file_docs.keys()):
            path = Path(filepath)
            dir_files[str(path.parent)].append(path.name)
        
        for directory in sorted(dir_files.keys()):
            index_content.append(f"\n## {directory or 'Root'}\n")
            for filename in sorted(dir_files[directory]):
                safe_name = filename.replace('.py', '.md')
                index_content.append(f"- [{filename}]({directory}/{safe_name})")
        
        return "\n".join(index_content)


def reassemble_docs(chunks: List[Dict[str, Any]], 
                   docs: List[str], 
                   output_dir: str = "./generated_docs",
                   preserve_structure: bool = True) -> Dict[str, str]:
    """
    Reassemble documentation from chunks and save to disk.
    
    Args:
        chunks: List of chunk dictionaries
        docs: List of generated documentation
        output_dir: Output directory
        preserve_structure: Whether to preserve directory structure
    
    Returns:
        Dictionary of file paths to documentation
    """
    reassembler = DocReassembler(output_dir)
    
    # Reassemble by file
    file_docs = reassembler.reassemble_by_file(chunks, docs)
    
    # Save to disk
    reassembler.save_file_docs(file_docs, preserve_structure)
    
    # Create index
    index = reassembler.create_index(file_docs)
    (reassembler.output_dir / "index.md").write_text(index)
    
    return file_docs

