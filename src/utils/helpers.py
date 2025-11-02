"""
Helper utilities for the documentation generation pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_file: str):
    """
    Save chunks to a JSONL file.
    
    Args:
        chunks: List of chunk dictionaries
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    logger.info(f"Saved {len(chunks)} chunks to {output_file}")


def load_chunks_from_jsonl(input_file: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSONL file.
    
    Args:
        input_file: Input file path
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks


def group_chunks_by_file(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group chunks by file path.
    
    Args:
        chunks: List of chunk dictionaries
    
    Returns:
        Dictionary mapping file paths to chunk lists
    """
    grouped = {}
    
    for chunk in chunks:
        filepath = chunk['filepath']
        if filepath not in grouped:
            grouped[filepath] = []
        grouped[filepath].append(chunk)
    
    # Sort chunks by start_line
    for filepath in grouped:
        grouped[filepath].sort(key=lambda x: x['start_line'])
    
    return grouped


def compute_file_hash(filepath: str) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        filepath: File path
    
    Returns:
        Hexadecimal hash string
    """
    with open(filepath, 'rb') as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()


def ensure_directory(path: str):
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_relative_path(abs_path: str, base_path: str) -> str:
    """
    Get relative path from base.
    
    Args:
        abs_path: Absolute path
        base_path: Base path
    
    Returns:
        Relative path
    """
    return str(Path(abs_path).relative_to(Path(base_path)))


def extract_project_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract statistics from processed chunks.
    
    Args:
        chunks: List of chunk dictionaries
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'total_chunks': len(chunks),
        'files_processed': len(set(ch['filepath'] for ch in chunks)),
        'total_lines': sum(ch['end_line'] - ch['start_line'] + 1 for ch in chunks),
        'languages': set(ch['language'] for ch in chunks),
        'node_types': {}
    }
    
    # Count node types
    for chunk in chunks:
        node_type = chunk.get('node_type', 'unknown')
        stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
    
    stats['languages'] = list(stats['languages'])
    
    return stats

