"""
RAG context retrieval module that fetches relevant chunks from CodexGLUE dataset.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGContextRetriever:
    """Retrieve contextual chunks from embedded knowledge base."""
    
    def __init__(self, rag_chroma_path: Optional[str] = None,
                 model_name: Optional[str] = None,
                 device: Optional[str] = None):
        # Load from environment if not provided
        self.rag_chroma_path = rag_chroma_path or os.getenv('RAG_CHROMA_PATH', "./rag_chroma")
        self.model_name = model_name or os.getenv('SENTENCE_TRANSFORMER_MODEL', "all-mpnet-base-v2")
        self.device = device or os.getenv('MODEL_DEVICE', 'cpu')

        # Initialize attributes
        self.client = None
        self.collection = None
        self.model = None
        
        self._setup_model()
        self._setup_rag_chroma()
    
    def _setup_model(self):
        """Initialize SentenceTransformer model."""
        try:
            # Configure device
            if self.device == 'cuda' and torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
            logger.info(f"Using device: {device}")

            # Load model with device setting
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded RAG embedding model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to lightweight model
            try:
                self.model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device='cpu'
                )
                logger.info("Using fallback model: all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise RuntimeError("Could not load any embedding model")
    
    def _setup_rag_chroma(self):
        """Setup ChromaDB connection to RAG knowledge base."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.rag_chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Try to get collection (should already exist if CodexGLUE is embedded)
            try:
                self.collection = self.client.get_collection("codexglue_chunks")
                logger.info("Connected to RAG ChromaDB (codexglue_chunks)")
            except Exception:
                logger.warning("RAG collection 'codexglue_chunks' not found. Creating empty collection.")
                self.collection = self.client.get_or_create_collection(
                    name="codexglue_chunks",
                    metadata={"description": "CodexGLUE dataset chunks"}
                )
        except Exception as e:
            logger.error(f"Failed to setup RAG ChromaDB: {e}")
            logger.info("Continuing without RAG context...")
            self.collection = None
    
    def retrieve_context(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: Query string or code
            n_results: Number of results to return
        
        Returns:
            List of relevant chunk dictionaries
        """
        # If codexglue_chunks is missing or empty, try falling back to the project's code_chunks collection
        try:
            if (not self.collection) or (hasattr(self.collection, 'count') and self.collection.count() == 0):
                logger.warning("RAG collection is empty or not available. Attempting to fallback to 'code_chunks' collection.")
                try:
                    # Try to use project collection where code embeddings are stored
                    fallback = self.client.get_collection('code_chunks')
                    logger.info("Falling back to 'code_chunks' collection for retrieval")
                    self.collection = fallback
                except Exception:
                    logger.warning("Fallback collection 'code_chunks' not found or empty. No RAG context available.")
                    return []
        except Exception:
            logger.warning("Unable to evaluate RAG collection state; proceeding cautiously.")
            return []
        
        # Check if collection is empty
        try:
            if self.collection.count() == 0:
                logger.warning("RAG collection is empty")
                return []
        except Exception as e:
            logger.warning(f"Could not check collection count: {e}")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_tensor=False)[0]
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            context_chunks = []
            for i in range(len(results['ids'][0])):
                context = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                context_chunks.append(context)
            
            logger.info(f"Retrieved {len(context_chunks)} context chunks")
            return context_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def enrich_chunk_with_context(self, chunk_text: str, n_context: int = 5) -> str:
        """
        Enrich a code chunk with relevant context from RAG knowledge base.
        
        Args:
            chunk_text: The code chunk to enrich
            n_context: Number of context chunks to include
        
        Returns:
            Enriched text with context
        """
        context_chunks = self.retrieve_context(chunk_text, n_results=n_context)
        
        if not context_chunks:
            return chunk_text
        
        # Build enriched text
        context_text = "\n\n## Related Context from CodexGLUE:\n\n"
        
        for i, ctx in enumerate(context_chunks, 1):
            context_text += f"### Context {i} (similarity: {ctx['similarity']:.3f})\n\n"
            context_text += f"```python\n{ctx['text'][:500]}\n```\n\n"
        
        enriched_text = context_text + "\n\n## Current Code:\n\n" + chunk_text
        
        return enriched_text


def retrieve_context_for_chunk(chunk: Dict[str, Any], rag_chroma_path: str = "./rag_chroma") -> List[Dict[str, Any]]:
    """
    Retrieve RAG context for a specific chunk.
    
    Args:
        chunk: Chunk dictionary
        rag_chroma_path: Path to RAG ChromaDB
    
    Returns:
        List of context chunks
    """
    retriever = RAGContextRetriever(rag_chroma_path=rag_chroma_path)
    return retriever.retrieve_context(chunk['text'])

