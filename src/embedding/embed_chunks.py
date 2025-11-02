"""
Embedding module that generates vector embeddings for code chunks using SentenceTransformers
and stores them in ChromaDB.
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


class ChunkEmbedder:
    """Embeds code chunks and stores them in ChromaDB."""
    
    def __init__(self, chroma_path: Optional[str] = None,
                 model_name: Optional[str] = None,
                 device: Optional[str] = None):
        # Load configuration from environment
        self.chroma_path = chroma_path or os.getenv('CHROMA_PATH', './chroma_store')
        self.model_name = model_name or os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-mpnet-base-v2')
        self.device = device or os.getenv('MODEL_DEVICE', 'cpu')
        
        # Initialize components
        self.client = None
        self.collection = None
        self.model = None
        
        # Load model and setup ChromaDB
        self._setup_model()
        self._setup_chroma()
    
    def _setup_model(self):
        """Initialize SentenceTransformer model."""
        try:
            # Configure device 
            if self.device == 'cuda' and torch.cuda.is_available():
                device = 'cuda'
                logger.info("Using CUDA for embeddings")
            else:
                device = 'cpu'
                logger.info("Using CPU for embeddings")

            # First try loading specified model
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded embedding model: {self.model_name}")

        except Exception as primary_error:
            logger.error(f"Failed to load primary model {self.model_name}: {primary_error}")
            
            try:
                # Fallback to lightweight model
                logger.info("Attempting to load fallback model...")
                self.model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device='cpu'
                )
                logger.info("Using fallback model: all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise RuntimeError("Could not load any embedding model")
    
    def _setup_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name="code_chunks",
                metadata={"description": "Code repository chunks with embeddings"}
            )
            logger.info(f"Connected to ChromaDB at {self.chroma_path}")
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Generate embeddings for chunks and store in ChromaDB."""
        if not chunks:
            logger.warning("No chunks to embed")
            return
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False
        )
        
        # Prepare data for ChromaDB
        ids = [chunk['id'] for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {
                'filepath': chunk['filepath'],
                'language': chunk['language'],
                'node_type': chunk['node_type'],
                'name': chunk.get('name', ''),
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line'],
                'summary': chunk['summary'],
                'tokens_estimate': chunk.get('tokens_estimate', 0)
            }
            metadatas.append(metadata)
        
        # Store in ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings.tolist()
            )
            logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB")
        except Exception as e:
            logger.error(f"Failed to store chunks in ChromaDB: {e}")
            raise
    
    def retrieve_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar chunks based on query."""
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=False)[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": "code_chunks",
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


def embed_project_chunks(chunks: List[Dict[str, Any]], chroma_path: str = "./chroma_store") -> None:
    """
    Embed all chunks from a project into ChromaDB.
    
    Args:
        chunks: List of chunk dictionaries
        chroma_path: Path to ChromaDB storage
    """
    embedder = ChunkEmbedder(chroma_path=chroma_path)
    embedder.embed_chunks(chunks)
    
    # Print stats
    stats = embedder.get_collection_stats()
    logger.info(f"Collection stats: {stats}")
