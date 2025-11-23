"""
Vector Store

Manages FAISS vector database for storing and retrieving article embeddings.
"""

import os
import pickle
from typing import List, Dict, Tuple, Optional
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """Manages FAISS vector database for article embeddings."""
    
    def __init__(self, index_path: Optional[str] = None, dimension: int = 4096):
        """
        Initialize the vector store.
        
        Args:
            index_path: Path to save/load FAISS index (default: from .env)
            dimension: Dimension of embedding vectors
        """
        self.index_path = index_path or os.getenv('FAISS_INDEX_PATH', 'data/embeddings/articles.index')
        self.dimension = dimension
        self.index = None
        self.metadata = []  # Store metadata for each vector
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Try to load existing index
        self.load_index()
        
        # Create new index if none exists
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, any]]
    ) -> None:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries (one per embedding)
        """
        if not embeddings:
            return
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Update dimension if this is the first addition
        if self.index.ntotal == 0 and vectors.shape[1] != self.dimension:
            self.dimension = vectors.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add to index
        self.index.add(vectors)
        
        # Store metadata
        self.metadata.extend(metadata)
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Tuple[float, Dict[str, any]]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (distance, metadata) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Convert to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Combine results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append((float(dist), self.metadata[idx]))
        
        return results
    
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            path: Path to save index (default: self.index_path)
        """
        save_path = path or self.index_path
        
        # Save FAISS index
        faiss.write_index(self.index, save_path)
        
        # Save metadata
        metadata_path = save_path + '.metadata'
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Index saved to {save_path}")
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            path: Path to load index from (default: self.index_path)
            
        Returns:
            True if successful, False otherwise
        """
        load_path = path or self.index_path
        
        try:
            # Load FAISS index
            if os.path.exists(load_path):
                self.index = faiss.read_index(load_path)
                
                # Load metadata
                metadata_path = load_path + '.metadata'
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)
                
                print(f"Index loaded from {load_path}")
                return True
        except Exception as e:
            print(f"Error loading index: {str(e)}")
        
        return False
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats
        """
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata)
        }
