"""
Vector Store with FAISS HNSW Indexing

High-performance vector store using FAISS HNSW (Hierarchical Navigable Small World)
for fast approximate nearest neighbor search on article embeddings.

Optimized for 768-dimensional embeddings with sub-100ms query performance.
"""

import os
import pickle
from typing import List, Dict, Tuple, Optional
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """
    Production-ready vector store using FAISS HNSW indexing.
    
    Features:
    - HNSW graph-based approximate nearest neighbor search
    - Optimized for 768-dimensional embeddings (standard sentence transformers)
    - Sub-100ms query performance for thousands of vectors
    - Robust metadata management with synchronization guarantees
    - Atomic save/load operations with integrity checks
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        dimension: int = 768,
        M: int = 32,
        efConstruction: int = 200,
        efSearch: int = 128
    ):
        """
        Initialize the HNSW vector store.
        
        Args:
            index_path: Path to save/load FAISS index (default: from .env)
            dimension: Dimension of embedding vectors (default: 768 for sentence transformers)
            M: Number of connections per node in HNSW graph (16-48, default: 32)
                Higher M = better recall, more memory. 32 is optimal for 768-dim.
            efConstruction: Search depth during index construction (default: 200)
                Higher = better quality graph, slower build. 200 is production-ready.
            efSearch: Search depth during queries (default: 128)
                Higher = better recall, slower search. Tunable at runtime.
        """
        self.index_path = index_path or os.getenv(
            'FAISS_INDEX_PATH',
            'data/embeddings/articles.index'
        )
        self.dimension = dimension
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        
        # Metadata storage - synchronized with vector index
        self.metadata: List[Dict] = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize HNSW index
        self.index = None
        self._initialize_index()
        
        # Try to load existing index
        if os.path.exists(self.index_path):
            self.load_index()
    
    def _initialize_index(self) -> None:
        """Initialize a new HNSW index with optimized parameters."""
        # Create HNSW index with L2 distance
        self.index = faiss.IndexHNSWFlat(self.dimension, self.M)
        
        # Set construction parameters
        self.index.hnsw.efConstruction = self.efConstruction
        
        # Set search parameters
        self.index.hnsw.efSearch = self.efSearch
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict]
    ) -> None:
        """
        Add embeddings to the vector store with associated metadata.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries (one per embedding)
        
        Raises:
            ValueError: If embeddings and metadata counts don't match
            ValueError: If embedding dimensions don't match index dimension
        """
        # Handle empty input
        if not embeddings:
            return
        
        # Validate input
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) must match "
                f"metadata count ({len(metadata)})"
            )
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Validate dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({vectors.shape[1]}) must match "
                f"index dimension ({self.dimension})"
            )
        
        # Add to index in batches to prevent memory issues with HNSW
        # HNSW construction can be memory-intensive for large batches
        batch_size = 1000
        num_vectors = len(vectors)
        
        for i in range(0, num_vectors, batch_size):
            end_idx = min(i + batch_size, num_vectors)
            batch = vectors[i:end_idx]
            self.index.add(batch)
        
        # Store metadata (synchronized with index)
        self.metadata.extend(metadata)
        
        # Verify synchronization
        assert self.index.ntotal == len(self.metadata), \
            "CRITICAL: Metadata out of sync with index"
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        efSearch: Optional[int] = None
    ) -> List[Tuple[float, Dict]]:
        """
        Search for similar embeddings using HNSW approximate nearest neighbor.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            efSearch: Override default efSearch for this query (optional)
        
        Returns:
            List of (similarity_score, metadata) tuples, ordered by similarity
            Similarity scores are L2 distances (lower = more similar)
        
        Raises:
            ValueError: If query dimension doesn't match index dimension
            ValueError: If k is negative
        """
        # Handle edge cases
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        
        if k == 0:
            return []
        
        if self.index.ntotal == 0:
            return []
        
        # Validate query dimension
        query_vector = np.array([query_embedding], dtype=np.float32)
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension ({query_vector.shape[1]}) must match "
                f"index dimension ({self.dimension})"
            )
        
        # Set efSearch if provided
        if efSearch is not None:
            original_efSearch = self.index.hnsw.efSearch
            self.index.hnsw.efSearch = efSearch
        
        try:
            # Perform search
            actual_k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, actual_k)
            
            # Combine results with metadata
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata) and idx >= 0:
                    # L2 distance (lower is better)
                    results.append((float(dist), self.metadata[idx]))
            
            return results
        
        finally:
            # Restore original efSearch if it was changed
            if efSearch is not None:
                self.index.hnsw.efSearch = original_efSearch
    
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save FAISS index and metadata to disk with atomic write.
        
        Args:
            path: Path to save index (default: self.index_path)
        """
        save_path = path or self.index_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, save_path)
        
        # Save metadata with atomic write
        metadata_path = save_path + '.metadata'
        temp_metadata_path = metadata_path + '.tmp'
        
        try:
            with open(temp_metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            os.replace(temp_metadata_path, metadata_path)
        
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_metadata_path):
                os.remove(temp_metadata_path)
            raise e
    
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
            # Check if index file exists
            if not os.path.exists(load_path):
                return False
            
            # Load FAISS index
            loaded_index = faiss.read_index(load_path)
            
            # Verify it's an HNSW index
            if not isinstance(loaded_index, faiss.IndexHNSWFlat):
                raise ValueError(
                    f"Loaded index is not IndexHNSWFlat, got {type(loaded_index)}"
                )
            
            # Load metadata
            metadata_path = load_path + '.metadata'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    loaded_metadata = pickle.load(f)
            else:
                loaded_metadata = []
            
            # Verify synchronization
            if loaded_index.ntotal != len(loaded_metadata):
                raise ValueError(
                    f"Index has {loaded_index.ntotal} vectors but "
                    f"metadata has {len(loaded_metadata)} entries"
                )
            
            # Update instance variables
            self.index = loaded_index
            self.metadata = loaded_metadata
            self.dimension = self.index.d
            
            # Restore search parameters
            self.index.hnsw.efSearch = self.efSearch
            
            return True
        
        except Exception as e:
            # On error, reinitialize empty index
            self._initialize_index()
            self.metadata = []
            return False
    
    def clear(self) -> None:
        """
        Clear all vectors and metadata, resetting to empty state.
        """
        self._initialize_index()
        self.metadata = []
    
    def count(self) -> int:
        """
        Get the total number of vectors in the index.
        
        Returns:
            Number of vectors
        """
        return self.index.ntotal
    
    def get_dimension(self) -> int:
        """
        Get the dimension of vectors in the index.
        
        Returns:
            Vector dimension
        """
        return self.dimension
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata),
            'M': self.M,
            'efConstruction': self.efConstruction,
            'efSearch': self.index.hnsw.efSearch,
            'index_type': 'IndexHNSWFlat'
        }
    
    def set_efSearch(self, efSearch: int) -> None:
        """
        Update efSearch parameter for runtime query tuning.
        
        Args:
            efSearch: New efSearch value (higher = better recall, slower)
        """
        self.efSearch = efSearch
        self.index.hnsw.efSearch = efSearch
    
    def __repr__(self) -> str:
        """String representation of the vector store."""
        return (
            f"VectorStore(vectors={self.count()}, "
            f"dimension={self.dimension}, "
            f"M={self.M}, "
            f"efSearch={self.index.hnsw.efSearch})"
        )
