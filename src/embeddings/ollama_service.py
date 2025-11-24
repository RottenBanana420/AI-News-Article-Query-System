"""
Ollama Embedding Service

A robust service for generating text embeddings using Ollama's local LLM models.
Provides direct API integration with advanced features including:
- Connection verification
- Text chunking with configurable overlap
- Batch processing with progress tracking
- Hybrid caching (in-memory + optional disk persistence)
- Comprehensive error handling
"""

import os
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance tracking."""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    cache_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return self.hits / self.total_requests if self.total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'hit_rate': self.hit_rate
        }


@dataclass
class EmbeddingMetadata:
    """Metadata for cached embeddings."""
    text_hash: str
    timestamp: str
    model: str
    dimensions: int


class OllamaConnectionError(Exception):
    """Raised when unable to connect to Ollama service."""
    pass


class OllamaModelError(Exception):
    """Raised when specified model is not available."""
    pass


class EmbeddingDimensionError(Exception):
    """Raised when embedding dimensions don't match expected value."""
    pass


class OllamaEmbeddingService:
    """
    Service for generating embeddings using Ollama's local LLM models.
    
    Features:
    - Direct Ollama API integration
    - Connection verification
    - Text chunking with overlap
    - Batch processing with progress tracking
    - Hybrid caching (memory + optional disk)
    - Comprehensive error handling
    """
    
    EXPECTED_DIMENSIONS = 768  # nomic-embed-text produces 768-dimensional embeddings
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 10,
        enable_disk_cache: bool = True,
        cache_dir: Optional[str] = None,
        verify_dimensions: bool = True,
        timeout: int = 30
    ):
        """
        Initialize the Ollama embedding service.
        
        Args:
            model: Ollama model name (default: nomic-embed-text)
            base_url: Ollama base URL (default: from .env or http://localhost:11434)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks to preserve context
            batch_size: Number of texts to process in each batch
            enable_disk_cache: Enable persistent disk caching
            cache_dir: Directory for cache files (default: data/embeddings/cache)
            verify_dimensions: Verify embedding dimensions match expected value
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.verify_dimensions = verify_dimensions
        self.timeout = timeout
        
        # Initialize caching
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._cache_metadata: Dict[str, EmbeddingMetadata] = {}
        self._cache_stats = CacheStats()
        self.enable_disk_cache = enable_disk_cache
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            project_root = Path(__file__).parent.parent.parent
            self.cache_dir = project_root / 'data' / 'embeddings' / 'cache'
        
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()
        
        logger.info(f"Initialized OllamaEmbeddingService with model: {self.model}")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        logger.info(f"Batch size: {self.batch_size}, Disk cache: {self.enable_disk_cache}")
    
    def _compute_hash(self, text: str) -> str:
        """
        Compute SHA-256 hash of text for caching.
        
        Args:
            text: Input text
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _load_disk_cache(self) -> None:
        """Load cached embeddings from disk."""
        if not self.cache_dir.exists():
            return
        
        cache_index_path = self.cache_dir / 'cache_index.json'
        if not cache_index_path.exists():
            logger.info("No disk cache found, starting fresh")
            return
        
        try:
            with open(cache_index_path, 'r') as f:
                cache_index = json.load(f)
            
            loaded_count = 0
            for text_hash, metadata_dict in cache_index.items():
                embedding_path = self.cache_dir / f"{text_hash}.npy"
                if embedding_path.exists():
                    embedding = np.load(embedding_path)
                    self._memory_cache[text_hash] = embedding
                    self._cache_metadata[text_hash] = EmbeddingMetadata(**metadata_dict)
                    loaded_count += 1
            
            self._cache_stats.cache_size = loaded_count
            logger.info(f"Loaded {loaded_count} embeddings from disk cache")
        
        except Exception as e:
            logger.error(f"Error loading disk cache: {e}")
    
    def _save_to_disk_cache(self, text_hash: str, embedding: np.ndarray) -> None:
        """
        Save embedding to disk cache.
        
        Args:
            text_hash: Hash of the text
            embedding: Embedding vector
        """
        if not self.enable_disk_cache:
            return
        
        try:
            # Save embedding as numpy file
            embedding_path = self.cache_dir / f"{text_hash}.npy"
            np.save(embedding_path, embedding)
            
            # Update cache index
            cache_index_path = self.cache_dir / 'cache_index.json'
            cache_index = {}
            
            if cache_index_path.exists():
                with open(cache_index_path, 'r') as f:
                    cache_index = json.load(f)
            
            cache_index[text_hash] = asdict(self._cache_metadata[text_hash])
            
            with open(cache_index_path, 'w') as f:
                json.dump(cache_index, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving to disk cache: {e}")
    
    def verify_connection(self) -> bool:
        """
        Verify connection to Ollama service.
        
        Returns:
            True if connection successful
            
        Raises:
            OllamaConnectionError: If unable to connect
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("✓ Successfully connected to Ollama service")
            return True
        
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(
                f"Unable to connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running (try: ollama serve)"
            )
        except requests.exceptions.Timeout:
            raise OllamaConnectionError(
                f"Connection to Ollama timed out after {self.timeout}s"
            )
        except Exception as e:
            error_str = str(e).lower()
            if 'timeout' in error_str or 'timed out' in error_str:
                raise OllamaConnectionError(
                    f"Connection to Ollama timed out after {self.timeout}s"
                )
            raise OllamaConnectionError(f"Error connecting to Ollama: {str(e)}")
    
    def verify_model_available(self) -> bool:
        """
        Verify that the specified model is available.
        
        Returns:
            True if model is available
            
        Raises:
            OllamaModelError: If model is not available
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            models = response.json().get('models', [])
            available_models = [m['name'] for m in models]
            
            # Check for exact match or match with :latest suffix
            model_found = self.model in available_models or f"{self.model}:latest" in available_models
            
            if not model_found:
                raise OllamaModelError(
                    f"Model '{self.model}' not found. Available models: {available_models}. "
                    f"Try: ollama pull {self.model}"
                )
            
            logger.info(f"✓ Model '{self.model}' is available")
            return True
        
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Error checking model availability: {str(e)}")
    
    def _verify_embedding_dimensions(self, embedding: np.ndarray) -> None:
        """
        Verify embedding dimensions match expected value.
        
        Args:
            embedding: Embedding vector
            
        Raises:
            EmbeddingDimensionError: If dimensions don't match
        """
        if not self.verify_dimensions:
            return
        
        actual_dims = len(embedding)
        if actual_dims != self.EXPECTED_DIMENSIONS:
            raise EmbeddingDimensionError(
                f"Expected {self.EXPECTED_DIMENSIONS} dimensions, got {actual_dims}. "
                f"This may indicate an issue with the model or API."
            )
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position, accounting for overlap
            if chunk_overlap > 0:
                start = end - chunk_overlap
            else:
                start = end
            
            # Prevent infinite loop if overlap >= chunk_size
            if chunk_overlap >= chunk_size and len(chunks) > 1:
                break
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            OllamaConnectionError: If unable to connect to Ollama
            EmbeddingDimensionError: If dimensions don't match expected value
        """
        # Update stats
        self._cache_stats.total_requests += 1
        
        # Check cache
        text_hash = self._compute_hash(text)
        if use_cache and text_hash in self._memory_cache:
            self._cache_stats.hits += 1
            logger.debug(f"Cache hit for text hash: {text_hash[:8]}...")
            return self._memory_cache[text_hash]
        
        self._cache_stats.misses += 1
        
        # Generate embedding via API
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            embedding_list = response.json()['embedding']
            embedding = np.array(embedding_list, dtype=np.float32)
            
            # Verify dimensions
            self._verify_embedding_dimensions(embedding)
            
            # Cache the result
            if use_cache:
                self._memory_cache[text_hash] = embedding
                self._cache_metadata[text_hash] = EmbeddingMetadata(
                    text_hash=text_hash,
                    timestamp=datetime.now().isoformat(),
                    model=self.model,
                    dimensions=len(embedding)
                )
                self._cache_stats.cache_size = len(self._memory_cache)
                
                # Save to disk if enabled
                self._save_to_disk_cache(text_hash, embedding)
            
            return embedding
        
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Unable to connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running."
            )
        except requests.exceptions.Timeout:
            raise OllamaConnectionError(
                f"Request timed out after {self.timeout}s"
            )
        except requests.exceptions.InvalidURL as e:
            raise OllamaConnectionError(
                f"Invalid Ollama URL: {self.base_url}. Error: {str(e)}"
            )
        except requests.exceptions.HTTPError as e:
            # Handle server errors (like 500 for very long text)
            if e.response.status_code == 500:
                raise RuntimeError(
                    f"Ollama server error (500). The text may be too long or the model may be overloaded. "
                    f"Try reducing chunk size or text length."
                )
            raise RuntimeError(f"HTTP error from Ollama: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Unexpected API response format: missing {e}")
        except Exception as e:
            # Check if it's a timeout-related error
            error_str = str(e).lower()
            if 'timeout' in error_str or 'timed out' in error_str:
                raise OllamaConnectionError(f"Request timed out after {self.timeout}s")
            raise RuntimeError(f"Error generating embedding: {str(e)}")
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batch processing.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use caching
            show_progress: Whether to log progress
            progress_callback: Optional callback function(current, total)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        total = len(texts)
        
        logger.info(f"Processing {total} texts in batches of {self.batch_size}")
        start_time = time.time()
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.generate_embedding(text, use_cache=use_cache)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Progress tracking
            current = min(i + self.batch_size, total)
            if show_progress:
                logger.info(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
            
            if progress_callback:
                progress_callback(current, total)
        
        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        logger.info(f"Completed {total} embeddings in {elapsed:.2f}s ({rate:.2f} embeddings/s)")
        
        return embeddings
    
    def process_article(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_cache: bool = True,
        show_progress: bool = False
    ) -> Dict[str, any]:
        """
        Process an article: chunk text and generate embeddings.
        
        Args:
            text: Article text
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            use_cache: Whether to use caching
            show_progress: Whether to show progress
            
        Returns:
            Dictionary with chunks and embeddings
        """
        if not text:
            return {
                'chunks': [],
                'embeddings': [],
                'num_chunks': 0,
                'total_characters': 0
            }
        
        # Chunk the text
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        logger.info(f"Processing article with {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(
            chunks,
            use_cache=use_cache,
            show_progress=show_progress
        )
        
        result = {
            'chunks': chunks,
            'embeddings': embeddings,
            'num_chunks': len(chunks),
            'total_characters': len(text)
        }
        
        return result
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self._cache_stats.to_dict()
    
    def clear_cache(self, clear_disk: bool = False) -> None:
        """
        Clear the embedding cache.
        
        Args:
            clear_disk: Whether to also clear disk cache
        """
        self._memory_cache.clear()
        self._cache_metadata.clear()
        self._cache_stats = CacheStats()
        
        if clear_disk and self.enable_disk_cache:
            try:
                # Remove all cache files
                for cache_file in self.cache_dir.glob('*.npy'):
                    cache_file.unlink()
                
                cache_index_path = self.cache_dir / 'cache_index.json'
                if cache_index_path.exists():
                    cache_index_path.unlink()
                
                logger.info("Cleared disk cache")
            except Exception as e:
                logger.error(f"Error clearing disk cache: {e}")
        
        logger.info("Cleared memory cache")
