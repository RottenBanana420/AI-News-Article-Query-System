"""
Main Pipeline System

Orchestrates all components into a cohesive system for article ingestion,
embedding generation, vector storage, and intelligent querying.

This is the central integration point that coordinates:
- Article extraction
- Embedding generation
- Vector storage
- Query processing
- State persistence
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

from .ingestion.article_extractor import ArticleExtractor
from .embeddings.ollama_service import OllamaEmbeddingService
from .storage.vector_store import VectorStore
from .query.handler import QueryHandler

load_dotenv()


class ArticleQuerySystem:
    """
    Main pipeline system that integrates all components.
    
    Provides high-level methods for:
    - Article ingestion (single, batch, from file)
    - Semantic search and RAG-based Q&A
    - State persistence (save/load)
    - System statistics
    """
    
    def __init__(
        self,
        extractor: Optional[ArticleExtractor] = None,
        embedding_service: Optional[OllamaEmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        query_handler: Optional[QueryHandler] = None,
        storage_dir: Optional[str] = None,
        index_path: Optional[str] = None,
        saved_states_dir: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the article query system.
        
        Args:
            extractor: ArticleExtractor instance (or None for default)
            embedding_service: OllamaEmbeddingService instance (or None for default)
            vector_store: VectorStore instance (or None for default)
            query_handler: QueryHandler instance (or None for default)
            storage_dir: Directory for storing extracted articles
            index_path: Path to FAISS index file
            saved_states_dir: Directory for saved states
            log_level: Logging level
        """
        # Setup logging
        self._setup_logging(log_level)
        
        # Store configuration
        self.storage_dir = storage_dir or os.getenv('ARTICLE_CACHE_DIR', 'data/raw_articles')
        self.index_path = index_path or os.getenv('FAISS_INDEX_PATH', 'data/embeddings/articles.index')
        self.saved_states_dir = saved_states_dir or os.getenv('SAVED_STATES_DIR', 'data/saved_states')
        
        # Create directories
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(self.saved_states_dir, exist_ok=True)
        
        # Initialize components (dependency injection or defaults)
        self.extractor = extractor or ArticleExtractor(storage_dir=self.storage_dir)
        self.embedding_service = embedding_service or OllamaEmbeddingService()
        self.vector_store = vector_store or VectorStore(index_path=self.index_path)
        self.query_handler = query_handler or QueryHandler(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store
        )
        
        # Track ingested articles
        self.article_index = {}
        self._load_article_index()
        
        self.logger.info("ArticleQuerySystem initialized successfully")
    
    def _setup_logging(self, log_level: int):
        """Configure logging for the system."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _load_article_index(self):
        """Load the article index from disk."""
        index_file = os.path.join(self.storage_dir, 'article_index.json')
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    self.article_index = json.load(f)
                self.logger.info(f"Loaded article index with {len(self.article_index)} articles")
            except Exception as e:
                self.logger.warning(f"Failed to load article index: {e}")
                self.article_index = {}
        else:
            self.article_index = {}
    
    def _save_article_index(self):
        """Save the article index to disk."""
        index_file = os.path.join(self.storage_dir, 'article_index.json')
        try:
            with open(index_file, 'w') as f:
                json.dump(self.article_index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save article index: {e}")
    
    def ingest_article(self, url: str) -> Dict[str, Any]:
        """
        Ingest a single article: extract → chunk → embed → store.
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with ingestion results:
                - success: bool
                - url: str
                - article_id: str (if successful)
                - chunks_created: int
                - embeddings_created: int
                - processing_time: float
                - error: str (if failed)
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Ingesting article: {url}")
            
            # Step 1: Extract article
            article_data = self.extractor.extract_article(url)
            if not article_data:
                return {
                    'success': False,
                    'url': url,
                    'error': 'Failed to extract article'
                }
            
            # Step 2: Process article (chunk and embed)
            result = self.embedding_service.process_article(
                article_data['text'],
                use_cache=True,
                show_progress=False
            )
            
            if not result['embeddings']:
                return {
                    'success': False,
                    'url': url,
                    'error': 'Failed to generate embeddings'
                }
            
            # Step 3: Store embeddings in vector database
            metadata_list = []
            for i, chunk in enumerate(result['chunks']):
                metadata_list.append({
                    'chunk': chunk,
                    'chunk_index': i,
                    'title': article_data.get('title', 'Unknown'),
                    'url': url,
                    'authors': article_data.get('authors', []),
                    'publish_date': article_data.get('publish_date', 'Unknown'),
                    'article_id': article_data.get('url_hash', url)
                })
            
            self.vector_store.add_embeddings(result['embeddings'], metadata_list)
            
            # Step 4: Update article index
            article_id = article_data.get('url_hash', url)
            self.article_index[article_id] = {
                'url': url,
                'title': article_data.get('title', 'Unknown'),
                'chunks': len(result['chunks']),
                'ingested_at': datetime.now().isoformat()
            }
            self._save_article_index()
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Successfully ingested article: {article_data.get('title', url)} "
                f"({len(result['chunks'])} chunks in {processing_time:.2f}s)"
            )
            
            return {
                'success': True,
                'url': url,
                'article_id': article_id,
                'chunks_created': len(result['chunks']),
                'embeddings_created': len(result['embeddings']),
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting article {url}: {e}")
            return {
                'success': False,
                'url': url,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def ingest_batch(
        self,
        urls: List[str],
        delay: float = 1.0,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest multiple articles in batch.
        
        Args:
            urls: List of article URLs
            delay: Delay between requests (seconds)
            show_progress: Show progress bar
            
        Returns:
            Dictionary with batch results:
                - total: int
                - successful: int
                - failed: int
                - processing_time: float
                - details: List of individual results
        """
        start_time = time.time()
        results = []
        
        iterator = tqdm(urls, desc="Ingesting articles") if show_progress else urls
        
        for url in iterator:
            result = self.ingest_article(url)
            results.append(result)
            
            # Delay between requests (except for last one)
            if url != urls[-1]:
                time.sleep(delay)
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        return {
            'total': len(urls),
            'successful': successful,
            'failed': failed,
            'processing_time': time.time() - start_time,
            'details': results
        }
    
    def ingest_from_file(
        self,
        file_path: str,
        delay: float = 1.0,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest articles from a file containing URLs (one per line).
        
        Args:
            file_path: Path to file with URLs
            delay: Delay between requests (seconds)
            show_progress: Show progress bar
            
        Returns:
            Dictionary with batch results
        """
        # Read URLs from file
        urls = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    urls.append(line)
        
        self.logger.info(f"Loaded {len(urls)} URLs from {file_path}")
        
        return self.ingest_batch(urls, delay=delay, show_progress=show_progress)
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search over indexed articles.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            
        Returns:
            List of search results with chunks and metadata
        """
        if not query_text:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query_text)
            
            # Search vector store
            results = self.vector_store.search(query_embedding.tolist(), k=top_k)
            
            # Format results
            formatted_results = []
            for distance, metadata in results:
                formatted_results.append({
                    'chunk': metadata.get('chunk', ''),
                    'metadata': {
                        'title': metadata.get('title', 'Unknown'),
                        'url': metadata.get('url', ''),
                        'authors': metadata.get('authors', []),
                        'publish_date': metadata.get('publish_date', 'Unknown')
                    },
                    'distance': distance,
                    'similarity': 1 / (1 + distance)
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error querying: {e}")
            return []
    
    def ask_question(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Ask a question and get an AI-generated answer with citations.
        
        Args:
            question: User's question
            session_id: Optional session ID for conversation continuity
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dictionary with question, answer, sources, and metadata
        """
        try:
            result = self.query_handler.ask_question(
                question=question,
                session_id=session_id,
                top_k=top_k,
                include_sources=True
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'sources': [],
                'session_id': session_id,
                'response_time': 0
            }
    
    def save_state(
        self,
        state_name: str,
        description: str = "",
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """
        Save the current system state (vector store + metadata).
        
        Args:
            state_name: Name for this saved state
            description: Optional description
            overwrite: Allow overwriting existing state
            
        Returns:
            Dictionary with save results
        """
        try:
            # Create state directory
            state_dir = os.path.join(self.saved_states_dir, state_name)
            
            # Check if state exists
            if os.path.exists(state_dir) and not overwrite:
                return {
                    'success': False,
                    'state_name': state_name,
                    'error': 'State already exists. Use overwrite=True to replace.'
                }
            
            os.makedirs(state_dir, exist_ok=True)
            
            # Save vector store
            index_path = os.path.join(state_dir, 'vector_store.index')
            self.vector_store.save_index(index_path)
            
            # Save article index
            article_index_path = os.path.join(state_dir, 'article_index.json')
            with open(article_index_path, 'w') as f:
                json.dump(self.article_index, f, indent=2)
            
            # Save metadata
            metadata = {
                'name': state_name,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'total_articles': len(self.article_index),
                'total_chunks': self.vector_store.count()
            }
            
            metadata_path = os.path.join(state_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved state: {state_name}")
            
            return {
                'success': True,
                'state_name': state_name,
                'path': state_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return {
                'success': False,
                'state_name': state_name,
                'error': str(e)
            }
    
    def load_state(self, state_name: str) -> Dict[str, Any]:
        """
        Load a previously saved state.
        
        Args:
            state_name: Name of the state to load
            
        Returns:
            Dictionary with load results
        """
        try:
            state_dir = os.path.join(self.saved_states_dir, state_name)
            
            if not os.path.exists(state_dir):
                return {
                    'success': False,
                    'state_name': state_name,
                    'error': 'State not found'
                }
            
            # Load vector store
            index_path = os.path.join(state_dir, 'vector_store.index')
            if not self.vector_store.load_index(index_path):
                return {
                    'success': False,
                    'state_name': state_name,
                    'error': 'Failed to load vector store'
                }
            
            # Load article index
            article_index_path = os.path.join(state_dir, 'article_index.json')
            with open(article_index_path, 'r') as f:
                self.article_index = json.load(f)
            
            self.logger.info(f"Loaded state: {state_name}")
            
            return {
                'success': True,
                'state_name': state_name
            }
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return {
                'success': False,
                'state_name': state_name,
                'error': str(e)
            }
    
    def list_states(self) -> List[Dict[str, Any]]:
        """
        List all saved states.
        
        Returns:
            List of state metadata dictionaries
        """
        states = []
        
        if not os.path.exists(self.saved_states_dir):
            return states
        
        for state_name in os.listdir(self.saved_states_dir):
            state_dir = os.path.join(self.saved_states_dir, state_name)
            
            if not os.path.isdir(state_dir):
                continue
            
            metadata_path = os.path.join(state_dir, 'metadata.json')
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Add size information
                total_size = sum(
                    os.path.getsize(os.path.join(state_dir, f))
                    for f in os.listdir(state_dir)
                    if os.path.isfile(os.path.join(state_dir, f))
                )
                metadata['size_bytes'] = total_size
                
                states.append(metadata)
                
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for state {state_name}: {e}")
        
        # Sort by creation date (newest first)
        states.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return states
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with statistics
        """
        vector_stats = self.vector_store.get_stats()
        cache_stats = self.embedding_service.get_cache_stats()
        
        total_vectors = vector_stats.get('total_vectors', 0)
        
        return {
            'total_articles': len(self.article_index),
            'total_chunks': total_vectors,
            'total_embeddings': total_vectors,
            'vector_store_stats': {
                **vector_stats,
                'count': total_vectors  # Add count for backward compatibility
            },
            'cache_stats': cache_stats
        }
