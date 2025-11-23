"""
Query Handler

Processes queries and retrieves relevant articles from the vector database.
"""

from typing import List, Dict, Optional
from ..embeddings.generator import EmbeddingGenerator
from ..storage.vector_store import VectorStore


class QueryHandler:
    """Handles query processing and article retrieval."""
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize the query handler.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
    
    def query(self, query_text: str, k: int = 5) -> List[Dict[str, any]]:
        """
        Query the vector database for relevant articles.
        
        Args:
            query_text: Query string
            k: Number of results to return
            
        Returns:
            List of result dictionaries with distance and metadata
        """
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_query_embedding(query_text)
        
        if not query_embedding:
            return []
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)
        
        # Format results
        formatted_results = []
        for distance, metadata in results:
            formatted_results.append({
                'distance': distance,
                'similarity': 1 / (1 + distance),  # Convert distance to similarity score
                'chunk': metadata.get('chunk', ''),
                'title': metadata.get('title', 'Unknown'),
                'url': metadata.get('url', ''),
                'authors': metadata.get('authors', []),
                'publish_date': metadata.get('publish_date', 'Unknown')
            })
        
        return formatted_results
    
    def get_context(self, query_text: str, k: int = 3) -> str:
        """
        Get context for a query by retrieving relevant chunks.
        
        Args:
            query_text: Query string
            k: Number of chunks to retrieve
            
        Returns:
            Combined context string
        """
        results = self.query(query_text, k=k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source {i}: {result['title']}]")
            context_parts.append(result['chunk'])
            context_parts.append("")  # Empty line between sources
        
        return "\n".join(context_parts)
    
    def answer_query(self, query_text: str, k: int = 3) -> Dict[str, any]:
        """
        Answer a query with context and sources.
        
        Args:
            query_text: Query string
            k: Number of sources to use
            
        Returns:
            Dictionary with context and sources
        """
        results = self.query(query_text, k=k)
        context = self.get_context(query_text, k=k)
        
        return {
            'query': query_text,
            'context': context,
            'sources': [
                {
                    'title': r['title'],
                    'url': r['url'],
                    'similarity': r['similarity']
                }
                for r in results
            ]
        }
