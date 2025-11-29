"""
Query Handler

Processes queries and retrieves relevant articles from the vector database.
Enhanced with RAG capabilities for intelligent question answering.
"""

from typing import List, Dict, Optional, Any
from ..embeddings.generator import EmbeddingGenerator
from ..embeddings.ollama_service import OllamaEmbeddingService
from ..storage.vector_store import VectorStore
from .rag_service import RAGService
from .conversation_manager import ConversationManager


class QueryHandler:
    """
    Handles query processing and article retrieval with RAG capabilities.
    
    Provides both traditional semantic search and AI-powered question answering
    with context retrieval and citation generation.
    """
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        embedding_service: Optional[OllamaEmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        enable_rag: bool = True,
        enable_conversation: bool = True
    ):
        """
        Initialize the query handler.
        
        Args:
            embedding_generator: EmbeddingGenerator instance (legacy)
            embedding_service: OllamaEmbeddingService instance (for RAG)
            vector_store: VectorStore instance
            enable_rag: Enable RAG-based question answering
            enable_conversation: Enable conversation history management
        """
        # Legacy support
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # RAG components
        self.embedding_service = embedding_service or OllamaEmbeddingService()
        self.vector_store = vector_store or VectorStore()
        
        # Initialize RAG service if enabled
        self.enable_rag = enable_rag
        if self.enable_rag:
            self.rag_service = RAGService(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store
            )
        
        # Initialize conversation manager if enabled
        self.enable_conversation = enable_conversation
        if self.enable_conversation:
            self.conversation_manager = ConversationManager()
    
    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for relevant articles (legacy method).
        
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
    
    def answer_query(self, query_text: str, k: int = 3) -> Dict[str, Any]:
        """
        Answer a query with context and sources (legacy method).
        
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
    
    def ask_question(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question and get an AI-generated answer with citations.
        
        This method uses RAG (Retrieval-Augmented Generation) to:
        1. Retrieve relevant article chunks
        2. Generate a context-aware answer using LLM
        3. Extract and return source citations
        4. Maintain conversation history for follow-up questions
        
        Args:
            question: User's question
            session_id: Optional conversation session ID for multi-turn dialogue
            top_k: Number of context chunks to retrieve
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary with:
                - question: Original question
                - answer: AI-generated answer
                - sources: List of cited sources (if include_sources=True)
                - session_id: Session ID for conversation continuity
                - response_time: Time taken to generate response
                
        Raises:
            ValueError: If RAG is not enabled
        """
        if not self.enable_rag:
            raise ValueError("RAG is not enabled. Initialize with enable_rag=True")
        
        # Get conversation history if session exists
        conversation_history = None
        if session_id and self.enable_conversation:
            conversation_history = self.conversation_manager.get_history(session_id)
        
        # Generate answer using RAG
        result = self.rag_service.query(
            question=question,
            conversation_history=conversation_history,
            top_k=top_k
        )
        
        # Create or use existing session
        if self.enable_conversation:
            if not session_id:
                session_id = self.conversation_manager.create_session()
            
            # Add this turn to conversation history
            self.conversation_manager.add_turn(
                session_id=session_id,
                question=question,
                answer=result['answer']
            )
        
        # Build response
        response = {
            'question': result['question'],
            'answer': result['answer'],
            'session_id': session_id,
            'response_time': result['response_time']
        }
        
        if include_sources:
            response['sources'] = result['sources']
        
        return response
