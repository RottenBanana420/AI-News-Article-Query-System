"""
Integration Tests for RAG Query System

Tests the complete end-to-end RAG pipeline with all components working together.
These tests verify that the system can handle real-world scenarios.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.query.handler import QueryHandler
from src.embeddings.ollama_service import OllamaEmbeddingService
from src.storage.vector_store import VectorStore


class TestRAGIntegration:
    """Test complete RAG pipeline integration."""
    
    @patch('src.query.rag_service.ChatOllama')
    def test_single_turn_question_answering(self, mock_chat_ollama):
        """Test asking a single question and getting an answer with citations."""
        # Setup mocks
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        # Mock embedding generation
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        
        # Mock vector store search results
        mock_results = [
            (0.1, {
                'chunk': 'Artificial intelligence is transforming healthcare through predictive analytics.',
                'title': 'AI in Healthcare',
                'url': 'https://example.com/ai-health',
                'article_id': 'art1'
            }),
            (0.2, {
                'chunk': 'Machine learning algorithms can detect diseases earlier than traditional methods.',
                'title': 'ML for Disease Detection',
                'url': 'https://example.com/ml-disease',
                'article_id': 'art2'
            })
        ]
        vector_store.search.return_value = mock_results
        
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="AI is transforming healthcare through predictive analytics and early disease detection [1][2]."
        )
        mock_chat_ollama.return_value = mock_llm
        
        # Create handler
        handler = QueryHandler(
            embedding_service=embedding_service,
            vector_store=vector_store,
            enable_rag=True,
            enable_conversation=True
        )
        
        # Ask question
        result = handler.ask_question("How is AI used in healthcare?")
        
        # Verify result structure
        assert 'question' in result
        assert 'answer' in result
        assert 'sources' in result
        assert 'session_id' in result
        assert 'response_time' in result
        
        # Verify content
        assert result['question'] == "How is AI used in healthcare?"
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0
        assert isinstance(result['sources'], list)
        assert len(result['sources']) > 0
        assert result['session_id'] is not None
        assert result['response_time'] >= 0
    
    @patch('src.query.rag_service.ChatOllama')
    def test_multi_turn_conversation(self, mock_chat_ollama):
        """Test multi-turn conversation with context retention."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        
        # Different results for different queries
        def search_side_effect(embedding, k):
            return [
                (0.1, {
                    'chunk': 'Test content',
                    'title': 'Test Article',
                    'url': 'https://example.com/test',
                    'article_id': 'art1'
                })
            ]
        
        vector_store.search.side_effect = search_side_effect
        
        # Mock LLM responses
        mock_llm = MagicMock()
        responses = [
            MagicMock(content="Machine learning is a subset of AI."),
            MagicMock(content="Deep learning uses neural networks, which are part of machine learning.")
        ]
        mock_llm.invoke.side_effect = responses
        mock_chat_ollama.return_value = mock_llm
        
        handler = QueryHandler(
            embedding_service=embedding_service,
            vector_store=vector_store,
            enable_rag=True,
            enable_conversation=True
        )
        
        # First question
        result1 = handler.ask_question("What is machine learning?")
        session_id = result1['session_id']
        
        assert result1['answer'] == "Machine learning is a subset of AI."
        
        # Follow-up question using same session
        result2 = handler.ask_question(
            "What about deep learning?",
            session_id=session_id
        )
        
        assert result2['session_id'] == session_id
        assert "neural networks" in result2['answer']
        
        # Verify conversation history was used
        assert mock_llm.invoke.call_count == 2
        
        # Second call should have history in prompt
        second_call_prompt = mock_llm.invoke.call_args_list[1][0][0]
        assert "What is machine learning?" in second_call_prompt
    
    @patch('src.query.rag_service.ChatOllama')
    def test_no_relevant_context_handling(self, mock_chat_ollama):
        """Test handling when no relevant context is found."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []  # No results
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="I don't have information about that in the provided context."
        )
        mock_chat_ollama.return_value = mock_llm
        
        handler = QueryHandler(
            embedding_service=embedding_service,
            vector_store=vector_store,
            enable_rag=True
        )
        
        result = handler.ask_question("What is quantum entanglement?")
        
        assert 'answer' in result
        assert 'sources' in result
        assert len(result['sources']) == 0
        assert "don't have information" in result['answer'].lower()
    
    @patch('src.query.rag_service.ChatOllama')
    def test_citation_accuracy(self, mock_chat_ollama):
        """Test that citations match the retrieved sources."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        
        mock_results = [
            (0.1, {
                'chunk': 'Climate change is affecting global temperatures.',
                'title': 'Climate Change Report',
                'url': 'https://example.com/climate',
                'article_id': 'art1'
            }),
            (0.2, {
                'chunk': 'Renewable energy is key to reducing emissions.',
                'title': 'Renewable Energy Guide',
                'url': 'https://example.com/renewable',
                'article_id': 'art2'
            })
        ]
        vector_store.search.return_value = mock_results
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Climate change affects temperatures [1] and renewable energy helps [2]."
        )
        mock_chat_ollama.return_value = mock_llm
        
        handler = QueryHandler(
            embedding_service=embedding_service,
            vector_store=vector_store,
            enable_rag=True
        )
        
        result = handler.ask_question("What about climate change?")
        
        # Verify citations
        assert len(result['sources']) >= 1
        source_titles = [s['title'] for s in result['sources']]
        assert 'Climate Change Report' in source_titles
    
    @patch('src.query.rag_service.ChatOllama')
    def test_response_time_tracking(self, mock_chat_ollama):
        """Test that response time is tracked correctly."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test answer")
        mock_chat_ollama.return_value = mock_llm
        
        handler = QueryHandler(
            embedding_service=embedding_service,
            vector_store=vector_store,
            enable_rag=True
        )
        
        result = handler.ask_question("Test question")
        
        assert 'response_time' in result
        assert result['response_time'] >= 0
        assert result['response_time'] < 10  # Should be fast with mocks
    
    def test_rag_disabled_error(self):
        """Test that asking questions without RAG enabled raises error."""
        handler = QueryHandler(enable_rag=False)
        
        with pytest.raises(ValueError) as exc_info:
            handler.ask_question("Test question")
        
        assert "RAG is not enabled" in str(exc_info.value)
    
    @patch('src.query.rag_service.ChatOllama')
    def test_conversation_persistence(self, mock_chat_ollama):
        """Test that conversation history persists across questions."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test answer")
        mock_chat_ollama.return_value = mock_llm
        
        handler = QueryHandler(
            embedding_service=embedding_service,
            vector_store=vector_store,
            enable_rag=True,
            enable_conversation=True
        )
        
        # Ask multiple questions in same session
        result1 = handler.ask_question("Question 1")
        session_id = result1['session_id']
        
        result2 = handler.ask_question("Question 2", session_id=session_id)
        result3 = handler.ask_question("Question 3", session_id=session_id)
        
        # Verify session ID is maintained
        assert result2['session_id'] == session_id
        assert result3['session_id'] == session_id
        
        # Verify conversation history exists
        history = handler.conversation_manager.get_history(session_id)
        assert len(history) == 6  # 3 questions + 3 answers
    
    @patch('src.query.rag_service.ChatOllama')
    def test_custom_top_k_parameter(self, mock_chat_ollama):
        """Test that custom top_k parameter is respected."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test answer")
        mock_chat_ollama.return_value = mock_llm
        
        handler = QueryHandler(
            embedding_service=embedding_service,
            vector_store=vector_store,
            enable_rag=True
        )
        
        # Ask question with custom top_k
        handler.ask_question("Test question", top_k=10)
        
        # Verify vector_store.search was called with k=10
        vector_store.search.assert_called()
        call_kwargs = vector_store.search.call_args[1]
        assert call_kwargs['k'] == 10


class TestBackwardCompatibility:
    """Test that legacy methods still work."""
    
    def test_legacy_query_method(self):
        """Test that legacy query() method still works."""
        # This test verifies backward compatibility
        embedding_generator = Mock()
        vector_store = Mock()
        
        embedding_generator.generate_query_embedding.return_value = np.random.rand(768).tolist()
        vector_store.search.return_value = [
            (0.1, {
                'chunk': 'Test',
                'title': 'Test Article',
                'url': 'https://example.com',
                'authors': [],
                'publish_date': '2024-01-01'
            })
        ]
        
        handler = QueryHandler(
            embedding_generator=embedding_generator,
            vector_store=vector_store,
            enable_rag=False  # Disable RAG for legacy test
        )
        
        results = handler.query("test query")
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'distance' in results[0]
        assert 'similarity' in results[0]
        assert 'chunk' in results[0]
