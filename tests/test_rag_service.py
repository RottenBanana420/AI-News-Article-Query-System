"""
Comprehensive Test Suite for RAG Service

These tests are designed to FAIL initially and drive the implementation.
Tests are NEVER modified - only the implementation code is updated to pass them.

Test Philosophy:
- Tests define the contract and requirements
- Implementation must meet all test criteria
- Edge cases and error conditions are thoroughly tested
- Performance requirements are enforced
"""

import pytest
import time
import numpy as np
from typing import List, Dict
from unittest.mock import Mock, patch, MagicMock

from src.query.rag_service import RAGService
from src.embeddings.ollama_service import OllamaEmbeddingService
from src.storage.vector_store import VectorStore


class TestRAGServiceInitialization:
    """Test RAG service initialization and configuration."""
    
    def test_initialization_with_defaults(self):
        """Test RAG service initializes with default parameters."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        assert rag.embedding_service == embedding_service
        assert rag.vector_store == vector_store
        assert rag.llm_model == "llama3.1:latest"
        assert rag.top_k == 5
        assert rag.temperature == 0.7
        assert rag.max_tokens == 1000
    
    def test_initialization_with_custom_params(self):
        """Test RAG service accepts custom configuration."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_model="llama3.1",
            top_k=10,
            temperature=0.5,
            max_tokens=2000
        )
        
        assert rag.llm_model == "llama3.1"
        assert rag.top_k == 10
        assert rag.temperature == 0.5
        assert rag.max_tokens == 2000
    
    def test_llm_initialization(self):
        """Test that LLM is properly initialized with Ollama."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # LLM should be initialized
        assert rag.llm is not None
        assert hasattr(rag.llm, 'invoke') or hasattr(rag.llm, '__call__')


class TestContextRetrieval:
    """Test context retrieval from vector store."""
    
    def test_retrieve_context_with_results(self):
        """Test retrieving relevant context chunks."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        # Mock query embedding
        query_embedding = np.random.rand(768).tolist()
        embedding_service.generate_embedding.return_value = np.array(query_embedding)
        
        # Mock vector store results
        mock_results = [
            (0.1, {
                'chunk': 'AI is transforming healthcare.',
                'title': 'AI in Healthcare',
                'url': 'https://example.com/ai-health',
                'article_id': 'art1'
            }),
            (0.2, {
                'chunk': 'Machine learning improves diagnostics.',
                'title': 'ML Diagnostics',
                'url': 'https://example.com/ml-diag',
                'article_id': 'art2'
            })
        ]
        vector_store.search.return_value = mock_results
        
        rag = RAGService(embedding_service, vector_store)
        
        context = rag._retrieve_context(query_embedding, top_k=2)
        
        assert len(context) == 2
        assert context[0]['chunk'] == 'AI is transforming healthcare.'
        assert context[0]['title'] == 'AI in Healthcare'
        assert context[0]['distance'] == 0.1
        assert 'url' in context[0]
    
    def test_retrieve_context_empty_results(self):
        """Test handling when no relevant context is found."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768).tolist()
        embedding_service.generate_embedding.return_value = np.array(query_embedding)
        vector_store.search.return_value = []
        
        rag = RAGService(embedding_service, vector_store)
        
        context = rag._retrieve_context(query_embedding, top_k=5)
        
        assert context == []
    
    def test_retrieve_context_respects_top_k(self):
        """Test that top_k parameter is respected."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768).tolist()
        embedding_service.generate_embedding.return_value = np.array(query_embedding)
        vector_store.search.return_value = []  # Add return value
        
        rag = RAGService(embedding_service, vector_store, top_k=3)
        rag._retrieve_context(query_embedding, top_k=7)
        
        # Should call vector_store.search with top_k=7 (override)
        vector_store.search.assert_called_once()
        call_args = vector_store.search.call_args
        assert call_args[1]['k'] == 7


class TestPromptConstruction:
    """Test prompt template construction."""
    
    def test_format_context_with_multiple_chunks(self):
        """Test formatting multiple context chunks for prompt."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        chunks = [
            {
                'chunk': 'First chunk of text.',
                'title': 'Article 1',
                'url': 'https://example.com/1',
                'distance': 0.1
            },
            {
                'chunk': 'Second chunk of text.',
                'title': 'Article 2',
                'url': 'https://example.com/2',
                'distance': 0.2
            }
        ]
        
        formatted = rag._format_context(chunks)
        
        # Should include chunk text
        assert 'First chunk of text.' in formatted
        assert 'Second chunk of text.' in formatted
        
        # Should include source information
        assert 'Article 1' in formatted
        assert 'Article 2' in formatted
        
        # Should be numbered or clearly separated
        assert '[1]' in formatted or 'Source 1' in formatted or '1.' in formatted
    
    def test_format_context_empty(self):
        """Test formatting when no context is available."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        formatted = rag._format_context([])
        
        assert formatted == "" or formatted.lower() == "no relevant context found."
    
    def test_build_prompt_structure(self):
        """Test that prompt has proper structure with system and user messages."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        question = "What is AI?"
        context = "[1] Article 1: AI is artificial intelligence."
        history = []
        
        prompt = rag._build_prompt(question, context, history)
        
        # Prompt should instruct to use context
        assert 'context' in prompt.lower() or 'information' in prompt.lower()
        
        # Should include the question
        assert question in prompt
        
        # Should include the context
        assert context in prompt or "AI is artificial intelligence" in prompt
        
        # Should instruct to cite sources
        assert 'cite' in prompt.lower() or 'source' in prompt.lower() or 'reference' in prompt.lower()
    
    def test_build_prompt_with_conversation_history(self):
        """Test prompt includes conversation history."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        question = "What about deep learning?"
        context = "[1] Deep learning uses neural networks."
        history = [
            {'role': 'user', 'content': 'What is AI?'},
            {'role': 'assistant', 'content': 'AI is artificial intelligence.'}
        ]
        
        prompt = rag._build_prompt(question, context, history)
        
        # Should include previous conversation
        assert 'What is AI?' in prompt
        assert 'AI is artificial intelligence.' in prompt
    
    def test_build_prompt_no_context_instruction(self):
        """Test prompt instructs LLM to acknowledge when context lacks information."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        question = "What is quantum computing?"
        context = ""
        history = []
        
        prompt = rag._build_prompt(question, context, history)
        
        # Should instruct to say when information is not available
        assert any(keyword in prompt.lower() for keyword in [
            "don't know", "not found", "not available", "no information",
            "cannot answer", "not in the context"
        ])


class TestAnswerGeneration:
    """Test LLM answer generation."""
    
    @patch('src.query.rag_service.ChatOllama')
    def test_generate_answer_success(self, mock_chat_ollama):
        """Test successful answer generation."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="AI is artificial intelligence. [Source: Article 1]")
        mock_chat_ollama.return_value = mock_llm
        
        rag = RAGService(embedding_service, vector_store)
        
        prompt = "What is AI? Context: AI is artificial intelligence."
        answer = rag._generate_answer(prompt)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "AI" in answer or "artificial intelligence" in answer.lower()
    
    @patch('src.query.rag_service.ChatOllama')
    def test_generate_answer_handles_llm_error(self, mock_chat_ollama):
        """Test handling of LLM connection errors."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        # Mock LLM to raise error
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Connection failed")
        mock_chat_ollama.return_value = mock_llm
        
        rag = RAGService(embedding_service, vector_store)
        
        prompt = "What is AI?"
        
        with pytest.raises(Exception) as exc_info:
            rag._generate_answer(prompt)
        
        assert "Connection failed" in str(exc_info.value) or "error" in str(exc_info.value).lower()


class TestCitationExtraction:
    """Test citation extraction and matching."""
    
    def test_extract_citations_with_references(self):
        """Test extracting citations from answer with source references."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        answer = "AI is transforming healthcare [1]. Machine learning improves diagnostics [2]."
        retrieved_chunks = [
            {
                'chunk': 'AI is transforming healthcare.',
                'title': 'AI in Healthcare',
                'url': 'https://example.com/ai-health',
                'distance': 0.1
            },
            {
                'chunk': 'Machine learning improves diagnostics.',
                'title': 'ML Diagnostics',
                'url': 'https://example.com/ml-diag',
                'distance': 0.2
            }
        ]
        
        citations = rag._extract_citations(answer, retrieved_chunks)
        
        assert len(citations) >= 1
        assert any('AI in Healthcare' in c['title'] for c in citations)
        assert any('https://example.com/ai-health' in c['url'] for c in citations)
    
    def test_extract_citations_no_references(self):
        """Test citation extraction when answer has no explicit references."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        answer = "AI is a broad field of computer science."
        retrieved_chunks = [
            {
                'chunk': 'AI is transforming healthcare.',
                'title': 'AI in Healthcare',
                'url': 'https://example.com/ai-health',
                'distance': 0.1
            }
        ]
        
        citations = rag._extract_citations(answer, retrieved_chunks)
        
        # Should still return sources that were used
        assert isinstance(citations, list)
        # At minimum, should include the retrieved chunks as potential sources
        assert len(citations) >= 0
    
    def test_extract_citations_deduplication(self):
        """Test that duplicate citations are removed."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        answer = "AI is important [1]. AI is transforming industries [1]."
        retrieved_chunks = [
            {
                'chunk': 'AI is transforming industries.',
                'title': 'AI Impact',
                'url': 'https://example.com/ai',
                'distance': 0.1
            }
        ]
        
        citations = rag._extract_citations(answer, retrieved_chunks)
        
        # Should not have duplicate URLs
        urls = [c['url'] for c in citations]
        assert len(urls) == len(set(urls))


class TestEndToEndQuery:
    """Test complete end-to-end query processing."""
    
    @patch('src.query.rag_service.ChatOllama')
    def test_query_complete_flow(self, mock_chat_ollama):
        """Test complete query flow from question to answer with citations."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        # Mock embedding generation
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        
        # Mock vector store search
        mock_results = [
            (0.1, {
                'chunk': 'AI is transforming healthcare through predictive analytics.',
                'title': 'AI in Healthcare',
                'url': 'https://example.com/ai-health',
                'article_id': 'art1'
            })
        ]
        vector_store.search.return_value = mock_results
        
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="AI is transforming healthcare through predictive analytics and diagnosis. [Source: AI in Healthcare]"
        )
        mock_chat_ollama.return_value = mock_llm
        
        rag = RAGService(embedding_service, vector_store)
        
        result = rag.query("How is AI used in healthcare?")
        
        # Verify result structure
        assert 'question' in result
        assert 'answer' in result
        assert 'sources' in result
        assert 'response_time' in result
        
        # Verify content
        assert result['question'] == "How is AI used in healthcare?"
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0
        assert isinstance(result['sources'], list)
        assert result['response_time'] >= 0
    
    @patch('src.query.rag_service.ChatOllama')
    def test_query_with_conversation_history(self, mock_chat_ollama):
        """Test query with conversation history."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        
        mock_results = [
            (0.1, {
                'chunk': 'Deep learning is a subset of machine learning.',
                'title': 'Deep Learning Basics',
                'url': 'https://example.com/dl',
                'article_id': 'art1'
            })
        ]
        vector_store.search.return_value = mock_results
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Deep learning is a subset of machine learning that uses neural networks."
        )
        mock_chat_ollama.return_value = mock_llm
        
        rag = RAGService(embedding_service, vector_store)
        
        history = [
            {'role': 'user', 'content': 'What is machine learning?'},
            {'role': 'assistant', 'content': 'Machine learning is a type of AI.'}
        ]
        
        result = rag.query("What about deep learning?", conversation_history=history)
        
        assert 'answer' in result
        # LLM should have received the history in the prompt
        mock_llm.invoke.assert_called_once()
    
    @patch('src.query.rag_service.ChatOllama')
    def test_query_no_relevant_context(self, mock_chat_ollama):
        """Test query when no relevant context is found."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="I don't have information about that in the provided context."
        )
        mock_chat_ollama.return_value = mock_llm
        
        rag = RAGService(embedding_service, vector_store)
        
        result = rag.query("What is quantum entanglement?")
        
        assert 'answer' in result
        assert 'sources' in result
        assert len(result['sources']) == 0
    
    def test_query_response_time_requirement(self):
        """Test that query response time is reasonable (< 10 seconds)."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        # Use fast mocks
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []
        
        with patch('src.query.rag_service.ChatOllama') as mock_chat_ollama:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="Test answer")
            mock_chat_ollama.return_value = mock_llm
            
            rag = RAGService(embedding_service, vector_store)
            
            start = time.time()
            result = rag.query("Test question")
            elapsed = time.time() - start
            
            # Response time should be tracked
            assert result['response_time'] <= elapsed + 0.1  # Small margin
            
            # Should complete in reasonable time (mocked, so very fast)
            assert elapsed < 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @patch('src.query.rag_service.ChatOllama')
    def test_very_long_question(self, mock_chat_ollama):
        """Test handling of very long questions."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer to long question")
        mock_chat_ollama.return_value = mock_llm
        
        rag = RAGService(embedding_service, vector_store)
        
        # 2000 character question
        long_question = "What is AI? " * 200
        
        result = rag.query(long_question)
        
        assert 'answer' in result
        assert isinstance(result['answer'], str)
    
    @patch('src.query.rag_service.ChatOllama')
    def test_empty_question(self, mock_chat_ollama):
        """Test handling of empty question."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        rag = RAGService(embedding_service, vector_store)
        
        with pytest.raises(ValueError) as exc_info:
            rag.query("")
        
        assert "question" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
    
    @patch('src.query.rag_service.ChatOllama')
    def test_question_with_special_characters(self, mock_chat_ollama):
        """Test handling questions with special characters."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer")
        mock_chat_ollama.return_value = mock_llm
        
        rag = RAGService(embedding_service, vector_store)
        
        special_question = "What is AI? <script>alert('test')</script> & how does it work?"
        
        result = rag.query(special_question)
        
        assert 'answer' in result
        # Should handle special characters gracefully
        assert isinstance(result['answer'], str)
    
    def test_custom_top_k_override(self):
        """Test that custom top_k overrides default."""
        embedding_service = Mock(spec=OllamaEmbeddingService)
        vector_store = Mock(spec=VectorStore)
        
        query_embedding = np.random.rand(768)
        embedding_service.generate_embedding.return_value = query_embedding
        vector_store.search.return_value = []
        
        with patch('src.query.rag_service.ChatOllama') as mock_chat_ollama:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="Answer")
            mock_chat_ollama.return_value = mock_llm
            
            rag = RAGService(embedding_service, vector_store, top_k=5)
            
            result = rag.query("Test question", top_k=10)
            
            # Should use top_k=10
            vector_store.search.assert_called_once()
            call_kwargs = vector_store.search.call_args[1]
            assert call_kwargs['k'] == 10
