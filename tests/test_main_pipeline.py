"""
Comprehensive Integration Tests for Main Pipeline System

Following TDD principles: These tests are designed to FAIL initially,
then we implement the code to make them pass.

Tests cover:
- System initialization
- End-to-end article ingestion
- Query pipeline
- Persistence (save/load)
- Statistics
- Error handling
- Performance
"""

import pytest
import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
from unittest.mock import Mock, patch, MagicMock

# Import the main pipeline (will fail initially - that's expected in TDD)
from src.main_pipeline import ArticleQuerySystem


class TestSystemInitialization:
    """Test system initialization and component setup."""
    
    def test_default_initialization(self):
        """Test system initializes with default components."""
        system = ArticleQuerySystem()
        
        assert system is not None
        assert system.extractor is not None
        assert system.embedding_service is not None
        assert system.vector_store is not None
        assert system.query_handler is not None
    
    def test_custom_component_injection(self):
        """Test dependency injection with custom components."""
        mock_extractor = Mock()
        mock_embedding = Mock()
        mock_vector_store = Mock()
        mock_query_handler = Mock()
        
        system = ArticleQuerySystem(
            extractor=mock_extractor,
            embedding_service=mock_embedding,
            vector_store=mock_vector_store,
            query_handler=mock_query_handler
        )
        
        assert system.extractor == mock_extractor
        assert system.embedding_service == mock_embedding
        assert system.vector_store == mock_vector_store
        assert system.query_handler == mock_query_handler
    
    def test_custom_directories(self):
        """Test initialization with custom storage directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            
            assert system.storage_dir == tmpdir
            assert system.index_path == os.path.join(tmpdir, "test.index")


class TestArticleIngestion:
    """Test end-to-end article ingestion pipeline."""
    
    @pytest.fixture
    def system(self):
        """Create a fresh system for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            yield system
    
    def test_ingest_single_article_success(self, system):
        """Test successful ingestion of a single article."""
        # Use a real, accessible URL
        url = "https://www.bbc.com/news/technology"
        
        result = system.ingest_article(url)
        
        assert result['success'] is True
        assert result['url'] == url
        assert 'article_id' in result
        assert result['chunks_created'] > 0
        assert result['embeddings_created'] > 0
        assert result['processing_time'] > 0
    
    def test_ingest_invalid_url(self, system):
        """Test ingestion fails gracefully with invalid URL."""
        url = "https://this-is-not-a-valid-url-12345.com/article"
        
        result = system.ingest_article(url)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['url'] == url
    
    def test_ingest_batch_articles(self, system):
        """Test batch ingestion of multiple articles."""
        urls = [
            "https://www.bbc.com/news/technology",
            "https://www.reuters.com/technology/",
        ]
        
        results = system.ingest_batch(urls, delay=1.0)
        
        assert 'total' in results
        assert 'successful' in results
        assert 'failed' in results
        assert 'processing_time' in results
        assert results['total'] == len(urls)
        assert results['successful'] + results['failed'] == results['total']
        assert len(results['details']) == len(urls)
    
    def test_ingest_from_file(self, system):
        """Test ingestion from a file containing URLs."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("https://www.bbc.com/news/technology\n")
            f.write("https://www.reuters.com/technology/\n")
            f.write("# This is a comment\n")
            f.write("\n")  # Empty line
            temp_file = f.name
        
        try:
            results = system.ingest_from_file(temp_file, delay=1.0)
            
            assert results['total'] == 2  # Should skip comment and empty line
            assert 'successful' in results
            assert 'failed' in results
        finally:
            os.unlink(temp_file)
    
    def test_ingest_duplicate_article(self, system):
        """Test ingesting the same article twice."""
        url = "https://www.bbc.com/news/technology"
        
        result1 = system.ingest_article(url)
        result2 = system.ingest_article(url)
        
        # Should handle duplicates gracefully
        assert result1['success'] is True
        assert result2['success'] is True
        # Second ingestion should be faster (potentially cached)
        assert 'article_id' in result2
    
    def test_progress_tracking(self, system):
        """Test that progress tracking works for batch operations."""
        urls = [
            "https://www.bbc.com/news/technology",
            "https://www.reuters.com/technology/",
        ]
        
        # Progress should be tracked (via tqdm)
        results = system.ingest_batch(urls, show_progress=True)
        
        assert results is not None
        assert 'total' in results


class TestQueryPipeline:
    """Test query processing and retrieval."""
    
    @pytest.fixture
    def system_with_data(self):
        """Create a system with pre-ingested articles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            
            # Ingest a test article
            url = "https://www.bbc.com/news/technology"
            system.ingest_article(url)
            
            yield system
    
    def test_semantic_search(self, system_with_data):
        """Test semantic search returns relevant results."""
        query = "artificial intelligence technology"
        
        results = system_with_data.query(query, top_k=5)
        
        assert len(results) > 0
        assert len(results) <= 5
        for result in results:
            assert 'chunk' in result
            assert 'metadata' in result
            assert 'distance' in result
            assert 'similarity' in result
    
    def test_ask_question(self, system_with_data):
        """Test RAG-based question answering."""
        question = "What is this article about?"
        
        result = system_with_data.ask_question(question, top_k=3)
        
        assert 'question' in result
        assert 'answer' in result
        assert 'sources' in result
        assert 'response_time' in result
        assert len(result['answer']) > 0
        assert len(result['sources']) > 0
    
    def test_multi_turn_conversation(self, system_with_data):
        """Test conversation continuity across multiple questions."""
        # First question
        result1 = system_with_data.ask_question(
            "What is the main topic?",
            top_k=3
        )
        session_id = result1.get('session_id')
        
        # Follow-up question
        result2 = system_with_data.ask_question(
            "Tell me more about that",
            session_id=session_id,
            top_k=3
        )
        
        assert result2['session_id'] == session_id
        assert len(result2['answer']) > 0
    
    def test_query_empty_index(self):
        """Test querying an empty index returns empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            
            results = system.query("test query")
            
            assert len(results) == 0


class TestPersistence:
    """Test save and load functionality."""
    
    @pytest.fixture
    def system_with_data(self):
        """Create a system with pre-ingested articles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index"),
                saved_states_dir=os.path.join(tmpdir, "states")
            )
            
            # Ingest a test article
            url = "https://www.bbc.com/news/technology"
            system.ingest_article(url)
            
            yield system
    
    def test_save_state(self, system_with_data):
        """Test saving system state."""
        state_name = "test_state"
        description = "Test state for unit testing"
        
        result = system_with_data.save_state(state_name, description=description)
        
        assert result['success'] is True
        assert result['state_name'] == state_name
        assert 'path' in result
        assert os.path.exists(result['path'])
    
    def test_load_state(self, system_with_data):
        """Test loading a saved state."""
        # Save state
        state_name = "test_state"
        system_with_data.save_state(state_name)
        
        # Get stats before
        stats_before = system_with_data.get_stats()
        
        # Create new system and load state
        with tempfile.TemporaryDirectory() as tmpdir:
            new_system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "new.index"),
                saved_states_dir=system_with_data.saved_states_dir
            )
            
            result = new_system.load_state(state_name)
            
            assert result['success'] is True
            assert result['state_name'] == state_name
            
            # Stats should match
            stats_after = new_system.get_stats()
            assert stats_after['total_chunks'] == stats_before['total_chunks']
    
    def test_list_saved_states(self, system_with_data):
        """Test listing all saved states."""
        # Save multiple states
        system_with_data.save_state("state1", description="First state")
        system_with_data.save_state("state2", description="Second state")
        
        states = system_with_data.list_states()
        
        assert len(states) >= 2
        state_names = [s['name'] for s in states]
        assert "state1" in state_names
        assert "state2" in state_names
        
        for state in states:
            assert 'name' in state
            assert 'description' in state
            assert 'created_at' in state
            assert 'size_bytes' in state
    
    def test_load_nonexistent_state(self, system_with_data):
        """Test loading a non-existent state fails gracefully."""
        result = system_with_data.load_state("nonexistent_state")
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_overwrite_protection(self, system_with_data):
        """Test that saving over existing state requires confirmation."""
        state_name = "test_state"
        
        # Save first time
        result1 = system_with_data.save_state(state_name)
        assert result1['success'] is True
        
        # Save again without overwrite flag should fail
        result2 = system_with_data.save_state(state_name, overwrite=False)
        assert result2['success'] is False
        
        # Save with overwrite flag should succeed
        result3 = system_with_data.save_state(state_name, overwrite=True)
        assert result3['success'] is True


class TestStatistics:
    """Test system statistics and metrics."""
    
    @pytest.fixture
    def system_with_data(self):
        """Create a system with pre-ingested articles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            
            # Ingest test articles
            urls = [
                "https://www.bbc.com/news/technology",
            ]
            system.ingest_batch(urls)
            
            yield system
    
    def test_get_stats(self, system_with_data):
        """Test retrieving system statistics."""
        stats = system_with_data.get_stats()
        
        assert 'total_articles' in stats
        assert 'total_chunks' in stats
        assert 'total_embeddings' in stats
        assert 'vector_store_stats' in stats
        assert 'cache_stats' in stats
        
        assert stats['total_articles'] > 0
        assert stats['total_chunks'] > 0
        assert stats['total_embeddings'] > 0
    
    def test_stats_accuracy(self, system_with_data):
        """Test that statistics are accurate."""
        stats = system_with_data.get_stats()
        
        # Chunks and embeddings should match
        assert stats['total_chunks'] == stats['total_embeddings']
        
        # Vector store count should match total embeddings
        assert stats['vector_store_stats']['count'] == stats['total_embeddings']


class TestErrorHandling:
    """Test error handling throughout the pipeline."""
    
    def test_network_error_handling(self):
        """Test handling of network errors during ingestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            
            # Use an invalid URL
            result = system.ingest_article("https://invalid-url-12345.com")
            
            assert result['success'] is False
            assert 'error' in result
    
    def test_malformed_url_handling(self):
        """Test handling of malformed URLs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            
            result = system.ingest_article("not-a-url")
            
            assert result['success'] is False
            assert 'error' in result
    
    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            
            results = system.query("")
            
            # Should return empty results, not crash
            assert isinstance(results, list)


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def system(self):
        """Create a fresh system for performance tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index")
            )
            yield system
    
    def test_ingestion_performance(self, system):
        """Test article ingestion completes in reasonable time."""
        url = "https://www.bbc.com/news/technology"
        
        start_time = time.time()
        result = system.ingest_article(url)
        elapsed = time.time() - start_time
        
        # Should complete within 30 seconds
        assert elapsed < 30
        assert result['processing_time'] < 30
    
    def test_query_performance(self, system):
        """Test query response time is sub-second."""
        # Ingest an article first
        system.ingest_article("https://www.bbc.com/news/technology")
        
        # Measure query time
        start_time = time.time()
        results = system.query("technology", top_k=5)
        elapsed = time.time() - start_time
        
        # Should be very fast (sub-second)
        assert elapsed < 1.0
    
    def test_save_load_performance(self, system):
        """Test save/load operations are fast."""
        # Ingest data
        system.ingest_article("https://www.bbc.com/news/technology")
        
        # Test save performance
        start_time = time.time()
        system.save_state("perf_test")
        save_time = time.time() - start_time
        
        assert save_time < 5.0  # Should save within 5 seconds
        
        # Test load performance
        start_time = time.time()
        system.load_state("perf_test")
        load_time = time.time() - start_time
        
        assert load_time < 5.0  # Should load within 5 seconds


class TestIntegrationWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_complete_workflow(self):
        """Test a complete workflow from ingestion to query to persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Initialize system
            system = ArticleQuerySystem(
                storage_dir=tmpdir,
                index_path=os.path.join(tmpdir, "test.index"),
                saved_states_dir=os.path.join(tmpdir, "states")
            )
            
            # 2. Ingest articles
            urls = ["https://www.bbc.com/news/technology"]
            ingest_results = system.ingest_batch(urls)
            assert ingest_results['successful'] > 0
            
            # 3. Query the system
            query_results = system.query("technology")
            assert len(query_results) > 0
            
            # 4. Ask a question
            qa_result = system.ask_question("What is this about?")
            assert len(qa_result['answer']) > 0
            
            # 5. Check statistics
            stats = system.get_stats()
            assert stats['total_articles'] > 0
            
            # 6. Save state
            save_result = system.save_state("workflow_test")
            assert save_result['success'] is True
            
            # 7. Create new system and load state
            new_system = ArticleQuerySystem(
                storage_dir=os.path.join(tmpdir, "new"),
                index_path=os.path.join(tmpdir, "new.index"),
                saved_states_dir=os.path.join(tmpdir, "states")
            )
            load_result = new_system.load_state("workflow_test")
            assert load_result['success'] is True
            
            # 8. Verify loaded state works
            new_stats = new_system.get_stats()
            assert new_stats['total_articles'] == stats['total_articles']
            
            new_query_results = new_system.query("technology")
            assert len(new_query_results) > 0
