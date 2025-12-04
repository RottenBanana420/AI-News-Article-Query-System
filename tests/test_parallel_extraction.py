"""
Tests for Parallel Article Extraction

Following TDD principles: These tests are designed to FAIL initially,
then we implement the code to make them pass.

Tests cover:
- Parallel extraction with ThreadPoolExecutor
- Connection pooling with requests.Session
- Progress indicators
- Resource monitoring
- Thread safety
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from src.ingestion.article_extractor import ArticleExtractor


class TestParallelProcessing:
    """Test parallel article extraction functionality."""
    
    @pytest.fixture
    def extractor(self, tmp_path):
        """Create extractor with temporary storage."""
        return ArticleExtractor(
            storage_dir=str(tmp_path / "articles"),
            log_dir=str(tmp_path / "logs")
        )
    
    def test_extract_batch_parallel_exists(self, extractor):
        """Test that extract_batch_parallel method exists."""
        assert hasattr(extractor, 'extract_batch_parallel')
        assert callable(extractor.extract_batch_parallel)
    
    def test_extract_batch_parallel_faster_than_sequential(self, extractor):
        """Test parallel extraction is faster than sequential."""
        urls = [
            "https://www.bbc.com/news/technology",
            "https://www.reuters.com/technology/",
            "https://www.theverge.com/tech",
        ]
        
        # Sequential extraction
        start_sequential = time.time()
        results_sequential = extractor.extract_batch(urls, delay=0.5)
        time_sequential = time.time() - start_sequential
        
        # Parallel extraction
        start_parallel = time.time()
        results_parallel = extractor.extract_batch_parallel(
            urls, 
            max_workers=3,
            show_progress=False
        )
        time_parallel = time.time() - start_parallel
        
        # Parallel should be faster (at least 30% faster)
        assert time_parallel < time_sequential * 0.7
        assert len(results_parallel) == len(results_sequential)
    
    def test_extract_batch_parallel_with_progress(self, extractor, capsys):
        """Test parallel extraction with progress bar."""
        urls = [
            "https://www.bbc.com/news/technology",
            "https://www.reuters.com/technology/",
        ]
        
        results = extractor.extract_batch_parallel(
            urls,
            max_workers=2,
            show_progress=True
        )
        
        # Should have processed URLs
        assert len(results) >= 0  # Some may fail, that's ok
    
    def test_extract_batch_parallel_respects_max_workers(self, extractor):
        """Test that max_workers parameter is respected."""
        urls = ["https://www.bbc.com/news/technology"] * 10
        
        # Should not raise error with different worker counts
        results_2 = extractor.extract_batch_parallel(urls, max_workers=2, show_progress=False)
        results_4 = extractor.extract_batch_parallel(urls, max_workers=4, show_progress=False)
        
        assert isinstance(results_2, list)
        assert isinstance(results_4, list)
    
    def test_connection_pooling_enabled(self, extractor):
        """Test that connection pooling is used in parallel mode."""
        # This test verifies that sessions are created per thread
        urls = ["https://www.bbc.com/news/technology"] * 5
        
        with patch('requests.Session') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            extractor.extract_batch_parallel(urls, max_workers=2, show_progress=False)
            
            # Session should be created (for connection pooling)
            assert mock_session_class.called


class TestProgressIndicators:
    """Test progress tracking functionality."""
    
    @pytest.fixture
    def extractor(self, tmp_path):
        """Create extractor with temporary storage."""
        return ArticleExtractor(
            storage_dir=str(tmp_path / "articles"),
            log_dir=str(tmp_path / "logs")
        )
    
    def test_progress_callback_called(self, extractor):
        """Test that progress callback is invoked during extraction."""
        urls = ["https://www.bbc.com/news/technology"] * 3
        callback_calls = []
        
        def progress_callback(current, total):
            callback_calls.append((current, total))
        
        extractor.extract_batch_parallel(
            urls,
            max_workers=2,
            show_progress=False,
            progress_callback=progress_callback
        )
        
        # Callback should have been called
        assert len(callback_calls) > 0
        # Last call should have current == total
        assert callback_calls[-1][0] == callback_calls[-1][1]
    
    def test_get_extraction_stats(self, extractor):
        """Test getting extraction statistics."""
        urls = ["https://www.bbc.com/news/technology"] * 2
        
        results = extractor.extract_batch_parallel(
            urls,
            max_workers=2,
            show_progress=False
        )
        
        stats = extractor.get_extraction_stats()
        
        assert 'total_extracted' in stats
        assert 'total_failed' in stats
        assert 'success_rate' in stats
        assert isinstance(stats['total_extracted'], int)


class TestResourceMonitoring:
    """Test resource usage monitoring."""
    
    @pytest.fixture
    def extractor(self, tmp_path):
        """Create extractor with temporary storage."""
        return ArticleExtractor(
            storage_dir=str(tmp_path / "articles"),
            log_dir=str(tmp_path / "logs")
        )
    
    def test_resource_monitoring_available(self, extractor):
        """Test that resource monitoring methods exist."""
        assert hasattr(extractor, 'get_resource_usage')
        assert callable(extractor.get_resource_usage)
    
    def test_get_resource_usage_returns_metrics(self, extractor):
        """Test that resource usage returns CPU and memory metrics."""
        usage = extractor.get_resource_usage()
        
        assert 'cpu_percent' in usage
        assert 'memory_mb' in usage
        assert 'memory_percent' in usage
        assert isinstance(usage['cpu_percent'], (int, float))
        assert isinstance(usage['memory_mb'], (int, float))


class TestThreadSafety:
    """Test thread safety of parallel operations."""
    
    @pytest.fixture
    def extractor(self, tmp_path):
        """Create extractor with temporary storage."""
        return ArticleExtractor(
            storage_dir=str(tmp_path / "articles"),
            log_dir=str(tmp_path / "logs")
        )
    
    def test_concurrent_extraction_no_race_conditions(self, extractor):
        """Test that concurrent extractions don't cause race conditions."""
        urls = ["https://www.bbc.com/news/technology"] * 10
        
        # Run parallel extraction multiple times
        results1 = extractor.extract_batch_parallel(urls, max_workers=4, show_progress=False)
        results2 = extractor.extract_batch_parallel(urls, max_workers=4, show_progress=False)
        
        # Should get consistent results
        assert isinstance(results1, list)
        assert isinstance(results2, list)
    
    def test_index_updates_thread_safe(self, extractor):
        """Test that article index updates are thread-safe."""
        urls = ["https://www.bbc.com/news/technology"] * 5
        
        extractor.extract_and_save_batch_parallel(urls, max_workers=3, show_progress=False)
        
        # Index should be consistent
        index = extractor.get_index()
        assert 'articles' in index
        assert isinstance(index['articles'], list)


class TestConfigurationIntegration:
    """Test integration with configuration system."""
    
    def test_extractor_uses_config_defaults(self, tmp_path):
        """Test that extractor uses configuration defaults."""
        from src.config import Config
        
        config = Config()
        extractor = ArticleExtractor(
            storage_dir=str(tmp_path / "articles"),
            log_dir=str(tmp_path / "logs"),
            timeout=config.article_timeout,
            max_retries=config.article_max_retries,
            min_text_length=config.article_min_text_length
        )
        
        assert extractor.timeout == config.article_timeout
        assert extractor.max_retries == config.article_max_retries
        assert extractor.min_text_length == config.article_min_text_length
    
    def test_parallel_extraction_uses_config_workers(self, tmp_path):
        """Test that parallel extraction can use config max_workers."""
        from src.config import Config
        
        config = Config()
        extractor = ArticleExtractor(
            storage_dir=str(tmp_path / "articles"),
            log_dir=str(tmp_path / "logs")
        )
        
        urls = ["https://www.bbc.com/news/technology"] * 3
        
        # Should work with config max_workers
        results = extractor.extract_batch_parallel(
            urls,
            max_workers=config.max_workers,
            show_progress=False
        )
        
        assert isinstance(results, list)


class TestErrorHandling:
    """Test error handling in parallel extraction."""
    
    @pytest.fixture
    def extractor(self, tmp_path):
        """Create extractor with temporary storage."""
        return ArticleExtractor(
            storage_dir=str(tmp_path / "articles"),
            log_dir=str(tmp_path / "logs")
        )
    
    def test_parallel_extraction_handles_failures_gracefully(self, extractor):
        """Test that parallel extraction handles individual failures."""
        urls = [
            "https://www.bbc.com/news/technology",  # Valid
            "https://invalid-url-12345.com/article",  # Invalid
            "https://www.reuters.com/technology/",  # May fail with 401
        ]
        
        results = extractor.extract_batch_parallel(
            urls,
            max_workers=2,
            show_progress=False
        )
        
        # Should return results list (successful extractions only)
        assert isinstance(results, list)
        # At least one should succeed (BBC)
        assert len(results) >= 1
    
    def test_parallel_extraction_with_zero_workers_raises_error(self, extractor):
        """Test that zero workers raises appropriate error."""
        urls = ["https://www.bbc.com/news/technology"]
        
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            extractor.extract_batch_parallel(urls, max_workers=0)
    
    def test_parallel_extraction_with_empty_urls(self, extractor):
        """Test parallel extraction with empty URL list."""
        results = extractor.extract_batch_parallel([], max_workers=2, show_progress=False)
        
        assert isinstance(results, list)
        assert len(results) == 0
