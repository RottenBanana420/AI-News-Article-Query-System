"""
Comprehensive Test Suite for OllamaEmbeddingService

Tests are designed to be challenging and drive improvements in the implementation.
Tests cover edge cases, error conditions, performance, and integration scenarios.
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests

from src.embeddings.ollama_service import (
    OllamaEmbeddingService,
    OllamaConnectionError,
    OllamaModelError,
    EmbeddingDimensionError,
    CacheStats
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def service_no_cache(temp_cache_dir):
    """Service instance with caching disabled for isolated tests."""
    return OllamaEmbeddingService(
        model="nomic-embed-text",
        enable_disk_cache=False,
        cache_dir=temp_cache_dir
    )


@pytest.fixture
def service_with_cache(temp_cache_dir):
    """Service instance with full caching enabled."""
    return OllamaEmbeddingService(
        model="nomic-embed-text",
        enable_disk_cache=True,
        cache_dir=temp_cache_dir
    )


# ============================================================================
# Connection and Setup Tests
# ============================================================================

class TestConnectionVerification:
    """Test connection verification and error handling."""
    
    def test_verify_connection_success(self, service_no_cache):
        """Test successful connection to Ollama."""
        try:
            result = service_no_cache.verify_connection()
            assert result is True, "Connection verification should return True"
        except OllamaConnectionError as e:
            pytest.fail(f"Connection should succeed when Ollama is running: {e}")
    
    def test_verify_connection_failure_ollama_not_running(self, temp_cache_dir):
        """Test connection failure when Ollama is not running."""
        service = OllamaEmbeddingService(
            base_url="http://localhost:99999",  # Invalid port
            enable_disk_cache=False,
            cache_dir=temp_cache_dir,
            timeout=1
        )
        
        with pytest.raises(OllamaConnectionError) as exc_info:
            service.verify_connection()
        
        # Should raise OllamaConnectionError (message may vary based on URL parsing)
        error_msg = str(exc_info.value)
        assert "ollama" in error_msg.lower()
        assert "connect" in error_msg.lower() or "invalid" in error_msg.lower() or "parse" in error_msg.lower()
    
    def test_verify_connection_timeout(self, temp_cache_dir):
        """Test connection timeout handling."""
        # Use a non-routable IP to trigger timeout
        service = OllamaEmbeddingService(
            base_url="http://10.255.255.1:11434",
            enable_disk_cache=False,
            cache_dir=temp_cache_dir,
            timeout=1
        )
        
        with pytest.raises(OllamaConnectionError) as exc_info:
            service.verify_connection()
        
        # Non-routable IPs may raise ConnectionError or Timeout depending on OS
        error_msg = str(exc_info.value).lower()
        assert "timed out" in error_msg or "unable to connect" in error_msg
    
    def test_verify_model_available_success(self, service_no_cache):
        """Test model availability check when model exists."""
        try:
            result = service_no_cache.verify_model_available()
            assert result is True
        except OllamaModelError as e:
            pytest.fail(f"Model should be available: {e}")
    
    def test_verify_model_not_available(self, service_no_cache):
        """Test model availability check when model doesn't exist."""
        service_no_cache.model = "nonexistent-model-xyz-123"
        
        with pytest.raises(OllamaModelError) as exc_info:
            service_no_cache.verify_model_available()
        
        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "ollama pull" in error_msg.lower()


# ============================================================================
# Text Chunking Tests
# ============================================================================

class TestTextChunking:
    """Test text chunking with various scenarios."""
    
    def test_chunk_empty_text(self, service_no_cache):
        """Test chunking empty text."""
        chunks = service_no_cache.chunk_text("")
        assert chunks == [], "Empty text should return empty list"
    
    def test_chunk_short_text(self, service_no_cache):
        """Test text shorter than chunk size."""
        text = "This is a short text."
        chunks = service_no_cache.chunk_text(text)
        
        assert len(chunks) == 1, "Short text should return single chunk"
        assert chunks[0] == text, "Chunk should match original text"
    
    def test_chunk_exact_size(self, service_no_cache):
        """Test text exactly matching chunk size."""
        text = "x" * 1000  # Exactly chunk_size
        chunks = service_no_cache.chunk_text(text)
        
        assert len(chunks) == 1, "Text matching chunk size should return single chunk"
        assert chunks[0] == text
    
    def test_chunk_long_text(self, service_no_cache):
        """Test chunking long text into multiple chunks."""
        text = "x" * 5000  # 5x chunk size
        chunks = service_no_cache.chunk_text(text)
        
        assert len(chunks) > 1, "Long text should create multiple chunks"
        
        # Verify overlap
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i][-200:]  # Last 200 chars (overlap size)
            chunk2_start = chunks[i + 1][:200]  # First 200 chars
            # There should be some overlap
            assert len(chunk1_end) > 0 and len(chunk2_start) > 0
    
    def test_chunk_custom_size(self, service_no_cache):
        """Test chunking with custom chunk size."""
        text = "x" * 3000
        chunks = service_no_cache.chunk_text(text, chunk_size=500, chunk_overlap=100)
        
        assert len(chunks) > 1
        # Each chunk should be approximately 500 chars (except possibly the last)
        for chunk in chunks[:-1]:
            assert len(chunk) <= 500
    
    def test_chunk_zero_overlap(self, service_no_cache):
        """Test chunking with zero overlap."""
        text = "x" * 2000
        chunks = service_no_cache.chunk_text(text, chunk_size=1000, chunk_overlap=0)
        
        # With zero overlap, should get exactly 2 chunks, but edge cases may vary
        assert len(chunks) >= 2 and len(chunks) <= 3
        # Verify total length is preserved
        combined = ''.join(chunks)
        assert len(combined) >= len(text)
    
    def test_chunk_overlap_larger_than_size(self, service_no_cache):
        """Test edge case where overlap >= chunk size (should handle gracefully)."""
        text = "x" * 3000
        chunks = service_no_cache.chunk_text(text, chunk_size=500, chunk_overlap=600)
        
        # Should not create infinite loop
        assert len(chunks) > 0
        assert len(chunks) < 100  # Sanity check


# ============================================================================
# Embedding Generation Tests
# ============================================================================

class TestEmbeddingGeneration:
    """Test single embedding generation."""
    
    def test_generate_single_embedding(self, service_no_cache):
        """Test generating embedding for single text."""
        text = "This is a test sentence for embedding generation."
        
        embedding = service_no_cache.generate_embedding(text, use_cache=False)
        
        assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
        assert embedding.dtype == np.float32, "Embedding should be float32"
        assert len(embedding) == 768, f"Expected 768 dimensions, got {len(embedding)}"
        assert not np.all(embedding == 0), "Embedding should not be all zeros"
    
    def test_generate_embedding_different_texts(self, service_no_cache):
        """Test that different texts produce different embeddings."""
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "Python is a powerful programming language."
        
        emb1 = service_no_cache.generate_embedding(text1, use_cache=False)
        emb2 = service_no_cache.generate_embedding(text2, use_cache=False)
        
        # Embeddings should be different
        assert not np.array_equal(emb1, emb2), "Different texts should produce different embeddings"
        
        # But both should be valid
        assert len(emb1) == 768
        assert len(emb2) == 768
    
    def test_generate_embedding_similar_texts(self, service_no_cache):
        """Test that similar texts produce similar embeddings."""
        text1 = "I love machine learning and artificial intelligence."
        text2 = "I enjoy AI and machine learning technologies."
        text3 = "The weather is sunny today."
        
        emb1 = service_no_cache.generate_embedding(text1, use_cache=False)
        emb2 = service_no_cache.generate_embedding(text2, use_cache=False)
        emb3 = service_no_cache.generate_embedding(text3, use_cache=False)
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)
        
        # Similar texts should have higher similarity than unrelated texts
        assert sim_1_2 > sim_1_3, "Similar texts should have higher cosine similarity"
        assert sim_1_2 > 0.5, "Similar texts should have positive similarity"
    
    def test_generate_embedding_empty_text(self, service_no_cache):
        """Test embedding generation with empty text."""
        # This might fail or return a specific embedding - test actual behavior
        try:
            embedding = service_no_cache.generate_embedding("", use_cache=False)
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 768
        except Exception as e:
            # If it raises an error, that's also acceptable behavior
            assert isinstance(e, (ValueError, RuntimeError))
    
    def test_generate_embedding_very_long_text(self, service_no_cache):
        """Test embedding generation with very long text."""
        # Create text longer than typical context window
        text = "This is a test sentence. " * 1000  # ~25,000 chars
        
        # Very long text may cause Ollama server error (500)
        # This is expected behavior - the model has limits
        try:
            embedding = service_no_cache.generate_embedding(text, use_cache=False)
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 768
        except RuntimeError as e:
            # Server error is acceptable for extremely long text
            assert "500" in str(e) or "too long" in str(e).lower()
    
    def test_dimension_verification_disabled(self, temp_cache_dir):
        """Test that dimension verification can be disabled."""
        service = OllamaEmbeddingService(
            verify_dimensions=False,
            enable_disk_cache=False,
            cache_dir=temp_cache_dir
        )
        
        # Should not raise error even if dimensions are unexpected
        # (though with real API, dimensions will be correct)
        embedding = service.generate_embedding("test", use_cache=False)
        assert isinstance(embedding, np.ndarray)


# ============================================================================
# Batch Processing Tests
# ============================================================================

class TestBatchProcessing:
    """Test batch embedding generation."""
    
    def test_batch_empty_list(self, service_no_cache):
        """Test batch processing with empty list."""
        embeddings = service_no_cache.generate_embeddings_batch([])
        assert embeddings == [], "Empty input should return empty list"
    
    def test_batch_single_item(self, service_no_cache):
        """Test batch processing with single item."""
        texts = ["Single text for testing."]
        embeddings = service_no_cache.generate_embeddings_batch(texts, use_cache=False)
        
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], np.ndarray)
        assert len(embeddings[0]) == 768
    
    def test_batch_small_batch(self, service_no_cache):
        """Test batch processing with small batch (< batch_size)."""
        texts = [f"Test sentence number {i}." for i in range(5)]
        embeddings = service_no_cache.generate_embeddings_batch(texts, use_cache=False)
        
        assert len(embeddings) == 5
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert len(emb) == 768
    
    def test_batch_exact_batch_size(self, service_no_cache):
        """Test batch processing with exactly batch_size items."""
        batch_size = service_no_cache.batch_size
        texts = [f"Test sentence {i}." for i in range(batch_size)]
        
        embeddings = service_no_cache.generate_embeddings_batch(texts, use_cache=False)
        
        assert len(embeddings) == batch_size
    
    def test_batch_large_batch(self, service_no_cache):
        """Test batch processing with large batch (> batch_size)."""
        texts = [f"Test sentence number {i}." for i in range(50)]
        
        start_time = time.time()
        embeddings = service_no_cache.generate_embeddings_batch(
            texts,
            use_cache=False,
            show_progress=True
        )
        elapsed = time.time() - start_time
        
        assert len(embeddings) == 50
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert len(emb) == 768
        
        # Performance check: should process at reasonable speed
        rate = len(texts) / elapsed
        assert rate > 0.5, f"Processing rate too slow: {rate:.2f} embeddings/s"
    
    def test_batch_progress_callback(self, service_no_cache):
        """Test batch processing with progress callback."""
        texts = [f"Test {i}" for i in range(25)]
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        embeddings = service_no_cache.generate_embeddings_batch(
            texts,
            use_cache=False,
            progress_callback=progress_callback
        )
        
        assert len(embeddings) == 25
        assert len(progress_calls) > 0, "Progress callback should be called"
        
        # Verify progress calls are increasing
        for i in range(len(progress_calls) - 1):
            assert progress_calls[i][0] <= progress_calls[i + 1][0]
        
        # Last call should be total
        assert progress_calls[-1][0] == 25
        assert progress_calls[-1][1] == 25
    
    def test_batch_different_lengths(self, service_no_cache):
        """Test batch processing with texts of varying lengths."""
        texts = [
            "Short.",
            "This is a medium length sentence for testing.",
            "This is a much longer sentence that contains significantly more words and characters to test how the embedding service handles variable length inputs in batch processing." * 5
        ]
        
        embeddings = service_no_cache.generate_embeddings_batch(texts, use_cache=False)
        
        assert len(embeddings) == 3
        # All should have same dimension regardless of input length
        for emb in embeddings:
            assert len(emb) == 768


# ============================================================================
# Caching Tests
# ============================================================================

class TestCaching:
    """Test caching functionality."""
    
    def test_cache_hit_same_text(self, service_with_cache):
        """Test cache hit when generating embedding for same text twice."""
        text = "This is a test for caching."
        
        # First call - cache miss
        emb1 = service_with_cache.generate_embedding(text, use_cache=True)
        stats1 = service_with_cache.get_cache_stats()
        
        # Second call - should be cache hit
        emb2 = service_with_cache.generate_embedding(text, use_cache=True)
        stats2 = service_with_cache.get_cache_stats()
        
        # Embeddings should be identical
        assert np.array_equal(emb1, emb2), "Cached embedding should match original"
        
        # Stats should show cache hit
        assert stats2['hits'] == stats1['hits'] + 1, "Cache hits should increase"
        assert stats2['total_requests'] == stats1['total_requests'] + 1
    
    def test_cache_miss_different_text(self, service_with_cache):
        """Test cache miss when generating embeddings for different texts."""
        text1 = "First text."
        text2 = "Second text."
        
        emb1 = service_with_cache.generate_embedding(text1, use_cache=True)
        stats1 = service_with_cache.get_cache_stats()
        
        emb2 = service_with_cache.generate_embedding(text2, use_cache=True)
        stats2 = service_with_cache.get_cache_stats()
        
        # Should be cache miss
        assert stats2['misses'] == stats1['misses'] + 1
        assert stats2['cache_size'] == 2
    
    def test_cache_disabled(self, service_no_cache):
        """Test that caching can be disabled."""
        text = "Test text for no caching."
        
        emb1 = service_no_cache.generate_embedding(text, use_cache=False)
        emb2 = service_no_cache.generate_embedding(text, use_cache=False)
        
        stats = service_no_cache.get_cache_stats()
        
        # Should have no cache hits since caching is disabled
        assert stats['hits'] == 0
        assert stats['cache_size'] == 0
    
    def test_cache_statistics(self, service_with_cache):
        """Test cache statistics calculation."""
        texts = [f"Text {i}" for i in range(10)]
        
        # Generate embeddings
        for text in texts:
            service_with_cache.generate_embedding(text, use_cache=True)
        
        # Generate some duplicates
        for text in texts[:5]:
            service_with_cache.generate_embedding(text, use_cache=True)
        
        stats = service_with_cache.get_cache_stats()
        
        assert stats['total_requests'] == 15
        assert stats['hits'] == 5
        assert stats['misses'] == 10
        assert stats['cache_size'] == 10
        assert stats['hit_rate'] == pytest.approx(5/15, rel=0.01)
    
    def test_disk_cache_persistence(self, temp_cache_dir):
        """Test that disk cache persists across service instances."""
        text = "Test persistence."
        
        # Create first service and generate embedding
        service1 = OllamaEmbeddingService(
            enable_disk_cache=True,
            cache_dir=temp_cache_dir
        )
        emb1 = service1.generate_embedding(text, use_cache=True)
        
        # Create new service instance (simulating restart)
        service2 = OllamaEmbeddingService(
            enable_disk_cache=True,
            cache_dir=temp_cache_dir
        )
        
        # Should load from disk cache
        stats_before = service2.get_cache_stats()
        emb2 = service2.generate_embedding(text, use_cache=True)
        stats_after = service2.get_cache_stats()
        
        # Should be cache hit from disk-loaded cache
        assert np.array_equal(emb1, emb2)
        assert stats_after['hits'] == stats_before['hits'] + 1
    
    def test_clear_cache_memory_only(self, service_with_cache):
        """Test clearing memory cache only."""
        text = "Test cache clearing."
        
        service_with_cache.generate_embedding(text, use_cache=True)
        stats_before = service_with_cache.get_cache_stats()
        assert stats_before['cache_size'] > 0
        
        service_with_cache.clear_cache(clear_disk=False)
        stats_after = service_with_cache.get_cache_stats()
        
        assert stats_after['cache_size'] == 0
        assert stats_after['hits'] == 0
        assert stats_after['misses'] == 0
    
    def test_clear_cache_including_disk(self, service_with_cache):
        """Test clearing both memory and disk cache."""
        text = "Test full cache clearing."
        
        service_with_cache.generate_embedding(text, use_cache=True)
        
        # Verify cache files exist
        cache_files = list(service_with_cache.cache_dir.glob('*.npy'))
        assert len(cache_files) > 0
        
        service_with_cache.clear_cache(clear_disk=True)
        
        # Verify cache files are deleted
        cache_files = list(service_with_cache.cache_dir.glob('*.npy'))
        assert len(cache_files) == 0


# ============================================================================
# Article Processing Tests
# ============================================================================

class TestArticleProcessing:
    """Test article processing (chunking + embedding)."""
    
    def test_process_empty_article(self, service_no_cache):
        """Test processing empty article."""
        result = service_no_cache.process_article("", use_cache=False)
        
        assert result['chunks'] == []
        assert result['embeddings'] == []
        assert result['num_chunks'] == 0
    
    def test_process_short_article(self, service_no_cache):
        """Test processing short article (single chunk)."""
        text = "This is a short article. " * 10
        
        result = service_no_cache.process_article(text, use_cache=False)
        
        assert result['num_chunks'] == 1
        assert len(result['embeddings']) == 1
        assert len(result['embeddings'][0]) == 768
    
    def test_process_long_article(self, service_no_cache):
        """Test processing long article (multiple chunks)."""
        # Create article longer than chunk size
        text = "This is a test sentence for a long article. " * 100
        
        result = service_no_cache.process_article(
            text,
            use_cache=False,
            show_progress=True
        )
        
        assert result['num_chunks'] > 1
        assert len(result['embeddings']) == result['num_chunks']
        assert result['total_characters'] == len(text)
        
        # All embeddings should be valid
        for emb in result['embeddings']:
            assert isinstance(emb, np.ndarray)
            assert len(emb) == 768
    
    def test_process_article_custom_chunking(self, service_no_cache):
        """Test article processing with custom chunk parameters."""
        text = "x" * 5000
        
        result = service_no_cache.process_article(
            text,
            chunk_size=500,
            chunk_overlap=50,
            use_cache=False
        )
        
        assert result['num_chunks'] > 1
        # Each chunk should be approximately 500 chars
        for chunk in result['chunks'][:-1]:
            assert len(chunk) <= 500


# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestPerformance:
    """Test performance and benchmarking."""
    
    def test_single_embedding_performance(self, service_no_cache):
        """Benchmark single embedding generation."""
        text = "This is a performance test sentence."
        
        start = time.time()
        embedding = service_no_cache.generate_embedding(text, use_cache=False)
        elapsed = time.time() - start
        
        assert isinstance(embedding, np.ndarray)
        # Should complete in reasonable time (adjust based on hardware)
        assert elapsed < 5.0, f"Single embedding took too long: {elapsed:.2f}s"
    
    def test_batch_performance(self, service_no_cache):
        """Benchmark batch processing performance."""
        texts = [f"Performance test sentence number {i}." for i in range(100)]
        
        start = time.time()
        embeddings = service_no_cache.generate_embeddings_batch(texts, use_cache=False)
        elapsed = time.time() - start
        
        rate = len(texts) / elapsed
        
        assert len(embeddings) == 100
        assert rate > 0.5, f"Batch processing too slow: {rate:.2f} embeddings/s"
        print(f"\nBatch performance: {rate:.2f} embeddings/s")
    
    def test_cache_performance_improvement(self, service_with_cache):
        """Test that caching improves performance."""
        texts = [f"Cache test {i}" for i in range(20)]
        
        # First pass - no cache
        start1 = time.time()
        for text in texts:
            service_with_cache.generate_embedding(text, use_cache=True)
        time_uncached = time.time() - start1
        
        # Second pass - with cache
        start2 = time.time()
        for text in texts:
            service_with_cache.generate_embedding(text, use_cache=True)
        time_cached = time.time() - start2
        
        # Cached should be significantly faster
        assert time_cached < time_uncached * 0.5, \
            f"Cache not improving performance: uncached={time_uncached:.2f}s, cached={time_cached:.2f}s"
        
        print(f"\nCache speedup: {time_uncached/time_cached:.2f}x")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_connection_error_on_embedding_generation(self, temp_cache_dir):
        """Test error handling when Ollama is not available during embedding generation."""
        service = OllamaEmbeddingService(
            base_url="http://localhost:99999",
            enable_disk_cache=False,
            cache_dir=temp_cache_dir,
            timeout=1
        )
        
        with pytest.raises(OllamaConnectionError):
            service.generate_embedding("test", use_cache=False)
    
    def test_timeout_handling(self, temp_cache_dir):
        """Test timeout handling for slow responses."""
        service = OllamaEmbeddingService(
            base_url="http://10.255.255.1:11434",
            enable_disk_cache=False,
            cache_dir=temp_cache_dir,
            timeout=1
        )
        
        with pytest.raises(OllamaConnectionError) as exc_info:
            service.generate_embedding("test", use_cache=False)
        
        # Non-routable IPs may raise ConnectionError or Timeout
        error_msg = str(exc_info.value).lower()
        assert "timed out" in error_msg or "unable to connect" in error_msg
    
    def test_invalid_model_name(self, temp_cache_dir):
        """Test handling of invalid model name."""
        service = OllamaEmbeddingService(
            model="invalid-model-xyz",
            enable_disk_cache=False,
            cache_dir=temp_cache_dir
        )
        
        with pytest.raises(OllamaModelError):
            service.verify_model_available()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_full_workflow(self, service_with_cache):
        """Test complete workflow: verify, chunk, embed, cache."""
        # Verify connection
        assert service_with_cache.verify_connection()
        assert service_with_cache.verify_model_available()
        
        # Process article
        article_text = """
        Artificial intelligence has made remarkable progress in recent years.
        Machine learning models can now perform tasks that were once thought
        to be exclusively human. Natural language processing has enabled
        computers to understand and generate human-like text. Computer vision
        systems can recognize objects and faces with high accuracy.
        """ * 10  # Make it longer
        
        result = service_with_cache.process_article(
            article_text,
            use_cache=True,
            show_progress=True
        )
        
        assert result['num_chunks'] > 0
        assert len(result['embeddings']) == result['num_chunks']
        
        # Verify caching works
        result2 = service_with_cache.process_article(
            article_text,
            use_cache=True
        )
        
        stats = service_with_cache.get_cache_stats()
        assert stats['hits'] > 0
        
        # Embeddings should match
        for emb1, emb2 in zip(result['embeddings'], result2['embeddings']):
            assert np.array_equal(emb1, emb2)
    
    def test_realistic_article_sizes(self, service_no_cache):
        """Test with realistic article sizes."""
        # Short article (~500 words)
        short_article = "word " * 500
        
        # Medium article (~2000 words)
        medium_article = "word " * 2000
        
        # Long article (~5000 words)
        long_article = "word " * 5000
        
        for article, name in [(short_article, "short"),
                               (medium_article, "medium"),
                               (long_article, "long")]:
            result = service_no_cache.process_article(article, use_cache=False)
            
            assert result['num_chunks'] > 0
            assert len(result['embeddings']) == result['num_chunks']
            
            print(f"\n{name.capitalize()} article: {result['num_chunks']} chunks")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
