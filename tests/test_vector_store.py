"""
Comprehensive Test Suite for FAISS HNSW Vector Store

This test suite is designed with strict requirements to ensure production-ready quality.
Tests are written to FAIL first, then the implementation must be fixed to pass them.
NEVER modify these tests - only fix the implementation code.
"""

import os
import time
import tempfile
import shutil
import pytest
import numpy as np
from typing import List, Dict, Tuple

from src.storage.vector_store import VectorStore


class TestVectorStoreInitialization:
    """Test vector store initialization and configuration."""
    
    def test_creates_hnsw_index_not_flat(self):
        """MUST use IndexHNSWFlat, not IndexFlatL2."""
        store = VectorStore(dimension=768)
        
        # Check index type - must be HNSW
        index_type = type(store.index).__name__
        assert 'HNSW' in index_type, f"Index must be HNSW type, got {index_type}"
        assert 'Flat' not in index_type or 'HNSW' in index_type, "Must not use IndexFlatL2"
    
    def test_default_dimension_is_768(self):
        """Default dimension MUST be 768 for standard sentence transformers."""
        store = VectorStore()
        assert store.dimension == 768, "Default dimension must be 768"
    
    def test_custom_dimension_accepted(self):
        """Must support custom dimensions."""
        store = VectorStore(dimension=384)
        assert store.dimension == 384, "Custom dimension must be respected"
    
    def test_hnsw_parameters_set_correctly(self):
        """HNSW parameters must be optimized for 768-dim vectors."""
        store = VectorStore(dimension=768)
        
        # M should be between 16-32 for good performance
        assert hasattr(store, 'M') or hasattr(store.index, 'hnsw'), \
            "HNSW parameters must be accessible"
        
        # efConstruction should be at least 200 for quality
        if hasattr(store, 'efConstruction'):
            assert store.efConstruction >= 200, "efConstruction must be >= 200"
    
    def test_index_starts_empty(self):
        """New index must start with zero vectors."""
        store = VectorStore(dimension=768)
        assert store.count() == 0, "New index must be empty"
    
    def test_metadata_starts_empty(self):
        """Metadata list must start empty."""
        store = VectorStore(dimension=768)
        assert len(store.metadata) == 0, "Metadata must start empty"


class TestVectorAddition:
    """Test adding vectors with metadata."""
    
    def test_add_single_vector_with_metadata(self):
        """Must correctly add a single vector with metadata."""
        store = VectorStore(dimension=768)
        
        vector = np.random.randn(768).astype(np.float32).tolist()
        metadata = {
            'article_id': 'art_001',
            'title': 'Test Article',
            'url': 'https://example.com/article',
            'chunk_index': 0
        }
        
        store.add_embeddings([vector], [metadata])
        
        assert store.count() == 1, "Must have exactly 1 vector"
        assert len(store.metadata) == 1, "Must have exactly 1 metadata entry"
        assert store.metadata[0] == metadata, "Metadata must match exactly"
    
    def test_add_batch_vectors(self):
        """Must efficiently add multiple vectors at once."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(100)]
        metadata = [
            {
                'article_id': f'art_{i:03d}',
                'title': f'Article {i}',
                'url': f'https://example.com/article/{i}',
                'chunk_index': i % 5
            }
            for i in range(100)
        ]
        
        store.add_embeddings(vectors, metadata)
        
        assert store.count() == 100, "Must have exactly 100 vectors"
        assert len(store.metadata) == 100, "Must have exactly 100 metadata entries"
    
    def test_metadata_synchronization(self):
        """Metadata MUST stay synchronized with vectors - critical requirement."""
        store = VectorStore(dimension=768)
        
        # Add first batch
        vectors1 = [np.random.randn(768).astype(np.float32).tolist() for _ in range(50)]
        metadata1 = [{'id': i, 'batch': 1} for i in range(50)]
        store.add_embeddings(vectors1, metadata1)
        
        # Add second batch
        vectors2 = [np.random.randn(768).astype(np.float32).tolist() for _ in range(30)]
        metadata2 = [{'id': i, 'batch': 2} for i in range(50, 80)]
        store.add_embeddings(vectors2, metadata2)
        
        assert store.count() == 80, "Must have 80 vectors total"
        assert len(store.metadata) == 80, "Must have 80 metadata entries"
        
        # Verify synchronization
        for i in range(50):
            assert store.metadata[i]['batch'] == 1, f"First batch metadata wrong at {i}"
        for i in range(50, 80):
            assert store.metadata[i]['batch'] == 2, f"Second batch metadata wrong at {i}"
    
    def test_reject_dimension_mismatch(self):
        """MUST reject vectors with wrong dimensions."""
        store = VectorStore(dimension=768)
        
        wrong_vector = np.random.randn(384).astype(np.float32).tolist()
        
        with pytest.raises((ValueError, AssertionError, Exception)):
            store.add_embeddings([wrong_vector], [{'id': 1}])
    
    def test_reject_metadata_count_mismatch(self):
        """MUST reject when vector count != metadata count."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(5)]
        metadata = [{'id': i} for i in range(3)]  # Only 3 metadata for 5 vectors
        
        with pytest.raises((ValueError, AssertionError, Exception)):
            store.add_embeddings(vectors, metadata)


class TestSimilaritySearch:
    """Test similarity search functionality."""
    
    def test_search_returns_most_similar(self):
        """Search MUST return the most similar vectors."""
        store = VectorStore(dimension=768)
        
        # Create a known vector
        target_vector = np.random.randn(768).astype(np.float32)
        
        # Add target and some random vectors
        vectors = [target_vector.tolist()]
        vectors.extend([np.random.randn(768).astype(np.float32).tolist() for _ in range(99)])
        
        metadata = [{'id': i, 'is_target': i == 0} for i in range(100)]
        store.add_embeddings(vectors, metadata)
        
        # Search with the exact target vector
        results = store.search(target_vector.tolist(), k=5)
        
        assert len(results) == 5, "Must return exactly 5 results"
        
        # First result MUST be the target (or very close)
        best_match = results[0]
        assert best_match[1]['id'] == 0, "Most similar must be the target vector"
        assert best_match[1]['is_target'] is True, "Must find the target"
    
    def test_search_returns_similarity_scores(self):
        """Results MUST include similarity scores, not just distances."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(50)]
        metadata = [{'id': i} for i in range(50)]
        store.add_embeddings(vectors, metadata)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        results = store.search(query, k=10)
        
        assert len(results) == 10, "Must return 10 results"
        
        # Each result must be (score, metadata) tuple
        for score, meta in results:
            assert isinstance(score, (int, float)), "Score must be numeric"
            assert isinstance(meta, dict), "Metadata must be dict"
            assert 'id' in meta, "Metadata must contain id"
    
    def test_search_scores_are_ordered(self):
        """Results MUST be ordered by similarity (best first)."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(100)]
        metadata = [{'id': i} for i in range(100)]
        store.add_embeddings(vectors, metadata)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        results = store.search(query, k=20)
        
        scores = [score for score, _ in results]
        
        # Scores must be in descending order (higher = more similar)
        # OR ascending order if using distance (lower = more similar)
        # Check if sorted in either direction
        is_ascending = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
        is_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        assert is_ascending or is_descending, "Results must be ordered by score"
    
    def test_search_respects_k_parameter(self):
        """Must return exactly k results (or fewer if not enough vectors)."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(50)]
        metadata = [{'id': i} for i in range(50)]
        store.add_embeddings(vectors, metadata)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        
        results_5 = store.search(query, k=5)
        assert len(results_5) == 5, "Must return exactly 5 results"
        
        results_20 = store.search(query, k=20)
        assert len(results_20) == 20, "Must return exactly 20 results"
        
        results_100 = store.search(query, k=100)
        assert len(results_100) == 50, "Must return 50 (all available) when k > total"
    
    def test_search_on_empty_index_returns_empty(self):
        """Searching empty index MUST return empty list, not error."""
        store = VectorStore(dimension=768)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        results = store.search(query, k=10)
        
        assert results == [], "Empty index must return empty results"
    
    def test_search_with_invalid_dimension_raises_error(self):
        """MUST reject queries with wrong dimensions."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(10)]
        metadata = [{'id': i} for i in range(10)]
        store.add_embeddings(vectors, metadata)
        
        wrong_query = np.random.randn(384).astype(np.float32).tolist()
        
        with pytest.raises((ValueError, AssertionError, Exception)):
            store.search(wrong_query, k=5)


class TestPersistence:
    """Test save and load functionality."""
    
    def test_save_and_load_preserves_vectors(self):
        """Save/load MUST preserve all vectors with perfect integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            
            # Create and populate store
            store1 = VectorStore(dimension=768, index_path=index_path)
            vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(100)]
            metadata = [{'id': i, 'value': f'item_{i}'} for i in range(100)]
            store1.add_embeddings(vectors, metadata)
            
            # Save
            store1.save_index()
            
            # Load in new store
            store2 = VectorStore(dimension=768, index_path=index_path)
            store2.load_index()
            
            assert store2.count() == 100, "Must load all 100 vectors"
            assert len(store2.metadata) == 100, "Must load all 100 metadata entries"
            
            # Verify metadata integrity
            for i in range(100):
                assert store2.metadata[i]['id'] == i, f"Metadata {i} corrupted"
                assert store2.metadata[i]['value'] == f'item_{i}', f"Metadata {i} value wrong"
    
    def test_save_and_load_preserves_search_results(self):
        """Search results MUST be identical before and after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            
            # Create and populate store
            store1 = VectorStore(dimension=768, index_path=index_path)
            vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(100)]
            metadata = [{'id': i} for i in range(100)]
            store1.add_embeddings(vectors, metadata)
            
            # Search before save
            query = np.random.randn(768).astype(np.float32).tolist()
            results_before = store1.search(query, k=10)
            
            # Save and load
            store1.save_index()
            store2 = VectorStore(dimension=768, index_path=index_path)
            store2.load_index()
            
            # Search after load
            results_after = store2.search(query, k=10)
            
            assert len(results_after) == len(results_before), "Result count must match"
            
            # Results should be identical (same IDs in same order)
            for i in range(len(results_before)):
                assert results_before[i][1]['id'] == results_after[i][1]['id'], \
                    f"Result {i} ID mismatch after load"
    
    def test_load_nonexistent_index_creates_new(self):
        """Loading non-existent index MUST create new empty index, not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'nonexistent.index')
            
            store = VectorStore(dimension=768, index_path=index_path)
            
            assert store.count() == 0, "Non-existent index must create empty store"
            assert len(store.metadata) == 0, "Metadata must be empty"
    
    def test_save_creates_both_index_and_metadata_files(self):
        """Save MUST create both index file and metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            
            store = VectorStore(dimension=768, index_path=index_path)
            vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(10)]
            metadata = [{'id': i} for i in range(10)]
            store.add_embeddings(vectors, metadata)
            
            store.save_index()
            
            assert os.path.exists(index_path), "Index file must exist"
            assert os.path.exists(index_path + '.metadata'), "Metadata file must exist"


class TestPerformanceBenchmarks:
    """Performance tests - MUST meet strict latency requirements."""
    
    def test_search_100_vectors_under_10ms(self):
        """Search on 100 vectors MUST complete in under 10ms."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(100)]
        metadata = [{'id': i} for i in range(100)]
        store.add_embeddings(vectors, metadata)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        
        # Warm up
        store.search(query, k=10)
        
        # Measure
        start = time.perf_counter()
        for _ in range(10):
            store.search(query, k=10)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 10) * 1000
        assert avg_time_ms < 10, f"Search must be <10ms, got {avg_time_ms:.2f}ms"
    
    def test_search_1000_vectors_under_50ms(self):
        """Search on 1,000 vectors MUST complete in under 50ms."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(1000)]
        metadata = [{'id': i} for i in range(1000)]
        store.add_embeddings(vectors, metadata)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        
        # Warm up
        store.search(query, k=10)
        
        # Measure
        start = time.perf_counter()
        for _ in range(10):
            store.search(query, k=10)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 10) * 1000
        assert avg_time_ms < 50, f"Search must be <50ms, got {avg_time_ms:.2f}ms"
    
    def test_search_5000_vectors_under_100ms(self):
        """Search on 5,000 vectors MUST complete in under 100ms - CRITICAL."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(5000)]
        metadata = [{'id': i} for i in range(5000)]
        store.add_embeddings(vectors, metadata)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        
        # Warm up
        store.search(query, k=10)
        
        # Measure
        start = time.perf_counter()
        for _ in range(10):
            store.search(query, k=10)
        elapsed = time.perf_counter() - start
        
        avg_time_ms = (elapsed / 10) * 1000
        assert avg_time_ms < 100, f"Search must be <100ms, got {avg_time_ms:.2f}ms"
    
    def test_batch_add_is_efficient(self):
        """Batch adding MUST work correctly and complete in reasonable time."""
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(1000)]
        metadata = [{'id': i} for i in range(1000)]
        
        # Batch add should complete successfully
        store = VectorStore(dimension=768)
        start = time.perf_counter()
        store.add_embeddings(vectors, metadata)
        batch_time = time.perf_counter() - start
        
        # Verify correctness
        assert store.count() == 1000, "Must add all 1000 vectors"
        assert len(store.metadata) == 1000, "Must have all 1000 metadata entries"
        
        # Batch add should complete in reasonable time (< 1 second for 1000 vectors)
        # This is a sanity check, not a micro-benchmark
        assert batch_time < 1.0, \
            f"Batch add of 1000 vectors must complete in <1s, took {batch_time:.3f}s"
        
        # Verify search works correctly after batch add
        query = np.random.randn(768).astype(np.float32).tolist()
        results = store.search(query, k=5)
        assert len(results) == 5, "Search must work after batch add"



class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_add_empty_list_does_nothing(self):
        """Adding empty list MUST not crash or corrupt state."""
        store = VectorStore(dimension=768)
        
        store.add_embeddings([], [])
        
        assert store.count() == 0, "Empty add must not change count"
        assert len(store.metadata) == 0, "Empty add must not change metadata"
    
    def test_search_with_k_zero_returns_empty(self):
        """Searching with k=0 MUST return empty list."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(10)]
        metadata = [{'id': i} for i in range(10)]
        store.add_embeddings(vectors, metadata)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        results = store.search(query, k=0)
        
        assert results == [], "k=0 must return empty results"
    
    def test_search_with_negative_k_raises_error(self):
        """Negative k MUST raise error."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(10)]
        metadata = [{'id': i} for i in range(10)]
        store.add_embeddings(vectors, metadata)
        
        query = np.random.randn(768).astype(np.float32).tolist()
        
        with pytest.raises((ValueError, AssertionError, Exception)):
            store.search(query, k=-5)
    
    def test_clear_resets_to_empty_state(self):
        """Clear MUST completely reset the index."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(100)]
        metadata = [{'id': i} for i in range(100)]
        store.add_embeddings(vectors, metadata)
        
        assert store.count() == 100, "Should have 100 vectors"
        
        store.clear()
        
        assert store.count() == 0, "After clear, must have 0 vectors"
        assert len(store.metadata) == 0, "After clear, metadata must be empty"
        
        # Should be able to add new vectors after clear
        new_vector = np.random.randn(768).astype(np.float32).tolist()
        store.add_embeddings([new_vector], [{'id': 'new'}])
        assert store.count() == 1, "Must be able to add after clear"
    
    def test_duplicate_vectors_are_allowed(self):
        """Adding duplicate vectors MUST be allowed (same vector, different metadata)."""
        store = VectorStore(dimension=768)
        
        vector = np.random.randn(768).astype(np.float32).tolist()
        
        # Add same vector twice with different metadata
        store.add_embeddings([vector], [{'id': 1, 'source': 'first'}])
        store.add_embeddings([vector], [{'id': 2, 'source': 'second'}])
        
        assert store.count() == 2, "Must allow duplicate vectors"
        assert len(store.metadata) == 2, "Must have 2 metadata entries"
        assert store.metadata[0]['source'] == 'first'
        assert store.metadata[1]['source'] == 'second'
    
    def test_very_large_metadata_handled(self):
        """MUST handle large metadata dictionaries without issues."""
        store = VectorStore(dimension=768)
        
        vector = np.random.randn(768).astype(np.float32).tolist()
        large_metadata = {
            'id': 1,
            'title': 'A' * 10000,  # 10KB title
            'content': 'B' * 50000,  # 50KB content
            'tags': ['tag' + str(i) for i in range(1000)],  # 1000 tags
            'nested': {'level1': {'level2': {'level3': 'deep'}}}
        }
        
        store.add_embeddings([vector], [large_metadata])
        
        assert store.count() == 1, "Must handle large metadata"
        assert store.metadata[0]['title'] == 'A' * 10000, "Large metadata must be preserved"


class TestIndexManagement:
    """Test index management operations."""
    
    def test_count_returns_correct_number(self):
        """Count MUST always return exact number of vectors."""
        store = VectorStore(dimension=768)
        
        assert store.count() == 0, "Empty store must have count 0"
        
        for i in range(1, 101):
            vector = np.random.randn(768).astype(np.float32).tolist()
            store.add_embeddings([vector], [{'id': i}])
            assert store.count() == i, f"After adding {i} vectors, count must be {i}"
    
    def test_get_dimension_returns_correct_value(self):
        """get_dimension MUST return the configured dimension."""
        store = VectorStore(dimension=768)
        assert store.get_dimension() == 768, "Must return correct dimension"
        
        store2 = VectorStore(dimension=384)
        assert store2.get_dimension() == 384, "Must return custom dimension"
    
    def test_get_stats_provides_comprehensive_info(self):
        """get_stats MUST provide complete statistics."""
        store = VectorStore(dimension=768)
        
        vectors = [np.random.randn(768).astype(np.float32).tolist() for _ in range(50)]
        metadata = [{'id': i} for i in range(50)]
        store.add_embeddings(vectors, metadata)
        
        stats = store.get_stats()
        
        assert 'total_vectors' in stats, "Stats must include total_vectors"
        assert 'dimension' in stats, "Stats must include dimension"
        assert 'metadata_count' in stats, "Stats must include metadata_count"
        
        assert stats['total_vectors'] == 50, "Stats must show correct vector count"
        assert stats['dimension'] == 768, "Stats must show correct dimension"
        assert stats['metadata_count'] == 50, "Stats must show correct metadata count"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
