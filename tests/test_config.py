"""
Comprehensive Tests for Configuration Module

Following TDD principles: These tests are designed to FAIL initially,
then we implement the code to make them pass.

Tests cover:
- Configuration loading from environment variables
- Default values
- Validation
- Type checking
- Configuration updates
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

# This import will fail initially - that's expected in TDD
from src.config import Config, ConfigValidationError


class TestConfigurationDefaults:
    """Test default configuration values."""
    
    def test_default_ollama_settings(self):
        """Test default Ollama configuration."""
        config = Config()
        
        assert config.ollama_model == "nomic-embed-text"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_timeout == 30
    
    def test_default_embedding_settings(self):
        """Test default embedding configuration."""
        config = Config()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.embedding_batch_size == 10
        assert config.enable_disk_cache is True
    
    def test_default_performance_settings(self):
        """Test default performance configuration."""
        config = Config()
        
        assert config.max_workers == 4
        assert config.top_k_default == 5
    
    def test_default_article_extraction_settings(self):
        """Test default article extraction configuration."""
        config = Config()
        
        assert config.article_timeout == 30
        assert config.article_max_retries == 3
        assert config.article_min_text_length == 100
    
    def test_default_storage_paths(self):
        """Test default storage paths."""
        config = Config()
        
        assert config.article_cache_dir == "data/raw_articles"
        assert config.faiss_index_path == "data/embeddings/articles.index"
        assert config.saved_states_dir == "data/saved_states"
        assert config.cache_dir == "data/embeddings/cache"


class TestConfigurationFromEnvironment:
    """Test configuration loading from environment variables."""
    
    def test_load_ollama_from_env(self):
        """Test loading Ollama settings from environment."""
        with patch.dict(os.environ, {
            'OLLAMA_MODEL': 'custom-model',
            'OLLAMA_BASE_URL': 'http://custom:8080',
            'OLLAMA_TIMEOUT': '60'
        }):
            config = Config()
            
            assert config.ollama_model == "custom-model"
            assert config.ollama_base_url == "http://custom:8080"
            assert config.ollama_timeout == 60
    
    def test_load_performance_from_env(self):
        """Test loading performance settings from environment."""
        with patch.dict(os.environ, {
            'MAX_WORKERS': '8',
            'EMBEDDING_BATCH_SIZE': '20',
            'TOP_K_DEFAULT': '10'
        }):
            config = Config()
            
            assert config.max_workers == 8
            assert config.embedding_batch_size == 20
            assert config.top_k_default == 10
    
    def test_load_paths_from_env(self):
        """Test loading storage paths from environment."""
        with patch.dict(os.environ, {
            'ARTICLE_CACHE_DIR': '/custom/articles',
            'FAISS_INDEX_PATH': '/custom/index.faiss',
            'CACHE_DIR': '/custom/cache'
        }):
            config = Config()
            
            assert config.article_cache_dir == "/custom/articles"
            assert config.faiss_index_path == "/custom/index.faiss"
            assert config.cache_dir == "/custom/cache"
    
    def test_boolean_env_parsing(self):
        """Test parsing boolean values from environment."""
        with patch.dict(os.environ, {
            'ENABLE_DISK_CACHE': 'false'
        }):
            config = Config()
            assert config.enable_disk_cache is False
        
        with patch.dict(os.environ, {
            'ENABLE_DISK_CACHE': 'true'
        }):
            config = Config()
            assert config.enable_disk_cache is True
        
        with patch.dict(os.environ, {
            'ENABLE_DISK_CACHE': '1'
        }):
            config = Config()
            assert config.enable_disk_cache is True


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_validate_positive_integers(self):
        """Test validation of positive integer fields."""
        with patch.dict(os.environ, {'MAX_WORKERS': '0'}):
            with pytest.raises(ConfigValidationError, match="max_workers must be positive"):
                Config()
        
        with patch.dict(os.environ, {'CHUNK_SIZE': '-100'}):
            with pytest.raises(ConfigValidationError, match="chunk_size must be positive"):
                Config()
    
    def test_validate_chunk_overlap(self):
        """Test chunk overlap must be less than chunk size."""
        with patch.dict(os.environ, {
            'CHUNK_SIZE': '100',
            'CHUNK_OVERLAP': '150'
        }):
            with pytest.raises(ConfigValidationError, match="chunk_overlap must be less than chunk_size"):
                Config()
    
    def test_validate_url_format(self):
        """Test URL validation."""
        with patch.dict(os.environ, {'OLLAMA_BASE_URL': 'not-a-url'}):
            with pytest.raises(ConfigValidationError, match="Invalid URL"):
                Config()
    
    def test_validate_timeout_range(self):
        """Test timeout must be reasonable."""
        with patch.dict(os.environ, {'OLLAMA_TIMEOUT': '0'}):
            with pytest.raises(ConfigValidationError, match="timeout must be at least 1"):
                Config()


class TestConfigurationMethods:
    """Test configuration utility methods."""
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'ollama_model' in config_dict
        assert 'max_workers' in config_dict
        assert config_dict['ollama_model'] == config.ollama_model
    
    def test_repr(self):
        """Test string representation."""
        config = Config()
        repr_str = repr(config)
        
        assert 'Config' in repr_str
        assert 'ollama_model' in repr_str
    
    def test_update_config(self):
        """Test updating configuration values."""
        config = Config()
        
        config.update(max_workers=8, chunk_size=2000)
        
        assert config.max_workers == 8
        assert config.chunk_size == 2000
    
    def test_update_validates(self):
        """Test that update method validates new values."""
        config = Config()
        
        with pytest.raises(ConfigValidationError):
            config.update(max_workers=-1)
    
    def test_get_performance_config(self):
        """Test getting performance-related configuration."""
        config = Config()
        perf_config = config.get_performance_config()
        
        assert 'max_workers' in perf_config
        assert 'embedding_batch_size' in perf_config
        assert perf_config['max_workers'] == config.max_workers
    
    def test_get_storage_config(self):
        """Test getting storage-related configuration."""
        config = Config()
        storage_config = config.get_storage_config()
        
        assert 'article_cache_dir' in storage_config
        assert 'faiss_index_path' in storage_config
        assert storage_config['article_cache_dir'] == config.article_cache_dir


class TestConfigurationSingleton:
    """Test configuration singleton pattern."""
    
    def test_get_config_returns_same_instance(self):
        """Test that get_config() returns the same instance."""
        from src.config import get_config
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reset_config(self):
        """Test resetting configuration singleton."""
        from src.config import get_config, reset_config
        
        config1 = get_config()
        reset_config()
        config2 = get_config()
        
        # Should be different instances after reset
        assert config1 is not config2


class TestConfigurationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_integer_env_value(self):
        """Test handling of invalid integer environment values."""
        with patch.dict(os.environ, {'MAX_WORKERS': 'not-a-number'}):
            with pytest.raises(ConfigValidationError, match="Invalid integer"):
                Config()
    
    def test_empty_string_env_values(self):
        """Test handling of empty string environment values."""
        with patch.dict(os.environ, {'OLLAMA_MODEL': ''}):
            with pytest.raises(ConfigValidationError, match="cannot be empty"):
                Config()
    
    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed from string values."""
        with patch.dict(os.environ, {
            'OLLAMA_MODEL': '  nomic-embed-text  '
        }):
            config = Config()
            assert config.ollama_model == "nomic-embed-text"
    
    def test_path_expansion(self):
        """Test that paths are properly expanded."""
        with patch.dict(os.environ, {
            'ARTICLE_CACHE_DIR': '~/data/articles'
        }):
            config = Config()
            # Should expand ~ to home directory
            assert '~' not in config.article_cache_dir
