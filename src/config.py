"""
Centralized Configuration Module

Provides a single source of truth for all system configuration parameters.
Loads settings from environment variables with sensible defaults and validation.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class Config:
    """
    Centralized configuration for the AI News Article Query System.
    
    All configuration parameters are loaded from environment variables
    with sensible defaults. Validation is performed on initialization.
    """
    
    # Ollama Settings
    ollama_model: str = field(default="nomic-embed-text")
    ollama_base_url: str = field(default="http://localhost:11434")
    ollama_timeout: int = field(default=30)
    
    # Embedding Parameters
    chunk_size: int = field(default=1000)
    chunk_overlap: int = field(default=200)
    embedding_batch_size: int = field(default=10)
    enable_disk_cache: bool = field(default=True)
    
    # Performance Settings
    max_workers: int = field(default=4)
    top_k_default: int = field(default=5)
    
    # Article Extraction Settings
    article_timeout: int = field(default=30)
    article_max_retries: int = field(default=3)
    article_min_text_length: int = field(default=100)
    
    # Storage Paths
    article_cache_dir: str = field(default="data/raw_articles")
    faiss_index_path: str = field(default="data/embeddings/articles.index")
    saved_states_dir: str = field(default="data/saved_states")
    cache_dir: str = field(default="data/embeddings/cache")
    
    def __post_init__(self):
        """Load configuration from environment and validate."""
        self._load_from_environment()
        self._validate()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Ollama Settings
        self.ollama_model = self._get_env_str('OLLAMA_MODEL', self.ollama_model)
        self.ollama_base_url = self._get_env_str('OLLAMA_BASE_URL', self.ollama_base_url)
        self.ollama_timeout = self._get_env_int('OLLAMA_TIMEOUT', self.ollama_timeout)
        
        # Embedding Parameters
        self.chunk_size = self._get_env_int('CHUNK_SIZE', self.chunk_size)
        self.chunk_overlap = self._get_env_int('CHUNK_OVERLAP', self.chunk_overlap)
        self.embedding_batch_size = self._get_env_int('EMBEDDING_BATCH_SIZE', self.embedding_batch_size)
        self.enable_disk_cache = self._get_env_bool('ENABLE_DISK_CACHE', self.enable_disk_cache)
        
        # Performance Settings
        self.max_workers = self._get_env_int('MAX_WORKERS', self.max_workers)
        self.top_k_default = self._get_env_int('TOP_K_DEFAULT', self.top_k_default)
        
        # Article Extraction Settings
        self.article_timeout = self._get_env_int('ARTICLE_TIMEOUT', self.article_timeout)
        self.article_max_retries = self._get_env_int('ARTICLE_MAX_RETRIES', self.article_max_retries)
        self.article_min_text_length = self._get_env_int('ARTICLE_MIN_TEXT_LENGTH', self.article_min_text_length)
        
        # Storage Paths
        self.article_cache_dir = self._get_env_path('ARTICLE_CACHE_DIR', self.article_cache_dir)
        self.faiss_index_path = self._get_env_path('FAISS_INDEX_PATH', self.faiss_index_path)
        self.saved_states_dir = self._get_env_path('SAVED_STATES_DIR', self.saved_states_dir)
        self.cache_dir = self._get_env_path('CACHE_DIR', self.cache_dir)
    
    def _get_env_str(self, key: str, default: str) -> str:
        """Get string value from environment."""
        value = os.getenv(key, default)
        if isinstance(value, str):
            value = value.strip()
        return value
    
    def _get_env_int(self, key: str, default: int) -> int:
        """Get integer value from environment."""
        value = os.getenv(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            raise ConfigValidationError(
                f"Invalid integer value for {key}: '{value}'"
            )
    
    def _get_env_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment."""
        value = os.getenv(key)
        if value is None:
            return default
        
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
        else:
            return default
    
    def _get_env_path(self, key: str, default: str) -> str:
        """Get path value from environment with expansion."""
        value = os.getenv(key, default)
        if isinstance(value, str):
            value = value.strip()
            # Expand ~ to home directory
            value = os.path.expanduser(value)
        return value
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate non-empty strings
        if not self.ollama_model:
            raise ConfigValidationError("ollama_model cannot be empty")
        
        # Validate positive integers
        positive_int_fields = [
            ('max_workers', self.max_workers),
            ('chunk_size', self.chunk_size),
            ('chunk_overlap', self.chunk_overlap),
            ('embedding_batch_size', self.embedding_batch_size),
            ('top_k_default', self.top_k_default),
            ('article_max_retries', self.article_max_retries),
            ('article_min_text_length', self.article_min_text_length),
        ]
        
        for field_name, value in positive_int_fields:
            if value <= 0:
                raise ConfigValidationError(
                    f"{field_name} must be positive, got {value}"
                )
        
        # Validate timeouts (at least 1 second)
        if self.ollama_timeout < 1:
            raise ConfigValidationError(
                f"ollama_timeout must be at least 1, got {self.ollama_timeout}"
            )
        if self.article_timeout < 1:
            raise ConfigValidationError(
                f"article_timeout must be at least 1, got {self.article_timeout}"
            )
        
        # Validate chunk overlap < chunk size
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigValidationError(
                f"chunk_overlap must be less than chunk_size"
            )

        
        # Validate URL format
        try:
            parsed = urlparse(self.ollama_base_url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError("Missing scheme or netloc")
        except Exception as e:
            raise ConfigValidationError(
                f"Invalid URL for ollama_base_url: {self.ollama_base_url}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        items = []
        for key, value in self.to_dict().items():
            items.append(f"{key}={value!r}")
        return f"Config({', '.join(items)})"
    
    def update(self, **kwargs):
        """
        Update configuration values with validation.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Store original values for rollback
        original_values = {}
        
        try:
            # Update values
            for key, value in kwargs.items():
                if not hasattr(self, key):
                    raise ConfigValidationError(f"Unknown configuration parameter: {key}")
                original_values[key] = getattr(self, key)
                setattr(self, key, value)
            
            # Validate new configuration
            self._validate()
            
        except Exception as e:
            # Rollback on validation failure
            for key, value in original_values.items():
                setattr(self, key, value)
            raise
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            'max_workers': self.max_workers,
            'embedding_batch_size': self.embedding_batch_size,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k_default': self.top_k_default,
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage-related configuration."""
        return {
            'article_cache_dir': self.article_cache_dir,
            'faiss_index_path': self.faiss_index_path,
            'saved_states_dir': self.saved_states_dir,
            'cache_dir': self.cache_dir,
        }


# Singleton instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance (singleton pattern).
    
    Returns:
        Config: Global configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config():
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None
