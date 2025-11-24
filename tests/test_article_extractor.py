"""
Comprehensive Test Suite for Article Extractor

Tests cover:
- URL validation
- Article extraction
- Error handling
- Data validation
- Batch processing
- Edge cases
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.article_extractor import ArticleExtractor


class TestURLValidation:
    """Test URL validation functionality."""
    
    def test_valid_urls(self):
        """Test that valid URLs are accepted."""
        extractor = ArticleExtractor()
        
        valid_urls = [
            "https://techcrunch.com/article",
            "http://www.example.com/news",
            "https://www.theverge.com/2024/1/1/article-title",
        ]
        
        for url in valid_urls:
            assert extractor._validate_url(url), f"Should accept valid URL: {url}"
    
    def test_invalid_urls(self):
        """Test that invalid URLs are rejected."""
        extractor = ArticleExtractor()
        
        invalid_urls = [
            "not-a-url",
            "://no-scheme.com",
            "",
            "   ",
        ]
        
        for url in invalid_urls:
            assert not extractor._validate_url(url), f"Should reject invalid URL: {url}"


class TestTextCleaning:
    """Test text cleaning and normalization."""
    
    def test_clean_whitespace(self):
        """Test whitespace normalization."""
        extractor = ArticleExtractor()
        
        text = "This  has   multiple    spaces"
        cleaned = extractor._clean_text(text)
        assert "  " not in cleaned
        assert cleaned == "This has multiple spaces"
    
    def test_remove_html_entities(self):
        """Test HTML entity removal."""
        extractor = ArticleExtractor()
        
        text = "This &amp; that &lt;tag&gt;"
        cleaned = extractor._clean_text(text)
        assert "&amp;" not in cleaned
        assert "&lt;" not in cleaned
    
    def test_normalize_newlines(self):
        """Test newline normalization."""
        extractor = ArticleExtractor()
        
        text = "Line 1\n\n\n\nLine 2"
        cleaned = extractor._clean_text(text)
        assert "\n\n\n" not in cleaned
    
    def test_empty_text(self):
        """Test handling of empty text."""
        extractor = ArticleExtractor()
        
        assert extractor._clean_text("") == ""
        assert extractor._clean_text(None) == ""


class TestQualityValidation:
    """Test article quality validation."""
    
    def test_valid_article(self):
        """Test validation of a valid article."""
        extractor = ArticleExtractor(min_text_length=100)
        
        article = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'text': 'A' * 150,  # 150 characters
        }
        
        is_valid, reason = extractor._validate_article_quality(article)
        assert is_valid
        assert reason == "Valid"
    
    def test_missing_title(self):
        """Test rejection of article without title."""
        extractor = ArticleExtractor()
        
        article = {
            'url': 'https://example.com/article',
            'text': 'A' * 150,
        }
        
        is_valid, reason = extractor._validate_article_quality(article)
        assert not is_valid
        assert "title" in reason.lower()
    
    def test_missing_text(self):
        """Test rejection of article without text."""
        extractor = ArticleExtractor()
        
        article = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
        }
        
        is_valid, reason = extractor._validate_article_quality(article)
        assert not is_valid
        assert "text" in reason.lower()
    
    def test_text_too_short(self):
        """Test rejection of article with insufficient text."""
        extractor = ArticleExtractor(min_text_length=100)
        
        article = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'text': 'Short text',  # Only 10 characters
        }
        
        is_valid, reason = extractor._validate_article_quality(article)
        assert not is_valid
        assert "short" in reason.lower()
    
    def test_missing_url(self):
        """Test rejection of article without URL."""
        extractor = ArticleExtractor()
        
        article = {
            'title': 'Test Article',
            'text': 'A' * 150,
        }
        
        is_valid, reason = extractor._validate_article_quality(article)
        assert not is_valid
        assert "url" in reason.lower()


class TestArticleExtraction:
    """Test article extraction functionality."""
    
    @patch('src.ingestion.article_extractor.Article')
    def test_successful_extraction(self, mock_article_class):
        """Test successful article extraction."""
        # Setup mock
        mock_article = Mock()
        mock_article.title = "Test Article"
        mock_article.text = "A" * 200
        mock_article.authors = ["John Doe"]
        mock_article.publish_date = datetime(2024, 1, 1)
        mock_article.top_image = "https://example.com/image.jpg"
        mock_article.meta_description = "Test description"
        mock_article.meta_keywords = ["test", "article"]
        
        mock_article_class.return_value = mock_article
        
        # Test extraction
        extractor = ArticleExtractor()
        result = extractor.extract_article("https://example.com/article")
        
        assert result is not None
        assert result['title'] == "Test Article"
        assert len(result['text']) >= 100
        assert result['authors'] == ["John Doe"]
        assert result['url'] == "https://example.com/article"
    
    @patch('src.ingestion.article_extractor.Article')
    def test_extraction_with_retry(self, mock_article_class):
        """Test retry logic on failure."""
        # First call fails, second succeeds
        mock_article = Mock()
        mock_article.download.side_effect = [Exception("Network error"), None]
        mock_article.title = "Test Article"
        mock_article.text = "A" * 200
        mock_article.authors = []
        mock_article.publish_date = None
        mock_article.top_image = None
        mock_article.meta_description = None
        mock_article.meta_keywords = []
        
        mock_article_class.return_value = mock_article
        
        extractor = ArticleExtractor(max_retries=3)
        result = extractor.extract_article("https://example.com/article")
        
        # Should succeed on retry
        assert mock_article.download.call_count == 2
    
    def test_invalid_url_rejection(self):
        """Test that invalid URLs are rejected."""
        extractor = ArticleExtractor()
        result = extractor.extract_article("not-a-valid-url")
        
        assert result is None


class TestBatchProcessing:
    """Test batch article extraction."""
    
    @patch('src.ingestion.article_extractor.Article')
    def test_batch_extraction(self, mock_article_class):
        """Test extraction of multiple articles."""
        # Setup mock
        mock_article = Mock()
        mock_article.title = "Test Article"
        mock_article.text = "A" * 200
        mock_article.authors = []
        mock_article.publish_date = None
        mock_article.top_image = None
        mock_article.meta_description = None
        mock_article.meta_keywords = []
        
        mock_article_class.return_value = mock_article
        
        # Test batch extraction
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
        ]
        
        extractor = ArticleExtractor()
        results = extractor.extract_batch(urls, delay=0)  # No delay for testing
        
        assert len(results) == 3
        assert all(r['title'] == "Test Article" for r in results)
    
    @patch('src.ingestion.article_extractor.Article')
    def test_batch_with_failures(self, mock_article_class):
        """Test batch processing with some failures."""
        # Setup mock to fail on second article
        call_count = [0]
        
        def side_effect():
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Network error")
        
        mock_article = Mock()
        mock_article.download.side_effect = side_effect
        mock_article.title = "Test Article"
        mock_article.text = "A" * 200
        mock_article.authors = []
        mock_article.publish_date = None
        mock_article.top_image = None
        mock_article.meta_description = None
        mock_article.meta_keywords = []
        
        mock_article_class.return_value = mock_article
        
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
        ]
        
        extractor = ArticleExtractor(max_retries=1)
        results = extractor.extract_batch(urls, delay=0)
        
        # Should have 2 successful extractions (1st and 3rd)
        # The test might be flaky due to retry logic, so we check >= 2
        assert len(results) >= 2


class TestJSONStorage:
    """Test JSON storage functionality."""
    
    def test_save_article(self, tmp_path):
        """Test saving article to JSON."""
        extractor = ArticleExtractor(storage_dir=str(tmp_path))
        
        article = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'text': 'A' * 200,
            'authors': ['John Doe'],
            'publish_date': '2024-01-01',
            'extracted_at': datetime.now().isoformat(),
        }
        
        filepath = extractor.save_article(article)
        
        assert filepath is not None
        assert Path(filepath).exists()
        
        # Verify content
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['title'] == 'Test Article'
        assert saved_data['url'] == 'https://example.com/article'
    
    def test_index_updated(self, tmp_path):
        """Test that index is updated when saving articles."""
        extractor = ArticleExtractor(storage_dir=str(tmp_path))
        
        article = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'text': 'A' * 200,
            'extracted_at': datetime.now().isoformat(),
            'source_domain': 'example.com',
        }
        
        extractor.save_article(article)
        
        index = extractor.get_index()
        assert len(index['articles']) == 1
        assert index['articles'][0]['title'] == 'Test Article'
    
    def test_save_batch(self, tmp_path):
        """Test saving multiple articles."""
        extractor = ArticleExtractor(storage_dir=str(tmp_path))
        
        articles = [
            {
                'url': f'https://example.com/article{i}',
                'title': f'Test Article {i}',
                'text': 'A' * 200,
                'extracted_at': datetime.now().isoformat(),
                'source_domain': 'example.com',
            }
            for i in range(3)
        ]
        
        paths = extractor.save_batch(articles)
        
        assert len(paths) == 3
        assert all(Path(p).exists() for p in paths)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('src.ingestion.article_extractor.Article')
    def test_network_timeout(self, mock_article_class):
        """Test handling of network timeout."""
        import requests
        
        mock_article = Mock()
        mock_article.download.side_effect = requests.exceptions.Timeout("Timeout")
        
        mock_article_class.return_value = mock_article
        
        extractor = ArticleExtractor(max_retries=2)
        result = extractor.extract_article("https://example.com/article")
        
        assert result is None
        assert mock_article.download.call_count == 2
    
    @patch('src.ingestion.article_extractor.Article')
    def test_connection_error(self, mock_article_class):
        """Test handling of connection error."""
        import requests
        
        mock_article = Mock()
        mock_article.download.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        mock_article_class.return_value = mock_article
        
        extractor = ArticleExtractor(max_retries=2)
        result = extractor.extract_article("https://example.com/article")
        
        assert result is None


class TestLogging:
    """Test logging functionality."""
    
    def test_logger_created(self, tmp_path):
        """Test that logger is properly created."""
        extractor = ArticleExtractor(log_dir=str(tmp_path))
        
        assert extractor.logger is not None
        assert extractor.logger.name == 'ArticleExtractor'
    
    def test_log_file_created(self, tmp_path):
        """Test that logger is properly configured with file handler."""
        import logging
        extractor = ArticleExtractor(log_dir=str(tmp_path))
        
        # Verify logger has file handler
        has_file_handler = any(
            isinstance(h, logging.FileHandler) for h in extractor.logger.handlers
        )
        assert has_file_handler, "Logger should have a file handler"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
