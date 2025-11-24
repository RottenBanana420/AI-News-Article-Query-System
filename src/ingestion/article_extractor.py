"""
Article Extractor Module

Comprehensive article extraction with robust error handling, logging, and quality validation.
"""

import os
import re
import json
import time
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import requests
from newspaper import Article
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


class ArticleExtractor:
    """
    Extracts article content from news URLs with robust error handling and quality validation.
    
    Features:
    - Single URL or batch processing
    - Retry logic with exponential backoff
    - Text cleaning and normalization
    - Quality validation
    - JSON storage with metadata
    - Comprehensive logging
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        min_text_length: int = 100
    ):
        """
        Initialize the article extractor.
        
        Args:
            storage_dir: Directory to store extracted articles (default: data/raw_articles)
            log_dir: Directory for log files (default: logs)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retry attempts for failed requests (default: 3)
            min_text_length: Minimum text length for quality validation (default: 100)
        """
        self.storage_dir = Path(storage_dir or os.getenv('ARTICLE_CACHE_DIR', 'data/raw_articles'))
        self.log_dir = Path(log_dir or 'logs')
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_text_length = min_text_length
        
        # Create directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        self.current_user_agent_idx = 0
        
        # Index file for tracking all articles
        self.index_file = self.storage_dir / 'articles_index.json'
        self._load_index()
        
        self.logger.info(f"ArticleExtractor initialized with storage_dir={self.storage_dir}, timeout={self.timeout}s")
    
    def _setup_logging(self):
        """Configure logging with rotating file handler."""
        log_file = self.log_dir / 'article_ingestion.log'
        
        # Create logger
        self.logger = logging.getLogger('ArticleExtractor')
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _load_index(self):
        """Load the articles index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load index file: {e}. Starting with empty index.")
                self.index = {'articles': [], 'last_updated': None}
        else:
            self.index = {'articles': [], 'last_updated': None}
    
    def _save_index(self):
        """Save the articles index to file."""
        try:
            self.index['last_updated'] = datetime.now().isoformat()
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save index file: {e}")
    
    def _get_user_agent(self) -> str:
        """Get next user agent from rotation."""
        user_agent = self.user_agents[self.current_user_agent_idx]
        self.current_user_agent_idx = (self.current_user_agent_idx + 1) % len(self.user_agents)
        return user_agent
    
    def _validate_url(self, url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            result = urlparse(url)
            is_valid = all([result.scheme, result.netloc])
            if not is_valid:
                self.logger.warning(f"Invalid URL format: {url}")
            return is_valid
        except Exception as e:
            self.logger.warning(f"URL validation error for {url}: {e}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML entities
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s\.,!?;:\-\'"()\[\]{}]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _validate_article_quality(self, article_data: Dict) -> tuple[bool, str]:
        """
        Validate article meets minimum quality thresholds.
        
        Args:
            article_data: Extracted article data
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check required fields
        if not article_data.get('title'):
            return False, "Missing title"
        
        if not article_data.get('text'):
            return False, "Missing text content"
        
        # Check minimum text length
        text_length = len(article_data['text'])
        if text_length < self.min_text_length:
            return False, f"Text too short ({text_length} < {self.min_text_length} chars)"
        
        # Check URL
        if not article_data.get('url'):
            return False, "Missing URL"
        
        return True, "Valid"
    
    def _extract_with_newspaper(self, url: str) -> Optional[Dict]:
        """
        Extract article using newspaper3k library.
        
        Args:
            url: Article URL
            
        Returns:
            Extracted article data or None if failed
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Extract metadata
            data = {
                'url': url,
                'title': article.title,
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'text': article.text,
                'top_image': article.top_image,
                'meta_description': article.meta_description,
                'meta_keywords': article.meta_keywords,
                'source_domain': urlparse(url).netloc,
                'extraction_method': 'newspaper3k',
                'extracted_at': datetime.now().isoformat()
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Newspaper3k extraction failed for {url}: {e}")
            return None
    
    def extract_article(self, url: str) -> Optional[Dict]:
        """
        Extract article from a single URL with retry logic.
        
        Args:
            url: Article URL
            
        Returns:
            Extracted article data or None if failed
        """
        self.logger.info(f"Extracting article from: {url}")
        
        # Validate URL
        if not self._validate_url(url):
            self.logger.error(f"Invalid URL: {url}")
            return None
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Extract article
                article_data = self._extract_with_newspaper(url)
                
                if not article_data:
                    raise Exception("Extraction returned no data")
                
                # Clean text
                article_data['text'] = self._clean_text(article_data['text'])
                article_data['title'] = self._clean_text(article_data['title'])
                
                # Validate quality
                is_valid, reason = self._validate_article_quality(article_data)
                if not is_valid:
                    self.logger.warning(f"Article quality validation failed for {url}: {reason}")
                    return None
                
                self.logger.info(f"Successfully extracted article from {url} (length: {len(article_data['text'])} chars)")
                return article_data
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries} for {url}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Connection error on attempt {attempt + 1}/{self.max_retries} for {url}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    
            except Exception as e:
                self.logger.error(f"Extraction error on attempt {attempt + 1}/{self.max_retries} for {url}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
        
        self.logger.error(f"Failed to extract article from {url} after {self.max_retries} attempts")
        return None
    
    def extract_batch(self, urls: List[str], delay: float = 1.0) -> List[Dict]:
        """
        Extract articles from multiple URLs.
        
        Args:
            urls: List of article URLs
            delay: Delay between requests in seconds (default: 1.0)
            
        Returns:
            List of successfully extracted articles
        """
        self.logger.info(f"Starting batch extraction for {len(urls)} URLs")
        
        articles = []
        failed_urls = []
        
        for i, url in enumerate(urls, 1):
            self.logger.info(f"Processing {i}/{len(urls)}: {url}")
            
            article = self.extract_article(url)
            
            if article:
                articles.append(article)
            else:
                failed_urls.append(url)
            
            # Rate limiting - be respectful to servers
            if i < len(urls):
                time.sleep(delay)
        
        self.logger.info(f"Batch extraction complete: {len(articles)} successful, {len(failed_urls)} failed")
        
        if failed_urls:
            self.logger.warning(f"Failed URLs: {failed_urls}")
        
        return articles
    
    def save_article(self, article_data: Dict) -> Optional[str]:
        """
        Save article to JSON file.
        
        Args:
            article_data: Article data to save
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Generate filename from title and timestamp
            title = article_data.get('title', 'untitled')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Sanitize filename
            safe_title = re.sub(r'[^\w\s-]', '', title)
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            safe_title = safe_title[:50]  # Limit length
            
            filename = f"{safe_title}_{timestamp}.json"
            filepath = self.storage_dir / filename
            
            # Save article
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)
            
            # Update index
            self.index['articles'].append({
                'filename': filename,
                'url': article_data.get('url'),
                'title': article_data.get('title'),
                'extracted_at': article_data.get('extracted_at'),
                'source_domain': article_data.get('source_domain')
            })
            self._save_index()
            
            self.logger.info(f"Saved article to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save article: {e}")
            return None
    
    def save_batch(self, articles: List[Dict]) -> List[str]:
        """
        Save multiple articles to JSON files.
        
        Args:
            articles: List of article data
            
        Returns:
            List of saved file paths
        """
        self.logger.info(f"Saving {len(articles)} articles")
        
        saved_paths = []
        for article in articles:
            path = self.save_article(article)
            if path:
                saved_paths.append(path)
        
        self.logger.info(f"Saved {len(saved_paths)}/{len(articles)} articles")
        return saved_paths
    
    def extract_and_save(self, url: str) -> Optional[str]:
        """
        Extract article and save to file in one operation.
        
        Args:
            url: Article URL
            
        Returns:
            Path to saved file or None if failed
        """
        article = self.extract_article(url)
        if article:
            return self.save_article(article)
        return None
    
    def extract_and_save_batch(self, urls: List[str], delay: float = 1.0) -> List[str]:
        """
        Extract and save multiple articles.
        
        Args:
            urls: List of article URLs
            delay: Delay between requests in seconds
            
        Returns:
            List of saved file paths
        """
        articles = self.extract_batch(urls, delay=delay)
        return self.save_batch(articles)
    
    def get_index(self) -> Dict:
        """
        Get the articles index.
        
        Returns:
            Index dictionary with article metadata
        """
        return self.index
