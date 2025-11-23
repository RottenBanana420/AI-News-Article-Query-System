"""
Article Scraper

Fetches news articles from URLs using requests and newspaper3k.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import requests
from newspaper import Article
from dotenv import load_dotenv

load_dotenv()


class ArticleScraper:
    """Scrapes and extracts article content from URLs."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the article scraper.
        
        Args:
            cache_dir: Directory to cache raw articles (default: from .env)
        """
        self.cache_dir = cache_dir or os.getenv('ARTICLE_CACHE_DIR', 'data/raw_articles')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def scrape_article(self, url: str) -> Dict[str, str]:
        """
        Scrape a single article from a URL.
        
        Args:
            url: URL of the article to scrape
            
        Returns:
            Dictionary containing article metadata and content
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            return {
                'url': url,
                'title': article.title,
                'authors': article.authors,
                'publish_date': str(article.publish_date) if article.publish_date else None,
                'text': article.text,
                'top_image': article.top_image,
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {}
    
    def scrape_multiple(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Scrape multiple articles from a list of URLs.
        
        Args:
            urls: List of article URLs
            
        Returns:
            List of article dictionaries
        """
        articles = []
        for url in urls:
            article = self.scrape_article(url)
            if article:
                articles.append(article)
        return articles
    
    def save_article(self, article: Dict[str, str], filename: Optional[str] = None) -> str:
        """
        Save article to cache directory.
        
        Args:
            article: Article dictionary
            filename: Optional filename (default: generated from title)
            
        Returns:
            Path to saved file
        """
        if not filename:
            # Generate filename from title or URL
            title = article.get('title', 'untitled')
            filename = f"{title[:50].replace(' ', '_').replace('/', '_')}.txt"
        
        filepath = os.path.join(self.cache_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {article.get('title', 'N/A')}\n")
            f.write(f"URL: {article.get('url', 'N/A')}\n")
            f.write(f"Authors: {', '.join(article.get('authors', []))}\n")
            f.write(f"Published: {article.get('publish_date', 'N/A')}\n")
            f.write(f"Scraped: {article.get('scraped_at', 'N/A')}\n")
            f.write("\n" + "="*80 + "\n\n")
            f.write(article.get('text', ''))
        
        return filepath
