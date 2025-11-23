"""
HTML Parser

Parses HTML content using BeautifulSoup4 for custom scraping needs.
"""

from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import requests


class HTMLParser:
    """Parses HTML content to extract structured data."""
    
    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize the HTML parser.
        
        Args:
            user_agent: Custom user agent string for requests
        """
        self.user_agent = user_agent or 'Mozilla/5.0 (compatible; NewsBot/1.0)'
        self.headers = {'User-Agent': self.user_agent}
    
    def fetch_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content as string, or None if failed
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML content into BeautifulSoup object.
        
        Args:
            html: HTML content string
            
        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html, 'html.parser')
    
    def extract_links(self, soup: BeautifulSoup, base_url: str = '') -> List[str]:
        """
        Extract all links from parsed HTML.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for relative links
            
        Returns:
            List of URLs
        """
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                links.append(href)
            elif base_url and href.startswith('/'):
                links.append(base_url.rstrip('/') + href)
        return links
    
    def extract_text(self, soup: BeautifulSoup, selector: Optional[str] = None) -> str:
        """
        Extract text content from HTML.
        
        Args:
            soup: BeautifulSoup object
            selector: CSS selector for specific elements
            
        Returns:
            Extracted text
        """
        if selector:
            elements = soup.select(selector)
            return ' '.join([elem.get_text(strip=True) for elem in elements])
        return soup.get_text(strip=True)
    
    def extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Extract metadata from HTML (title, description, etc.).
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        return metadata
