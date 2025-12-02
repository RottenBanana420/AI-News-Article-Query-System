"""
Real-world test script for article extraction.

Tests extraction from actual news sources with valid URLs.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.article_extractor import ArticleExtractor


def main():
    """Run real-world extraction tests."""
    print("=" * 80)
    print("Article Extractor - Real World Test")
    print("=" * 80)
    print()
    
    # Initialize extractor
    extractor = ArticleExtractor()
    
    # Valid test URLs from recent articles (November 2024)
    test_urls = [
        # TechCrunch AI articles
        "https://techcrunch.com/2025/11/23/ai-is-too-risky-to-insure-say-people-whose-job-is-insuring-risk/",
        "https://techcrunch.com/2025/11/23/chatgpt-told-them-they-were-special-their-families-say-it-led-to-tragedy/",
        "https://techcrunch.com/2025/11/22/trump-administration-might-not-fight-state-ai-regulations-after-all/",
        
        # Wired technology articles
        "https://www.wired.com/story/the-pelvic-floor-is-a-problem/",
        "https://www.wired.com/story/bitcoin-scam-mining-as-service/",
        
        # The Verge (using homepage for now - will extract latest article)
        "https://www.theverge.com/",
    ]
    
    print(f"Testing extraction from {len(test_urls)} URLs...\n")
    
    # Test individual extraction
    print("Test 1: Individual Article Extraction")
    print("-" * 80)
    
    first_url = test_urls[0]
    print(f"Extracting: {first_url}")
    article = extractor.extract_article(first_url)
    
    if article:
        print(f"✓ Success!")
        print(f"  Title: {article['title'][:80]}...")
        print(f"  Text length: {len(article['text'])} characters")
        print(f"  Authors: {article.get('authors', [])}")
        print(f"  Source: {article.get('source_domain')}")
        
        # Save the article
        filepath = extractor.save_article(article)
        print(f"  Saved to: {filepath}")
    else:
        print(f"✗ Failed to extract article")
    
    print()
    
    # Test batch extraction
    print("Test 2: Batch Extraction")
    print("-" * 80)
    
    print(f"Extracting {len(test_urls)} articles...")
    articles = extractor.extract_batch(test_urls[:3], delay=2.0)  # Use first 3 URLs with 2s delay
    
    print(f"\n✓ Successfully extracted {len(articles)}/{len(test_urls[:3])} articles")
    
    for i, article in enumerate(articles, 1):
        print(f"\n  Article {i}:")
        print(f"    Title: {article['title'][:60]}...")
        print(f"    Length: {len(article['text'])} chars")
        print(f"    Source: {article.get('source_domain')}")
    
    # Save all articles
    print("\nSaving articles...")
    saved_paths = extractor.save_batch(articles)
    print(f"✓ Saved {len(saved_paths)} articles to {extractor.storage_dir}")
    
    print()
    
    # Test error handling
    print("Test 3: Error Handling")
    print("-" * 80)
    
    invalid_urls = [
        "not-a-valid-url",
        "https://this-domain-does-not-exist-12345.com/article",
    ]
    
    for url in invalid_urls:
        print(f"Testing invalid URL: {url}")
        result = extractor.extract_article(url)
        if result is None:
            print(f"  ✓ Correctly handled invalid URL")
        else:
            print(f"  ✗ Should have returned None")
    
    print()
    
    # Show index
    print("Test 4: Article Index")
    print("-" * 80)
    
    index = extractor.get_index()
    print(f"Total articles in index: {len(index['articles'])}")
    print(f"Last updated: {index.get('last_updated', 'N/A')}")
    
    if index['articles']:
        print("\nRecent articles:")
        for article_info in index['articles'][-5:]:  # Show last 5
            print(f"  - {article_info['title'][:60]}...")
            print(f"    URL: {article_info['url']}")
            print(f"    File: {article_info['filename']}")
    
    print()
    print("=" * 80)
    print("Test Complete!")
    print(f"Check logs at: {extractor.log_dir / 'article_ingestion.log'}")
    print(f"Check articles at: {extractor.storage_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
