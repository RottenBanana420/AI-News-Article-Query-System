# Examples

This directory contains example scripts demonstrating how to use the AI News Article Query System.

## Available Examples

### test_real_extraction.py

A real-world test script for article extraction that demonstrates:

- Single article extraction
- Batch article extraction
- Error handling for invalid URLs
- Article indexing

**Usage:**

```bash
# Run from project root
python examples/test_real_extraction.py
```

This script will:

1. Extract articles from real news sources (TechCrunch, Wired, The Verge)
2. Save extracted articles to `data/raw_articles/`
3. Display extraction statistics
4. Test error handling with invalid URLs

**Note:** The script uses real URLs that may change over time. Update the URLs in the script if needed.

## Creating Your Own Examples

When creating new example scripts:

1. Add them to this `examples/` directory
2. Include clear documentation in the script
3. Update this README with usage instructions
4. Use the main pipeline API for comprehensive examples:

```python
from src.main_pipeline import ArticleQuerySystem

system = ArticleQuerySystem()
# Your example code here
```
