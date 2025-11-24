# AI News Article Query System

A robust RAG-powered system for ingesting news articles, generating embeddings using local LLMs, and enabling intelligent semantic search over article content.

## Features

- 📰 **Article Extraction**: Robust article scraping with `newspaper3k` and `BeautifulSoup4`
  - Comprehensive error handling with retry logic
  - Quality validation and text normalization
  - Batch processing with configurable delays
  - Structured JSON storage with metadata indexing
  
- 🧠 **Embedding Generation**: High-performance embedding service using Ollama
  - Direct integration with `nomic-embed-text` model (768 dimensions)
  - Intelligent text chunking with configurable overlap
  - Hybrid caching (in-memory + disk persistence)
  - Batch processing with progress tracking
  - Connection and model verification
  
- 🗄️ **Vector Storage**: FAISS-based vector database (planned)
- 🔍 **Semantic Search**: Natural language querying (planned)
- 🏗️ **Modular Architecture**: Clean separation of concerns for easy maintenance

## Project Structure

```
AI-News-Article-Query-System/
├── src/
│   ├── ingestion/
│   │   ├── article_extractor.py  # Comprehensive article extraction
│   │   ├── scraper.py            # newspaper3k wrapper
│   │   └── parser.py             # BeautifulSoup4 parser
│   ├── embeddings/
│   │   ├── ollama_service.py     # Ollama embedding service
│   │   └── generator.py          # LangChain integration
│   ├── storage/
│   │   └── vector_store.py       # FAISS vector operations
│   └── query/
│       └── handler.py            # Query processing
├── data/
│   ├── raw_articles/             # Extracted articles (JSON)
│   └── embeddings/
│       └── cache/                # Embedding cache files
├── tests/
│   ├── test_article_extractor.py # Article extraction tests
│   └── test_ollama_service.py    # Embedding service tests
├── logs/                         # Application logs
├── scripts/
│   ├── activate.sh               # Unix activation script
│   └── activate.bat              # Windows activation script
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md
```

## Prerequisites

1. **pyenv** & **pyenv-virtualenv**: Python version management
   - pyenv: https://github.com/pyenv/pyenv#installation
   - pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv#installation

2. **Ollama**: Local LLM runtime for embeddings
   - Installation: https://ollama.ai/download
   - **Required model**: `ollama pull nomic-embed-text`

## Setup Instructions

### 1. Navigate to Project Directory

```bash
cd /Users/kusaihajuri/Projects/AI-News-Article-Query-System
```

### 2. Install Python 3.10+

```bash
# List available versions
pyenv install --list | grep "3.10"

# Install Python 3.10.15 (or latest 3.10.x)
pyenv install 3.10.15
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
pyenv virtualenv 3.10.15 ai-news-query

# Set local environment (auto-activates on cd)
pyenv local ai-news-query
```

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 5. Setup Ollama

```bash
# Ensure Ollama is running
ollama serve

# Pull the embedding model (required)
ollama pull nomic-embed-text

# Verify model is available
ollama list
```

### 6. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

### 7. Verify Installation

```bash
# Test imports
python -c "import langchain; import faiss; import bs4; import newspaper; import requests; import dotenv; import ollama; print('✓ All imports successful!')"

# Test Ollama connection
python -c "from src.embeddings.ollama_service import OllamaEmbeddingService; s = OllamaEmbeddingService(); s.verify_connection(); s.verify_model_available(); print('✓ Ollama configured correctly!')"
```

## Usage

### Article Extraction

```python
from src.ingestion.article_extractor import ArticleExtractor

# Initialize extractor
extractor = ArticleExtractor(
    storage_dir='data/raw_articles',
    timeout=30,
    max_retries=3
)

# Extract single article
article_data = extractor.extract_article('https://example.com/article')

# Extract and save
file_path = extractor.extract_and_save('https://example.com/article')

# Batch processing
urls = [
    'https://example.com/article1',
    'https://example.com/article2',
    'https://example.com/article3'
]
saved_paths = extractor.extract_and_save_batch(urls, delay=1.0)

# View extraction index
index = extractor.get_index()
print(f"Extracted {len(index)} articles")
```

### Embedding Generation

```python
from src.embeddings.ollama_service import OllamaEmbeddingService
import numpy as np

# Initialize service
service = OllamaEmbeddingService(
    model="nomic-embed-text",
    chunk_size=1000,
    chunk_overlap=200,
    batch_size=10,
    enable_disk_cache=True
)

# Verify connection
service.verify_connection()
service.verify_model_available()

# Generate single embedding
text = "This is a test sentence for embedding generation."
embedding = service.generate_embedding(text)
print(f"Embedding shape: {embedding.shape}")  # (768,)

# Process article with chunking
article_text = """
Your long article text here...
"""

result = service.process_article(
    article_text,
    use_cache=True,
    show_progress=True
)

print(f"Chunks: {result['num_chunks']}")
print(f"Embeddings: {len(result['embeddings'])}")

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = service.generate_embeddings_batch(
    texts,
    use_cache=True,
    show_progress=True
)

# Check cache statistics
stats = service.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['cache_size']}")
```

### Complete Workflow

```python
from src.ingestion.article_extractor import ArticleExtractor
from src.embeddings.ollama_service import OllamaEmbeddingService
import json

# 1. Extract articles
extractor = ArticleExtractor()
urls = ['https://example.com/article1', 'https://example.com/article2']
saved_paths = extractor.extract_and_save_batch(urls)

# 2. Initialize embedding service
embedding_service = OllamaEmbeddingService(
    model="nomic-embed-text",
    enable_disk_cache=True
)

# 3. Process each article
for article_path in saved_paths:
    with open(article_path, 'r') as f:
        article = json.load(f)
    
    # Generate embeddings for article text
    result = embedding_service.process_article(
        article['text'],
        use_cache=True,
        show_progress=True
    )
    
    print(f"Processed: {article['title']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Embeddings: {len(result['embeddings'])}")

# 4. View statistics
print(f"\nCache statistics: {embedding_service.get_cache_stats()}")
```

## Configuration

Environment variables in `.env`:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text  # Required for this project

# Vector Store (planned)
FAISS_INDEX_PATH=data/embeddings/articles.index

# Article Ingestion
MAX_ARTICLES=100
ARTICLE_CACHE_DIR=data/raw_articles

# Optional: LangChain
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
```

## Testing

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_ollama_service.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

**Article Extraction Tests** (`test_article_extractor.py`):
- URL validation and error handling
- Single and batch extraction
- Quality validation
- Retry logic with exponential backoff
- JSON storage and indexing

**Embedding Service Tests** (`test_ollama_service.py`):
- Connection verification
- Model availability checking
- Text chunking (various sizes and overlaps)
- Single and batch embedding generation
- Caching (memory and disk)
- Performance benchmarking
- Error handling and timeouts

### Manual Testing

```bash
# Test article extraction
python test_real_extraction.py

# Test embedding service
python -c "
from src.embeddings.ollama_service import OllamaEmbeddingService
service = OllamaEmbeddingService()
service.verify_connection()
service.verify_model_available()
emb = service.generate_embedding('test')
print(f'✓ Generated embedding with shape: {emb.shape}')
"
```

## Troubleshooting

### pyenv Issues

Ensure pyenv is initialized in your shell:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

### Ollama Issues

**Connection errors:**
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

**Model not found:**
```bash
# List available models
ollama list

# Pull required model
ollama pull nomic-embed-text
```

**Timeout errors:**
- Increase timeout in service initialization: `OllamaEmbeddingService(timeout=60)`
- Reduce chunk size for very long texts: `chunk_size=500`

### FAISS Installation

If FAISS fails to install:
```bash
pip install faiss-cpu --no-cache-dir
```

### Import Errors

Ensure you're in the project root with activated environment:
```bash
cd /Users/kusaihajuri/Projects/AI-News-Article-Query-System
pyenv local ai-news-query
which python  # Should show pyenv path
```

## Performance

### Embedding Generation Benchmarks

- **Single embedding**: ~0.1-0.5s (depending on text length)
- **Batch processing**: ~2-10 embeddings/s (without cache)
- **Cache speedup**: 50-100x faster for cached embeddings
- **Embedding dimensions**: 768 (nomic-embed-text)

### Optimization Tips

1. **Enable caching** for repeated texts
2. **Adjust batch size** based on available memory
3. **Tune chunk size** for your use case (default: 1000 chars)
4. **Use disk cache** for persistence across sessions

## Development

### Adding New Features

1. Create modules in appropriate `src/` subdirectories
2. Add comprehensive tests in `tests/`
3. Update `requirements.txt` if adding dependencies
4. Document usage in this README

### Code Style

- Follow **PEP 8** guidelines
- Use **type hints** for all function signatures
- Add **docstrings** to all classes and functions
- Include **error handling** for external dependencies

### Running Tests During Development

```bash
# Run tests in watch mode
pytest-watch tests/

# Run specific test class
pytest tests/test_ollama_service.py::TestCaching -v

# Run with debugging
pytest tests/ -v --pdb
```

## Roadmap

- [x] Article extraction with error handling
- [x] Ollama embedding service with caching
- [x] Comprehensive test suite
- [ ] FAISS vector store implementation
- [ ] Query handler with semantic search
- [ ] CLI interface
- [ ] Web API (FastAPI)
- [ ] Batch article processing pipeline

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

**RAG-powered news analysis platform for intelligent article exploration and querying**
