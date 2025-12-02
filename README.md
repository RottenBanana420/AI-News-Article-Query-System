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
  
- 🗄️ **Vector Storage**: FAISS HNSW-based vector database
  - Sub-100ms query performance
  - 768-dimensional embeddings
  - Robust metadata synchronization
  - Atomic save/load operations
  
- 🤖 **RAG Query Interface**: Conversational question answering with LangChain
  - Natural language query processing
  - Context-aware answer generation using Ollama llama3.1
  - Automatic source citation extraction
  - Multi-turn conversation support with history
  - Configurable retrieval parameters (top-k, temperature)
  
- 🔍 **Semantic Search**: Natural language querying over article content
- 🏗️ **Modular Architecture**: Clean separation of concerns for easy maintenance

## Project Structure

```text
AI-News-Article-Query-System/
├── src/
│   ├── main_pipeline.py          # Main system orchestration
│   ├── cli.py                    # Command-line interface
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
│       ├── handler.py            # RAG query processing
│       ├── rag_service.py        # RAG orchestration
│       └── conversation_manager.py # Conversation history
├── data/
│   ├── raw_articles/             # Extracted articles (JSON)
│   ├── embeddings/               # Vector embeddings
│   └── saved_states/             # Saved system states
├── tests/
│   ├── test_article_extractor.py # Article extraction tests
│   ├── test_ollama_service.py    # Embedding service tests
│   ├── test_vector_store.py      # Vector store tests
│   ├── test_rag_service.py       # RAG service tests
│   ├── test_conversation_manager.py # Conversation tests
│   ├── test_rag_integration.py   # End-to-end RAG tests
│   ├── test_main_pipeline.py     # Pipeline integration tests
│   └── test_cli.py               # CLI tests
├── examples/
│   ├── test_real_extraction.py   # Real-world extraction example
│   └── README.md                 # Examples documentation
├── logs/                         # Application logs
├── scripts/
│   ├── activate.sh               # Unix activation script
│   └── activate.bat              # Windows activation script
├── example_urls.txt              # Sample URLs for testing
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md
```

## Prerequisites

1. **pyenv** & **pyenv-virtualenv**: Python version management
   - pyenv: <https://github.com/pyenv/pyenv#installation>
   - pyenv-virtualenv: <https://github.com/pyenv/pyenv-virtualenv#installation>

2. **Ollama**: Local LLM runtime for embeddings
   - Installation: <https://ollama.ai/download>
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

```

## CLI Usage

The system provides a comprehensive command-line interface for all operations:

### Quick Start

```bash
# Ingest a single article
python -m src.cli ingest --url https://example.com/article

# Ingest articles from a file
python -m src.cli ingest --file example_urls.txt

# Search for articles
python -m src.cli query "artificial intelligence in healthcare"

# Ask a question with AI-generated answer
python -m src.cli ask "How is AI transforming healthcare?"

# View system statistics
python -m src.cli stats

# Save current state
python -m src.cli save my_state --description "Healthcare articles"

# Load a saved state
python -m src.cli load my_state

# List all saved states
python -m src.cli list-states
```

### CLI Commands

#### Ingest Articles

```bash
# Single URL
python -m src.cli ingest --url https://techcrunch.com/article

# Batch from file (with custom delay)
python -m src.cli ingest --file urls.txt --delay 2.0

# Enable verbose logging
python -m src.cli --verbose ingest --file urls.txt
```

#### Semantic Search

```bash
# Basic search
python -m src.cli query "machine learning"

# Get more results
python -m src.cli query "AI ethics" --top-k 10
```

#### RAG Question Answering

```bash
# Ask a question
python -m src.cli ask "What are the latest developments in AI?"

# Multi-turn conversation (use session ID)
python -m src.cli ask "Tell me more" --session abc123

# Disable source citations
python -m src.cli ask "Explain neural networks" --no-sources

# Retrieve more context
python -m src.cli ask "How does GPT work?" --top-k 10
```

#### State Management

```bash
# Save current state
python -m src.cli save healthcare_articles --description "Articles about AI in healthcare"

# Load a saved state
python -m src.cli load healthcare_articles

# List all saved states
python -m src.cli list-states

# Overwrite existing state
python -m src.cli save my_state --overwrite
```

#### System Statistics

```bash
# View comprehensive statistics
python -m src.cli stats
```

## Main Pipeline

The `ArticleQuerySystem` class provides a high-level integration layer that orchestrates all components:

### Python API

```python
from src.main_pipeline import ArticleQuerySystem

# Initialize the system
system = ArticleQuerySystem()

# Ingest a single article
result = system.ingest_article('https://example.com/article')
print(f"Ingested {result['chunks_created']} chunks in {result['processing_time']:.2f}s")

# Ingest from file
results = system.ingest_from_file('example_urls.txt', delay=1.0, show_progress=True)
print(f"Successfully ingested {results['successful']}/{results['total']} articles")

# Semantic search
results = system.query("artificial intelligence", top_k=5)
for result in results:
    print(f"{result['metadata']['title']}: {result['similarity']:.3f}")

# RAG question answering
answer = system.ask_question("How is AI transforming healthcare?")
print(f"Answer: {answer['answer']}")
print(f"Sources: {len(answer['sources'])}")

# Save current state
system.save_state('my_state', description='Healthcare articles')

# Load a saved state
system.load_state('my_state')

# Get statistics
stats = system.get_stats()
print(f"Total articles: {stats['total_articles']}")
print(f"Total chunks: {stats['total_chunks']}")
```

### Pipeline Features

- **Unified Interface**: Single entry point for all operations
- **Dependency Injection**: Customize components as needed
- **State Persistence**: Save and load complete system states
- **Progress Tracking**: Built-in progress indicators for batch operations
- **Error Handling**: Comprehensive error handling and logging
- **Statistics**: Detailed system metrics and cache statistics

## Component-Level Usage

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

### RAG Query Interface

```python
from src.query.handler import QueryHandler

# Initialize query handler (RAG enabled by default)
handler = QueryHandler()

# Ask a question and get an AI-generated answer with citations
result = handler.ask_question("How is AI transforming healthcare?")

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"\nSources:")
for i, source in enumerate(result['sources'], 1):
    print(f"  [{i}] {source['title']}")
    print(f"      {source['url']}")
print(f"\nResponse time: {result['response_time']:.2f}s")

# Multi-turn conversation
session_id = result['session_id']

# Follow-up question (uses conversation history)
result2 = handler.ask_question(
    "What are some specific applications?",
    session_id=session_id
)

print(f"\nFollow-up Answer: {result2['answer']}")

# Custom configuration
result3 = handler.ask_question(
    "Tell me more about machine learning in medicine",
    session_id=session_id,
    top_k=10  # Retrieve more context chunks
)
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

#### Article Extraction Tests

File: `test_article_extractor.py`

- URL validation and error handling
- Single and batch extraction
- Quality validation
- Retry logic with exponential backoff
- JSON storage and indexing

#### Embedding Service Tests

File: `test_ollama_service.py`

- Connection verification
- Model availability checking
- Text chunking (various sizes and overlaps)
- Single and batch embedding generation
- Caching (memory and disk)
- Performance benchmarking
- Error handling and timeouts

### Manual Testing

```bash
# Test article extraction (example script)
python examples/test_real_extraction.py

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
- [x] FAISS vector store implementation
- [x] RAG query interface with LangChain orchestration
- [x] Conversation history management
- [x] Multi-turn dialogue support
- [x] CLI interface
- [x] Main pipeline system integration
- [x] State persistence (save/load)
- [ ] Web API (FastAPI)
- [ ] Advanced retrieval (hybrid search, reranking)
- [ ] Streaming LLM responses
- [ ] Web UI dashboard

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
