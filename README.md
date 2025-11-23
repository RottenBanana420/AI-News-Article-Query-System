# AI News Article Query System

An AI-powered system for ingesting news articles, generating embeddings, storing them in a vector database, and querying them using semantic search with local LLMs.

## Features

- 📰 **Article Ingestion**: Scrape and parse news articles from URLs using `newspaper3k` and `BeautifulSoup4`
- 🧠 **Embedding Generation**: Generate embeddings using LangChain and Ollama (local LLM)
- 🗄️ **Vector Storage**: Store and retrieve embeddings using FAISS vector database
- 🔍 **Semantic Search**: Query articles using natural language with similarity search
- 🏗️ **Modular Architecture**: Clean separation of concerns for easy maintenance and extension

## Project Structure

```
AI-News-Article-Query-System/
├── src/
│   ├── ingestion/          # Article scraping and parsing
│   │   ├── scraper.py      # Article scraper using newspaper3k
│   │   └── parser.py       # HTML parser using BeautifulSoup4
│   ├── embeddings/         # Embedding generation
│   │   └── generator.py    # LangChain + Ollama embeddings
│   ├── storage/            # Vector database
│   │   └── vector_store.py # FAISS vector store operations
│   └── query/              # Query handling
│       └── handler.py      # Query processing and retrieval
├── data/
│   ├── raw_articles/       # Cached raw articles
│   └── embeddings/         # FAISS indexes and metadata
├── scripts/
│   ├── activate.sh         # Unix activation script
│   └── activate.bat        # Windows activation script
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md
```

## Prerequisites

Before setting up this project, ensure you have the following installed:

1. **pyenv**: Python version manager
   - Installation: https://github.com/pyenv/pyenv#installation

2. **pyenv-virtualenv**: pyenv plugin for virtual environments
   - Installation: https://github.com/pyenv/pyenv-virtualenv#installation

3. **Ollama**: Local LLM runtime (for embeddings)
   - Installation: https://ollama.ai/download
   - After installation, pull a model: `ollama pull llama2`

## Setup Instructions

### 1. Clone the Repository

```bash
cd /Users/kusaihajuri/Projects/AI-News-Article-Query-System
```

### 2. Install Python 3.10+

```bash
# List available Python versions
pyenv install --list | grep "3.10"

# Install Python 3.10.15 (or latest 3.10.x)
pyenv install 3.10.15
```

### 3. Create Virtual Environment

```bash
# Create virtual environment with Python 3.10.15
pyenv virtualenv 3.10.15 ai-news-query

# Set local environment for this project
pyenv local ai-news-query
```

The virtual environment will now activate automatically when you `cd` into this directory.

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 5. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your preferred settings
# nano .env  # or use your preferred editor
```

### 6. Verify Installation

```bash
# Test all imports
python -c "import langchain; import faiss; import bs4; import newspaper; import requests; import dotenv; import ollama; print('✓ All imports successful!')"
```

## Usage

### Basic Example

```python
from src.ingestion.scraper import ArticleScraper
from src.embeddings.generator import EmbeddingGenerator
from src.storage.vector_store import VectorStore
from src.query.handler import QueryHandler

# 1. Scrape articles
scraper = ArticleScraper()
articles = scraper.scrape_multiple([
    'https://example.com/article1',
    'https://example.com/article2'
])

# 2. Generate embeddings
generator = EmbeddingGenerator()
for article in articles:
    processed = generator.process_article(article)
    
    # 3. Store in vector database
    store = VectorStore()
    metadata = [
        {**processed['metadata'], 'chunk': chunk}
        for chunk in processed['chunks']
    ]
    store.add_embeddings(processed['embeddings'], metadata)
    store.save_index()

# 4. Query the system
handler = QueryHandler(generator, store)
results = handler.query("What are the latest developments in AI?", k=5)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity']:.2f}")
    print(f"Chunk: {result['chunk'][:200]}...")
    print("-" * 80)
```

### Using Activation Scripts

#### Unix/Linux/macOS

```bash
# Make script executable (already done)
chmod +x scripts/activate.sh

# Run activation script
./scripts/activate.sh
```

#### Windows

```cmd
# Run activation script
scripts\activate.bat
```

## Configuration

Edit `.env` to customize:

- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Model to use for embeddings (default: `llama2`)
- `FAISS_INDEX_PATH`: Path to FAISS index file
- `ARTICLE_CACHE_DIR`: Directory for cached articles
- `MAX_ARTICLES`: Maximum articles to process

## Testing

### Manual Testing

1. **Test Virtual Environment**:
   ```bash
   which python  # Should point to pyenv virtualenv
   python --version  # Should be 3.10.x
   ```

2. **Test Imports**:
   ```bash
   python -c "from src.ingestion.scraper import ArticleScraper; print('✓ Scraper OK')"
   python -c "from src.embeddings.generator import EmbeddingGenerator; print('✓ Embeddings OK')"
   python -c "from src.storage.vector_store import VectorStore; print('✓ Storage OK')"
   python -c "from src.query.handler import QueryHandler; print('✓ Query OK')"
   ```

3. **Test Ollama Connection**:
   ```bash
   python -c "import ollama; print('✓ Ollama OK')"
   ```

### Automated Testing

Run the comprehensive test script:

```bash
python -m pytest tests/  # (if tests are added)
```

## Troubleshooting

### pyenv not found

Ensure pyenv is properly initialized in your shell configuration:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

### Ollama connection errors

1. Ensure Ollama is running: `ollama serve`
2. Check if model is downloaded: `ollama list`
3. Pull model if needed: `ollama pull llama2`

### FAISS installation issues

If FAISS fails to install, try:
```bash
pip install faiss-cpu --no-cache-dir
```

### Import errors

Ensure you're in the project root directory and the virtual environment is activated:
```bash
cd /Users/kusaihajuri/Projects/AI-News-Article-Query-System
pyenv local ai-news-query
```

## Development

### Adding New Features

1. Create new modules in appropriate directories (`src/ingestion/`, `src/embeddings/`, etc.)
2. Update `requirements.txt` if new dependencies are needed
3. Add tests for new functionality
4. Update this README with usage examples

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

RAG-powered news analysis platform for intelligent article exploration and querying
