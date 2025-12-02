# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Command-line interface (CLI) with comprehensive commands for ingestion, querying, and state management
- Main pipeline system (`ArticleQuerySystem`) for unified component orchestration
- State persistence functionality (save/load system states)
- Examples directory with real-world extraction examples
- Comprehensive CLI documentation in README
- Main Pipeline API documentation
- Examples documentation (examples/README.md)

### Changed

- Updated README.md with CLI usage examples
- Updated README.md with Main Pipeline documentation
- Reorganized project structure (moved test_real_extraction.py to examples/)
- Enhanced .gitignore to exclude log files and saved states
- Updated roadmap to reflect completed features
- Improved project structure documentation

### Fixed

- Markdown lint warnings in README.md
- Duplicate heading issues in documentation

## [0.1.0] - 2025-12-02

### Added

- Article extraction with newspaper3k and BeautifulSoup4
- Ollama embedding service with nomic-embed-text model
- FAISS vector store for semantic search
- RAG query interface with LangChain
- Conversation history management
- Multi-turn dialogue support
- Comprehensive test suite
- Hybrid caching (in-memory + disk)
- Batch processing capabilities
- Progress tracking for long operations

### Features

- 📰 Robust article scraping with retry logic
- 🧠 High-performance embedding generation
- 🗄️ Sub-100ms vector search
- 🤖 RAG-powered question answering
- 🔍 Semantic search over article content
- 🏗️ Modular architecture
