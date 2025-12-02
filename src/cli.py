"""
Command-Line Interface for AI News Article Query System

Provides user-friendly CLI commands for:
- Article ingestion (single, batch, from file)
- Semantic search
- RAG-based question answering
- System statistics
- State management (save/load/list)
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from .main_pipeline import ArticleQuerySystem


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_ingest(args):
    """Handle the ingest command."""
    system = ArticleQuerySystem()
    
    if args.url:
        # Ingest single URL
        print(f"Ingesting article from: {args.url}")
        result = system.ingest_article(args.url)
        
        if result['success']:
            print(f"✓ Successfully ingested article")
            print(f"  Article ID: {result['article_id']}")
            print(f"  Chunks created: {result['chunks_created']}")
            print(f"  Processing time: {result['processing_time']:.2f}s")
        else:
            print(f"✗ Failed to ingest article: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    elif args.file:
        # Ingest from file
        if not Path(args.file).exists():
            print(f"✗ Error: File not found: {args.file}")
            sys.exit(1)
        
        print(f"Ingesting articles from: {args.file}")
        results = system.ingest_from_file(args.file, delay=args.delay, show_progress=True)
        
        print(f"\n{'='*60}")
        print(f"Ingestion Summary:")
        print(f"  Total URLs: {results['total']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Processing time: {results['processing_time']:.2f}s")
        print(f"{'='*60}")
        
        if results['failed'] > 0:
            print("\nFailed URLs:")
            for detail in results['details']:
                if not detail['success']:
                    print(f"  - {detail['url']}: {detail.get('error', 'Unknown error')}")
    
    else:
        print("✗ Error: Either --url or --file must be specified")
        sys.exit(1)


def cmd_query(args):
    """Handle the query command."""
    system = ArticleQuerySystem()
    
    print(f"Searching for: {args.query}")
    print()
    
    results = system.query(args.query, top_k=args.top_k)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['metadata']['title']}")
        print(f"    URL: {result['metadata']['url']}")
        print(f"    Similarity: {result['similarity']:.3f}")
        print(f"    Chunk: {result['chunk'][:200]}...")
        print()


def cmd_ask(args):
    """Handle the ask command."""
    system = ArticleQuerySystem()
    
    print(f"Question: {args.question}")
    print()
    
    result = system.ask_question(
        args.question,
        session_id=args.session,
        top_k=args.top_k
    )
    
    print("Answer:")
    print(f"{result['answer']}")
    print()
    
    if not args.no_sources and result.get('sources'):
        print("Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source['title']}")
            print(f"      {source['url']}")
        print()
    
    print(f"Response time: {result['response_time']:.2f}s")
    
    if result.get('session_id'):
        print(f"Session ID: {result['session_id']}")
        print("(Use this session ID for follow-up questions)")


def cmd_stats(args):
    """Handle the stats command."""
    system = ArticleQuerySystem()
    
    stats = system.get_stats()
    
    print("="*60)
    print("System Statistics")
    print("="*60)
    print(f"Total Articles: {stats['total_articles']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Total Embeddings: {stats['total_embeddings']}")
    print()
    
    print("Vector Store:")
    vs_stats = stats['vector_store_stats']
    print(f"  Dimension: {vs_stats.get('dimension', 'N/A')}")
    print(f"  Index Type: {vs_stats.get('index_type', 'N/A')}")
    print(f"  Total Vectors: {vs_stats.get('total_vectors', 0)}")
    print()
    
    print("Embedding Cache:")
    cache_stats = stats['cache_stats']
    print(f"  Cache Size: {cache_stats.get('cache_size', 0)}")
    print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
    print(f"  Hits: {cache_stats.get('hits', 0)}")
    print(f"  Misses: {cache_stats.get('misses', 0)}")
    print("="*60)


def cmd_save(args):
    """Handle the save command."""
    system = ArticleQuerySystem()
    
    print(f"Saving state: {args.name}")
    
    result = system.save_state(
        args.name,
        description=args.description or "",
        overwrite=args.overwrite
    )
    
    if result['success']:
        print(f"✓ Successfully saved state to: {result['path']}")
    else:
        print(f"✗ Failed to save state: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_load(args):
    """Handle the load command."""
    system = ArticleQuerySystem()
    
    print(f"Loading state: {args.name}")
    
    result = system.load_state(args.name)
    
    if result['success']:
        print(f"✓ Successfully loaded state: {args.name}")
        
        # Show stats after loading
        stats = system.get_stats()
        print(f"  Articles: {stats['total_articles']}")
        print(f"  Chunks: {stats['total_chunks']}")
    else:
        print(f"✗ Failed to load state: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_list_states(args):
    """Handle the list-states command."""
    system = ArticleQuerySystem()
    
    states = system.list_states()
    
    if not states:
        print("No saved states found.")
        return
    
    print(f"Found {len(states)} saved state(s):\n")
    
    for state in states:
        print(f"Name: {state['name']}")
        print(f"  Description: {state.get('description', 'N/A')}")
        print(f"  Created: {state.get('created_at', 'N/A')}")
        print(f"  Articles: {state.get('total_articles', 0)}")
        print(f"  Chunks: {state.get('total_chunks', 0)}")
        print(f"  Size: {state.get('size_bytes', 0) / 1024:.1f} KB")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AI News Article Query System - Intelligent article ingestion and querying',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single article
  python -m src.cli ingest --url https://example.com/article

  # Ingest articles from a file
  python -m src.cli ingest --file example_urls.txt

  # Query the system
  python -m src.cli query "artificial intelligence in healthcare"

  # Ask a question
  python -m src.cli ask "How is AI transforming healthcare?"

  # Save current state
  python -m src.cli save my_state --description "Healthcare articles"

  # Load a saved state
  python -m src.cli load my_state

  # View statistics
  python -m src.cli stats
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Ingest articles from URLs'
    )
    ingest_parser.add_argument(
        '--url',
        help='Single URL to ingest'
    )
    ingest_parser.add_argument(
        '--file',
        help='File containing URLs (one per line)'
    )
    ingest_parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Search for relevant articles'
    )
    query_parser.add_argument(
        'query',
        help='Search query'
    )
    query_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    query_parser.set_defaults(func=cmd_query)
    
    # Ask command
    ask_parser = subparsers.add_parser(
        'ask',
        help='Ask a question and get an AI-generated answer'
    )
    ask_parser.add_argument(
        'question',
        help='Question to ask'
    )
    ask_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of context chunks to retrieve (default: 5)'
    )
    ask_parser.add_argument(
        '--session',
        help='Session ID for multi-turn conversation'
    )
    ask_parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Disable source citations'
    )
    ask_parser.set_defaults(func=cmd_ask)
    
    # Stats command
    stats_parser = subparsers.add_parser(
        'stats',
        help='Display system statistics'
    )
    stats_parser.set_defaults(func=cmd_stats)
    
    # Save command
    save_parser = subparsers.add_parser(
        'save',
        help='Save current state'
    )
    save_parser.add_argument(
        'name',
        help='Name for the saved state'
    )
    save_parser.add_argument(
        '--description',
        help='Optional description'
    )
    save_parser.add_argument(
        '--overwrite',
        action='store_true',
        default=True,
        help='Overwrite existing state (default: True)'
    )
    save_parser.set_defaults(func=cmd_save)
    
    # Load command
    load_parser = subparsers.add_parser(
        'load',
        help='Load a saved state'
    )
    load_parser.add_argument(
        'name',
        help='Name of the state to load'
    )
    load_parser.set_defaults(func=cmd_load)
    
    # List states command
    list_parser = subparsers.add_parser(
        'list-states',
        help='List all saved states'
    )
    list_parser.set_defaults(func=cmd_list_states)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
