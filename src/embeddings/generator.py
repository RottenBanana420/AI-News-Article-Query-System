"""
Embedding Generator

Generates embeddings for text using LangChain and Ollama.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class EmbeddingGenerator:
    """Generates embeddings for text content."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model: Ollama model name (default: from .env)
            base_url: Ollama base URL (default: from .env)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.model = model or os.getenv('OLLAMA_MODEL', 'llama2')
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            return []
    
    def process_article(self, article: Dict[str, str]) -> Dict[str, any]:
        """
        Process an article: split into chunks and generate embeddings.
        
        Args:
            article: Article dictionary with 'text' field
            
        Returns:
            Dictionary with chunks and embeddings
        """
        text = article.get('text', '')
        if not text:
            return {'chunks': [], 'embeddings': []}
        
        # Split text into chunks
        chunks = self.split_text(text)
        
        # Generate embeddings for chunks
        embeddings = self.generate_embeddings(chunks)
        
        return {
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': {
                'title': article.get('title'),
                'url': article.get('url'),
                'authors': article.get('authors'),
                'publish_date': article.get('publish_date')
            }
        }
