"""
RAG Service for Question Answering with Context Retrieval

Orchestrates the complete RAG pipeline:
1. Query embedding generation
2. Context retrieval from vector store
3. Prompt construction with context and history
4. LLM-based answer generation
5. Citation extraction and formatting
"""

import time
import re
from typing import List, Dict, Optional, Any
import numpy as np
from langchain_ollama import ChatOllama

from ..embeddings.ollama_service import OllamaEmbeddingService
from ..storage.vector_store import VectorStore


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service for intelligent question answering.
    
    Combines semantic search over article embeddings with LLM-based answer generation
    to provide accurate, context-aware responses with proper source citations.
    """
    
    def __init__(
        self,
        embedding_service: OllamaEmbeddingService,
        vector_store: VectorStore,
        llm_model: str = "llama3.1:latest",
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the RAG service.
        
        Args:
            embedding_service: Service for generating query embeddings
            vector_store: Vector store for semantic search
            llm_model: Ollama model name for answer generation
            top_k: Default number of context chunks to retrieve
            temperature: LLM temperature (0.0-1.0, higher = more creative)
            max_tokens: Maximum tokens in generated answer
            ollama_base_url: Base URL for Ollama service
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=llm_model,
            temperature=temperature,
            base_url=ollama_base_url,
            num_predict=max_tokens
        )
    
    def _retrieve_context(
        self,
        query_embedding,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks from vector store.
        
        Args:
            query_embedding: Query embedding vector (list or ndarray)
            top_k: Number of chunks to retrieve
            
        Returns:
            List of context dictionaries with chunk text and metadata
        """
        # Convert to list if ndarray
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            k=top_k
        )
        
        # Format results
        context_chunks = []
        for distance, metadata in results:
            context_chunks.append({
                'chunk': metadata.get('chunk', ''),
                'title': metadata.get('title', 'Unknown'),
                'url': metadata.get('url', ''),
                'article_id': metadata.get('article_id', ''),
                'distance': distance
            })
        
        return context_chunks
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks for inclusion in prompt.
        
        Args:
            chunks: List of context chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        formatted_parts = []
        for i, chunk in enumerate(chunks, 1):
            formatted_parts.append(
                f"[{i}] {chunk['title']}\n{chunk['chunk']}\n"
            )
        
        return "\n".join(formatted_parts)
    
    def _build_prompt(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            question: User's question
            context: Retrieved context chunks
            history: Conversation history
            
        Returns:
            Complete prompt string
        """
        # System instructions
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context from news articles.

IMPORTANT INSTRUCTIONS:
1. Answer questions using ONLY the information provided in the context below
2. Cite your sources by referencing the article numbers in brackets, e.g., [1], [2]
3. If the context does not contain relevant information to answer the question, clearly state: "I don't have information about that in the provided context."
4. Be concise but comprehensive in your answers
5. Do not make up information or use knowledge outside the provided context
6. If you cannot answer based on the context, say "I don't have information about that in the provided context" """
        
        # Build conversation history section
        history_text = ""
        if history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n"
            for turn in history:
                role = turn['role'].capitalize()
                content = turn['content']
                history_text += f"{role}: {content}\n"
        
        # Build context section
        context_text = ""
        if context:
            context_text = f"\n\nCONTEXT FROM ARTICLES:\n{context}"
        else:
            context_text = "\n\nCONTEXT: No relevant articles found."
        
        # Combine all parts
        full_prompt = f"""{system_prompt}{history_text}{context_text}

QUESTION: {question}

ANSWER:"""
        
        return full_prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            prompt: Complete prompt with context and question
            
        Returns:
            Generated answer text
            
        Raises:
            Exception: If LLM generation fails
        """
        try:
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        
        except Exception as e:
            raise Exception(f"Error generating answer with LLM: {str(e)}")
    
    def _extract_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract citations from answer and match to source articles.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: Context chunks that were retrieved
            
        Returns:
            List of citation dictionaries with title and URL
        """
        citations = []
        seen_urls = set()
        
        # Find citation numbers in answer (e.g., [1], [2])
        citation_pattern = r'\[(\d+)\]'
        cited_numbers = set(re.findall(citation_pattern, answer))
        
        # Match citation numbers to chunks
        for num_str in cited_numbers:
            num = int(num_str)
            if 1 <= num <= len(retrieved_chunks):
                chunk = retrieved_chunks[num - 1]
                url = chunk['url']
                
                # Avoid duplicates
                if url and url not in seen_urls:
                    citations.append({
                        'title': chunk['title'],
                        'url': url
                    })
                    seen_urls.add(url)
        
        # If no explicit citations found, include all retrieved sources
        if not citations:
            for chunk in retrieved_chunks:
                url = chunk['url']
                if url and url not in seen_urls:
                    citations.append({
                        'title': chunk['title'],
                        'url': url
                    })
                    seen_urls.add(url)
        
        return citations
    
    def query(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a question and generate an answer with citations.
        
        Args:
            question: User's question
            conversation_history: Previous conversation turns
            top_k: Number of context chunks to retrieve (overrides default)
            
        Returns:
            Dictionary with:
                - question: Original question
                - answer: Generated answer
                - sources: List of cited sources
                - response_time: Time taken to generate response
                
        Raises:
            ValueError: If question is empty
        """
        # Validate input
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        start_time = time.time()
        
        # Use provided top_k or default
        k = top_k if top_k is not None else self.top_k
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(question)
        
        # Step 2: Retrieve relevant context
        retrieved_chunks = self._retrieve_context(query_embedding, k)
        
        # Step 3: Format context for prompt
        formatted_context = self._format_context(retrieved_chunks)
        
        # Step 4: Build complete prompt
        prompt = self._build_prompt(
            question,
            formatted_context,
            conversation_history or []
        )
        
        # Step 5: Generate answer
        answer = self._generate_answer(prompt)
        
        # Step 6: Extract citations
        citations = self._extract_citations(answer, retrieved_chunks)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        return {
            'question': question,
            'answer': answer,
            'sources': citations,
            'response_time': response_time
        }
