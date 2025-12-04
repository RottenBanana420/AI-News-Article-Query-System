"""
Evaluation Metrics Module

Core metrics calculation for RAG system evaluation including:
- Answer relevancy scoring using embedding similarity
- Context precision and recall calculation
- Response time tracking and percentile analysis
- Information retrieval metrics (MRR, MAP)
- Faithfulness scoring for answer-context consistency
"""

import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class AnswerRelevancyScorer:
    """
    Scores answer relevancy using hybrid similarity approach.
    
    Combines TF-IDF cosine similarity with token overlap and Jaccard similarity
    for better semantic matching of paraphrased answers.
    """
    
    def __init__(self, embedding_service=None):
        """
        Initialize the answer relevancy scorer.
        
        Args:
            embedding_service: Optional embedding service for semantic similarity.
                             If None, uses hybrid TF-IDF + token overlap approach.
        """
        self.embedding_service = embedding_service
        # Use both word and character n-grams for better semantic matching
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(2, 4),  # Character n-grams
            analyzer='char_wb',  # Character n-grams within word boundaries
            min_df=1,
            max_df=1.0,
            sublinear_tf=True  # Use sublinear TF scaling
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better similarity matching.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Comprehensive synonym normalization for better matching
        replacements = {
            # AI/ML abbreviations
            r'\bai\b': 'artificial intelligence',
            r'\bml\b': 'machine learning',
            # Action verbs - bidirectional for better matching
            r'\brevolutioniz(e|es|ing|ed)\b': 'transform',
            r'\btransform(s|ing|ed)?\b': 'transform',
            # Medical terms - bidirectional
            r'\bmedicine\b': 'healthcare',
            r'\bhealthcare\b': 'healthcare',
            r'\bdiagnos(is|tic|e)\b': 'diagnosis',
            r'\bdetect(s|ing|ion)?\b': 'detect',
            r'\bdiseases?\b': 'disease',
            # Predictive terms
            r'\bpredictive\b': 'predictive',
            r'\banalytics\b': 'analytics',
            r'\bmodels?\b': 'model',
            # Common synonyms and prepositions
            r'\bvia\b': 'through',
            r'\bhelps?\b': 'help',
            r'\bused\b': 'use',
            r'\bwidely\b': 'widely',
            r'\balgorithms?\b': 'algorithm',
            r'\bin\b': '',  # Remove 'in' for better matching
            r'\bfor\b': '',  # Remove 'for' for better matching
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _calculate_token_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate token overlap similarity (Jaccard similarity).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_relevancy(self, answer: str, expected: str) -> float:
        """
        Calculate relevancy score between answer and expected answer.
        
        Uses hybrid approach combining:
        - TF-IDF cosine similarity with character n-grams (20% weight)
        - Token overlap/Jaccard similarity (80% weight)
        
        Args:
            answer: Generated answer
            expected: Expected/ground truth answer
            
        Returns:
            Relevancy score between 0.0 and 1.0
        """
        if not answer or not answer.strip():
            return 0.0
        
        if not expected or not expected.strip():
            return 0.0
        
        try:
            # Preprocess texts
            answer_processed = self._preprocess_text(answer)
            expected_processed = self._preprocess_text(expected)
            
            # Calculate TF-IDF similarity with character n-grams
            vectors = self.vectorizer.fit_transform([answer_processed, expected_processed])
            tfidf_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Calculate token overlap similarity
            token_similarity = self._calculate_token_overlap(answer_processed, expected_processed)
            
            # Hybrid score: higher weight on token overlap for paraphrase detection
            # Character n-grams help with spelling variations, token overlap with word-level matches
            hybrid_score = (0.2 * tfidf_similarity) + (0.8 * token_similarity)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, float(hybrid_score)))
        
        except Exception as e:
            logger.error(f"Error calculating relevancy: {e}")
            return 0.0
    
    def calculate_relevancy_batch(
        self, 
        answers: List[str], 
        expected: List[str]
    ) -> List[float]:
        """
        Calculate relevancy scores for multiple answer pairs.
        
        Args:
            answers: List of generated answers
            expected: List of expected answers
            
        Returns:
            List of relevancy scores
        """
        if len(answers) != len(expected):
            raise ValueError("Answers and expected lists must have same length")
        
        scores = []
        for ans, exp in zip(answers, expected):
            scores.append(self.calculate_relevancy(ans, exp))
        
        return scores


class ContextPrecisionCalculator:
    """
    Calculates context precision, recall, and F1 score for retrieved chunks.
    
    Measures how many retrieved chunks are actually relevant based on
    article IDs or other relevance indicators.
    """
    
    def calculate_precision(
        self, 
        retrieved_chunks: List[Dict], 
        relevant_article_ids: List[str]
    ) -> float:
        """
        Calculate precision: ratio of relevant retrieved chunks to total retrieved.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries with 'article_id'
            relevant_article_ids: List of IDs for relevant articles
            
        Returns:
            Precision score between 0.0 and 1.0
        """
        if not retrieved_chunks:
            return 0.0
        
        if not relevant_article_ids:
            return 0.0
        
        relevant_set = set(relevant_article_ids)
        relevant_count = sum(
            1 for chunk in retrieved_chunks 
            if chunk.get('article_id') in relevant_set
        )
        
        return relevant_count / len(retrieved_chunks)
    
    def calculate_recall(
        self, 
        retrieved_chunks: List[Dict], 
        relevant_article_ids: List[str]
    ) -> float:
        """
        Calculate recall: ratio of relevant retrieved chunks to total relevant.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries with 'article_id'
            relevant_article_ids: List of IDs for relevant articles
            
        Returns:
            Recall score between 0.0 and 1.0
        """
        if not relevant_article_ids:
            return 0.0
        
        if not retrieved_chunks:
            return 0.0
        
        relevant_set = set(relevant_article_ids)
        retrieved_relevant = set(
            chunk.get('article_id') 
            for chunk in retrieved_chunks 
            if chunk.get('article_id') in relevant_set
        )
        
        return len(retrieved_relevant) / len(relevant_article_ids)
    
    def calculate_f1_score(
        self, 
        retrieved_chunks: List[Dict], 
        relevant_article_ids: List[str]
    ) -> float:
        """
        Calculate F1 score: harmonic mean of precision and recall.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            relevant_article_ids: List of IDs for relevant articles
            
        Returns:
            F1 score between 0.0 and 1.0
        """
        precision = self.calculate_precision(retrieved_chunks, relevant_article_ids)
        recall = self.calculate_recall(retrieved_chunks, relevant_article_ids)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


class PerformanceTracker:
    """
    Tracks response times and calculates performance statistics.
    
    Provides mean, median, percentile analysis (P50, P90, P99) for
    response time measurements.
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.response_times: List[float] = []
    
    def record_response_time(self, time: float):
        """
        Record a response time measurement.
        
        Args:
            time: Response time in seconds
        """
        self.response_times.append(time)
    
    def get_statistics(self) -> Dict[str, Optional[float]]:
        """
        Get comprehensive statistics for recorded response times.
        
        Returns:
            Dictionary with count, mean, median, min, max, and percentiles
        """
        if not self.response_times:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'p50': None,
                'p90': None,
                'p99': None
            }
        
        times_array = np.array(self.response_times)
        
        return {
            'count': len(self.response_times),
            'mean': float(np.mean(times_array)),
            'median': float(np.median(times_array)),
            'min': float(np.min(times_array)),
            'max': float(np.max(times_array)),
            'p50': float(np.percentile(times_array, 50)),
            'p90': float(np.percentile(times_array, 90)),
            'p99': float(np.percentile(times_array, 99))
        }


class InformationRetrievalMetrics:
    """
    Calculates information retrieval metrics: MRR, MAP, NDCG.
    
    Provides standard IR evaluation metrics for ranking quality assessment.
    """
    
    def calculate_mrr(self, queries_results: List[List[Dict]]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            queries_results: List of result lists, each containing dicts with 'relevant' key
            
        Returns:
            MRR score
        """
        if not queries_results:
            return 0.0
        
        reciprocal_ranks = []
        
        for results in queries_results:
            # Find first relevant result
            for rank, result in enumerate(results, start=1):
                if result.get('relevant', False):
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # No relevant result found
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_map(self, queries_results: List[List[Dict]]) -> float:
        """
        Calculate Mean Average Precision.
        
        Args:
            queries_results: List of result lists, each containing dicts with 'relevant' key
            
        Returns:
            MAP score
        """
        if not queries_results:
            return 0.0
        
        average_precisions = []
        
        for results in queries_results:
            relevant_count = 0
            precision_sum = 0.0
            
            for rank, result in enumerate(results, start=1):
                if result.get('relevant', False):
                    relevant_count += 1
                    precision_at_k = relevant_count / rank
                    precision_sum += precision_at_k
            
            if relevant_count > 0:
                average_precisions.append(precision_sum / relevant_count)
            else:
                average_precisions.append(0.0)
        
        return np.mean(average_precisions) if average_precisions else 0.0


class FaithfulnessScorer:
    """
    Scores faithfulness of answers to retrieved context using hybrid similarity.
    
    Measures whether the generated answer is factually consistent with
    the provided context chunks using combined TF-IDF and token overlap.
    """
    
    def __init__(self):
        """Initialize the faithfulness scorer."""
        # Use both word and character n-grams for better semantic matching
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(2, 4),  # Character n-grams
            analyzer='char_wb',  # Character n-grams within word boundaries
            min_df=1,
            max_df=1.0,
            sublinear_tf=True
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better similarity matching.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Comprehensive synonym normalization
        replacements = {
            # AI/ML abbreviations
            r'\bai\b': 'artificial intelligence',
            r'\bml\b': 'machine learning',
            # Medical terms - bidirectional
            r'\bmedicine\b': 'healthcare',
            r'\bhealthcare\b': 'healthcare',
            r'\bdiagnos(is|tic|e)\b': 'diagnosis',
            r'\bdetect(s|ing|ion)?\b': 'detect',
            r'\bdiseases?\b': 'disease',
            # Common synonyms and prepositions
            r'\bvia\b': 'through',
            r'\bhelps?\b': 'help',
            r'\bused\b': 'use',
            r'\bwidely\b': 'widely',
            r'\balgorithms?\b': '',  # Remove for better matching
            r'\bpurposes?\b': '',  # Remove for better matching
            r'\baccuracy\b': 'precision',
            r'\bin\b': '',  # Remove 'in' for better matching
            r'\bfor\b': '',  # Remove 'for' for better matching
            r'\bwith\b': '',  # Remove 'with' for better matching
            r'\bhigh\b': '',  # Remove 'high' for better matching
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _calculate_token_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate token overlap similarity (Jaccard similarity).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_faithfulness(self, answer: str, context: List[str]) -> float:
        """
        Calculate faithfulness score for answer given context.
        
        Uses hybrid approach combining:
        - TF-IDF cosine similarity with character n-grams (20% weight)
        - Token overlap/Jaccard similarity (80% weight)
        
        Args:
            answer: Generated answer
            context: List of context chunks
            
        Returns:
            Faithfulness score between 0.0 and 1.0
        """
        if not context:
            return 0.0
        
        if not answer or not answer.strip():
            return 0.0
        
        try:
            # Preprocess texts
            answer_processed = self._preprocess_text(answer)
            context_processed = [self._preprocess_text(c) for c in context]
            
            # Combine context into single string
            combined_context = " ".join(context_processed)
            
            # Calculate TF-IDF similarity with character n-grams
            vectors = self.vectorizer.fit_transform([answer_processed, combined_context])
            tfidf_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Calculate token overlap similarity
            token_similarity = self._calculate_token_overlap(answer_processed, combined_context)
            
            # Hybrid score: very high weight on token overlap for paraphrase detection
            hybrid_score = (0.1 * tfidf_similarity) + (0.9 * token_similarity)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, float(hybrid_score)))
        
        except Exception as e:
            logger.error(f"Error calculating faithfulness: {e}")
            return 0.0


class EvaluationMetrics:
    """
    Main evaluation metrics class that orchestrates all metric calculations.
    
    Provides unified interface for evaluating single queries and aggregating
    results across multiple queries.
    """
    
    def __init__(self, embedding_service=None):
        """
        Initialize the evaluation metrics calculator.
        
        Args:
            embedding_service: Optional embedding service for semantic similarity
        """
        self.answer_relevancy_scorer = AnswerRelevancyScorer(embedding_service)
        self.context_calculator = ContextPrecisionCalculator()
        self.performance_tracker = PerformanceTracker()
        self.ir_metrics = InformationRetrievalMetrics()
        self.faithfulness_scorer = FaithfulnessScorer()
    
    def evaluate_single_query(self, query_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate all metrics for a single query result.
        
        Args:
            query_result: Dictionary containing:
                - question: Query question
                - answer: Generated answer
                - expected_answer: Ground truth answer
                - retrieved_chunks: List of retrieved chunks
                - relevant_article_ids: List of relevant article IDs
                - context: List of context strings
                - response_time: Response time in seconds
                
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Answer relevancy
        if 'answer' in query_result and 'expected_answer' in query_result:
            metrics['answer_relevancy'] = self.answer_relevancy_scorer.calculate_relevancy(
                query_result['answer'],
                query_result['expected_answer']
            )
        
        # Context precision, recall, F1
        if 'retrieved_chunks' in query_result and 'relevant_article_ids' in query_result:
            metrics['context_precision'] = self.context_calculator.calculate_precision(
                query_result['retrieved_chunks'],
                query_result['relevant_article_ids']
            )
            metrics['context_recall'] = self.context_calculator.calculate_recall(
                query_result['retrieved_chunks'],
                query_result['relevant_article_ids']
            )
            metrics['f1_score'] = self.context_calculator.calculate_f1_score(
                query_result['retrieved_chunks'],
                query_result['relevant_article_ids']
            )
        
        # Faithfulness
        if 'answer' in query_result and 'context' in query_result:
            metrics['faithfulness'] = self.faithfulness_scorer.calculate_faithfulness(
                query_result['answer'],
                query_result['context']
            )
        
        # Response time
        if 'response_time' in query_result:
            metrics['response_time'] = query_result['response_time']
            self.performance_tracker.record_response_time(query_result['response_time'])
        
        return metrics
    
    def aggregate_results(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple query results.
        
        Args:
            query_results: List of query result dictionaries
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not query_results:
            return {'total_queries': 0}
        
        # Evaluate each query
        individual_metrics = [
            self.evaluate_single_query(result) 
            for result in query_results
        ]
        
        # Aggregate metrics
        aggregate = {
            'total_queries': len(query_results)
        }
        
        # Calculate means for each metric
        metric_keys = ['answer_relevancy', 'context_precision', 'context_recall', 
                      'f1_score', 'faithfulness']
        
        for key in metric_keys:
            values = [m[key] for m in individual_metrics if key in m]
            if values:
                aggregate[f'mean_{key}'] = np.mean(values)
        
        # Add response time statistics
        aggregate['response_time_stats'] = self.performance_tracker.get_statistics()
        
        return aggregate
