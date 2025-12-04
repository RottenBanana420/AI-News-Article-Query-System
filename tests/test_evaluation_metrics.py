"""
Tests for Evaluation Metrics Module

Comprehensive tests for all evaluation metrics including answer relevancy,
context precision, response time tracking, and information retrieval metrics.
Following TDD principles - these tests are written first and should fail initially.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import List, Dict


class TestAnswerRelevancyScorer:
    """Test answer relevancy scoring using embedding similarity."""
    
    def test_perfect_match_returns_high_score(self):
        """Test that identical answers return near-perfect relevancy score."""
        from src.evaluation.evaluation_metrics import AnswerRelevancyScorer
        
        scorer = AnswerRelevancyScorer()
        
        answer = "Artificial intelligence is transforming healthcare through predictive analytics."
        expected = "Artificial intelligence is transforming healthcare through predictive analytics."
        
        score = scorer.calculate_relevancy(answer, expected)
        
        assert score >= 0.95, "Identical answers should have relevancy >= 0.95"
        assert score <= 1.0, "Relevancy score should not exceed 1.0"
    
    def test_similar_answers_return_high_score(self):
        """Test that semantically similar answers return high relevancy score."""
        from src.evaluation.evaluation_metrics import AnswerRelevancyScorer
        
        scorer = AnswerRelevancyScorer()
        
        answer = "AI is revolutionizing medicine via predictive models."
        expected = "Artificial intelligence is transforming healthcare through predictive analytics."
        
        score = scorer.calculate_relevancy(answer, expected)
        
        assert score >= 0.70, "Similar answers should have relevancy >= 0.70"
        assert score < 0.95, "Similar but not identical answers should be < 0.95"
    
    def test_unrelated_answers_return_low_score(self):
        """Test that unrelated answers return low relevancy score."""
        from src.evaluation.evaluation_metrics import AnswerRelevancyScorer
        
        scorer = AnswerRelevancyScorer()
        
        answer = "The weather is sunny today."
        expected = "Artificial intelligence is transforming healthcare."
        
        score = scorer.calculate_relevancy(answer, expected)
        
        assert score < 0.30, "Unrelated answers should have relevancy < 0.30"
    
    def test_empty_answer_returns_zero_score(self):
        """Test that empty answers return zero relevancy."""
        from src.evaluation.evaluation_metrics import AnswerRelevancyScorer
        
        scorer = AnswerRelevancyScorer()
        
        score = scorer.calculate_relevancy("", "Some expected answer")
        
        assert score == 0.0, "Empty answer should have zero relevancy"
    
    def test_batch_scoring(self):
        """Test batch scoring of multiple answer pairs."""
        from src.evaluation.evaluation_metrics import AnswerRelevancyScorer
        
        scorer = AnswerRelevancyScorer()
        
        answers = [
            "AI transforms healthcare",
            "Machine learning detects diseases",
            "Weather is sunny"
        ]
        expected = [
            "AI transforms healthcare",
            "ML helps in disease detection",
            "AI in medicine"
        ]
        
        scores = scorer.calculate_relevancy_batch(answers, expected)
        
        assert len(scores) == 3
        assert scores[0] >= 0.95  # Perfect match
        assert scores[1] >= 0.70  # Similar
        assert scores[2] < 0.30   # Unrelated


class TestContextPrecisionCalculator:
    """Test context precision calculation for retrieved chunks."""
    
    def test_all_relevant_chunks_returns_perfect_precision(self):
        """Test that all relevant chunks return precision of 1.0."""
        from src.evaluation.evaluation_metrics import ContextPrecisionCalculator
        
        calculator = ContextPrecisionCalculator()
        
        retrieved_chunks = [
            {"chunk": "AI in healthcare", "article_id": "art1"},
            {"chunk": "ML for diagnosis", "article_id": "art2"}
        ]
        relevant_article_ids = ["art1", "art2"]
        
        precision = calculator.calculate_precision(retrieved_chunks, relevant_article_ids)
        
        assert precision == 1.0, "All relevant chunks should give precision 1.0"
    
    def test_no_relevant_chunks_returns_zero_precision(self):
        """Test that no relevant chunks return precision of 0.0."""
        from src.evaluation.evaluation_metrics import ContextPrecisionCalculator
        
        calculator = ContextPrecisionCalculator()
        
        retrieved_chunks = [
            {"chunk": "Weather forecast", "article_id": "art1"},
            {"chunk": "Sports news", "article_id": "art2"}
        ]
        relevant_article_ids = ["art3", "art4"]
        
        precision = calculator.calculate_precision(retrieved_chunks, relevant_article_ids)
        
        assert precision == 0.0, "No relevant chunks should give precision 0.0"
    
    def test_partial_relevance_returns_correct_precision(self):
        """Test that partial relevance returns correct precision ratio."""
        from src.evaluation.evaluation_metrics import ContextPrecisionCalculator
        
        calculator = ContextPrecisionCalculator()
        
        retrieved_chunks = [
            {"chunk": "AI in healthcare", "article_id": "art1"},
            {"chunk": "Weather forecast", "article_id": "art2"},
            {"chunk": "ML diagnosis", "article_id": "art3"}
        ]
        relevant_article_ids = ["art1", "art3"]
        
        precision = calculator.calculate_precision(retrieved_chunks, relevant_article_ids)
        
        assert precision == pytest.approx(2/3, abs=0.01), "Should return 2/3 precision"
    
    def test_empty_retrieved_chunks_returns_zero(self):
        """Test that empty retrieved chunks return 0.0."""
        from src.evaluation.evaluation_metrics import ContextPrecisionCalculator
        
        calculator = ContextPrecisionCalculator()
        
        precision = calculator.calculate_precision([], ["art1"])
        
        assert precision == 0.0, "Empty retrieved chunks should return 0.0"
    
    def test_calculate_recall(self):
        """Test recall calculation for retrieved chunks."""
        from src.evaluation.evaluation_metrics import ContextPrecisionCalculator
        
        calculator = ContextPrecisionCalculator()
        
        retrieved_chunks = [
            {"chunk": "AI in healthcare", "article_id": "art1"},
            {"chunk": "ML diagnosis", "article_id": "art2"}
        ]
        relevant_article_ids = ["art1", "art2", "art3"]
        
        recall = calculator.calculate_recall(retrieved_chunks, relevant_article_ids)
        
        assert recall == pytest.approx(2/3, abs=0.01), "Should return 2/3 recall"
    
    def test_calculate_f1_score(self):
        """Test F1 score calculation."""
        from src.evaluation.evaluation_metrics import ContextPrecisionCalculator
        
        calculator = ContextPrecisionCalculator()
        
        retrieved_chunks = [
            {"chunk": "AI in healthcare", "article_id": "art1"},
            {"chunk": "ML diagnosis", "article_id": "art2"},
            {"chunk": "Weather", "article_id": "art3"}
        ]
        relevant_article_ids = ["art1", "art2", "art4"]
        
        f1 = calculator.calculate_f1_score(retrieved_chunks, relevant_article_ids)
        
        # Precision = 2/3, Recall = 2/3, F1 = 2/3
        assert f1 == pytest.approx(2/3, abs=0.01), "Should return correct F1 score"


class TestPerformanceTracker:
    """Test response time and performance tracking."""
    
    def test_track_single_response_time(self):
        """Test tracking a single response time."""
        from src.evaluation.evaluation_metrics import PerformanceTracker
        
        tracker = PerformanceTracker()
        
        tracker.record_response_time(0.5)
        
        stats = tracker.get_statistics()
        
        assert stats['count'] == 1
        assert stats['mean'] == 0.5
        assert stats['median'] == 0.5
        assert stats['min'] == 0.5
        assert stats['max'] == 0.5
    
    def test_track_multiple_response_times(self):
        """Test tracking multiple response times."""
        from src.evaluation.evaluation_metrics import PerformanceTracker
        
        tracker = PerformanceTracker()
        
        times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for t in times:
            tracker.record_response_time(t)
        
        stats = tracker.get_statistics()
        
        assert stats['count'] == 10
        assert stats['mean'] == pytest.approx(0.55, abs=0.01)
        assert stats['median'] == pytest.approx(0.55, abs=0.01)
        assert stats['min'] == 0.1
        assert stats['max'] == 1.0
    
    def test_percentile_calculations(self):
        """Test percentile calculations (P50, P90, P99)."""
        from src.evaluation.evaluation_metrics import PerformanceTracker
        
        tracker = PerformanceTracker()
        
        # Add 100 values from 0.01 to 1.00
        for i in range(1, 101):
            tracker.record_response_time(i / 100)
        
        stats = tracker.get_statistics()
        
        assert 'p50' in stats
        assert 'p90' in stats
        assert 'p99' in stats
        
        assert stats['p50'] == pytest.approx(0.50, abs=0.02)
        assert stats['p90'] == pytest.approx(0.90, abs=0.02)
        assert stats['p99'] == pytest.approx(0.99, abs=0.02)
    
    def test_empty_tracker_returns_none_stats(self):
        """Test that empty tracker returns None for statistics."""
        from src.evaluation.evaluation_metrics import PerformanceTracker
        
        tracker = PerformanceTracker()
        
        stats = tracker.get_statistics()
        
        assert stats['count'] == 0
        assert stats['mean'] is None
        assert stats['median'] is None


class TestInformationRetrievalMetrics:
    """Test IR metrics: MRR, MAP, NDCG."""
    
    def test_mean_reciprocal_rank_perfect(self):
        """Test MRR when first result is always relevant."""
        from src.evaluation.evaluation_metrics import InformationRetrievalMetrics
        
        ir_metrics = InformationRetrievalMetrics()
        
        # Each query has relevant result at rank 1
        queries_results = [
            [{"article_id": "art1", "relevant": True}, {"article_id": "art2", "relevant": False}],
            [{"article_id": "art3", "relevant": True}, {"article_id": "art4", "relevant": False}],
        ]
        
        mrr = ir_metrics.calculate_mrr(queries_results)
        
        assert mrr == 1.0, "MRR should be 1.0 when first result is always relevant"
    
    def test_mean_reciprocal_rank_varied(self):
        """Test MRR with varied relevant positions."""
        from src.evaluation.evaluation_metrics import InformationRetrievalMetrics
        
        ir_metrics = InformationRetrievalMetrics()
        
        # Relevant at positions: 1, 2, 3
        queries_results = [
            [{"article_id": "art1", "relevant": True}],  # RR = 1
            [{"article_id": "art2", "relevant": False}, {"article_id": "art3", "relevant": True}],  # RR = 1/2
            [{"article_id": "art4", "relevant": False}, {"article_id": "art5", "relevant": False}, 
             {"article_id": "art6", "relevant": True}],  # RR = 1/3
        ]
        
        mrr = ir_metrics.calculate_mrr(queries_results)
        
        # MRR = (1 + 0.5 + 0.333) / 3 = 0.611
        assert mrr == pytest.approx(0.611, abs=0.01)
    
    def test_mean_average_precision(self):
        """Test MAP calculation."""
        from src.evaluation.evaluation_metrics import InformationRetrievalMetrics
        
        ir_metrics = InformationRetrievalMetrics()
        
        # Query with 2 relevant documents at positions 1 and 3
        queries_results = [
            [
                {"article_id": "art1", "relevant": True},   # Precision@1 = 1/1
                {"article_id": "art2", "relevant": False},
                {"article_id": "art3", "relevant": True},   # Precision@3 = 2/3
            ]
        ]
        
        map_score = ir_metrics.calculate_map(queries_results)
        
        # AP = (1.0 + 0.667) / 2 = 0.833
        assert map_score == pytest.approx(0.833, abs=0.01)
    
    def test_no_relevant_results_returns_zero_mrr(self):
        """Test that no relevant results return MRR of 0."""
        from src.evaluation.evaluation_metrics import InformationRetrievalMetrics
        
        ir_metrics = InformationRetrievalMetrics()
        
        queries_results = [
            [{"article_id": "art1", "relevant": False}, {"article_id": "art2", "relevant": False}]
        ]
        
        mrr = ir_metrics.calculate_mrr(queries_results)
        
        assert mrr == 0.0


class TestFaithfulnessScorer:
    """Test faithfulness scoring for answer-context consistency."""
    
    def test_faithful_answer_returns_high_score(self):
        """Test that answers supported by context return high faithfulness."""
        from src.evaluation.evaluation_metrics import FaithfulnessScorer
        
        scorer = FaithfulnessScorer()
        
        answer = "AI is used in healthcare for diagnosis. Machine learning helps detect diseases."
        context = [
            "Artificial intelligence is widely used in healthcare for diagnostic purposes.",
            "Machine learning algorithms can detect diseases with high accuracy."
        ]
        
        score = scorer.calculate_faithfulness(answer, context)
        
        assert score >= 0.80, "Faithful answer should have score >= 0.80"
    
    def test_unfaithful_answer_returns_low_score(self):
        """Test that answers not supported by context return low faithfulness."""
        from src.evaluation.evaluation_metrics import FaithfulnessScorer
        
        scorer = FaithfulnessScorer()
        
        answer = "AI can cure cancer completely. It has 100% success rate."
        context = [
            "AI assists in cancer detection and treatment planning.",
            "Machine learning shows promising results in oncology."
        ]
        
        score = scorer.calculate_faithfulness(answer, context)
        
        assert score < 0.50, "Unfaithful answer should have score < 0.50"
    
    def test_empty_context_returns_zero_score(self):
        """Test that empty context returns zero faithfulness."""
        from src.evaluation.evaluation_metrics import FaithfulnessScorer
        
        scorer = FaithfulnessScorer()
        
        score = scorer.calculate_faithfulness("Some answer", [])
        
        assert score == 0.0, "Empty context should return zero faithfulness"


class TestEvaluationMetricsIntegration:
    """Integration tests for the complete metrics module."""
    
    def test_calculate_all_metrics_for_single_query(self):
        """Test calculating all metrics for a single query result."""
        from src.evaluation.evaluation_metrics import EvaluationMetrics
        
        metrics = EvaluationMetrics()
        
        query_result = {
            'question': 'How is AI used in healthcare?',
            'answer': 'AI is used for diagnosis and treatment planning.',
            'expected_answer': 'AI helps in medical diagnosis and treatment.',
            'retrieved_chunks': [
                {'chunk': 'AI in diagnosis', 'article_id': 'art1'},
                {'chunk': 'Treatment planning', 'article_id': 'art2'}
            ],
            'relevant_article_ids': ['art1', 'art2'],
            'context': ['AI is used in medical diagnosis.', 'Treatment planning uses AI.'],
            'response_time': 0.5
        }
        
        result = metrics.evaluate_single_query(query_result)
        
        assert 'answer_relevancy' in result
        assert 'context_precision' in result
        assert 'context_recall' in result
        assert 'f1_score' in result
        assert 'faithfulness' in result
        assert 'response_time' in result
        
        assert 0 <= result['answer_relevancy'] <= 1
        assert 0 <= result['context_precision'] <= 1
        assert 0 <= result['faithfulness'] <= 1
    
    def test_aggregate_metrics_across_queries(self):
        """Test aggregating metrics across multiple queries."""
        from src.evaluation.evaluation_metrics import EvaluationMetrics
        
        metrics = EvaluationMetrics()
        
        query_results = [
            {
                'question': 'Q1',
                'answer': 'A1',
                'expected_answer': 'A1',
                'retrieved_chunks': [{'article_id': 'art1'}],
                'relevant_article_ids': ['art1'],
                'context': ['Context 1'],
                'response_time': 0.5
            },
            {
                'question': 'Q2',
                'answer': 'A2',
                'expected_answer': 'A2 similar',
                'retrieved_chunks': [{'article_id': 'art2'}],
                'relevant_article_ids': ['art2'],
                'context': ['Context 2'],
                'response_time': 0.7
            }
        ]
        
        aggregate = metrics.aggregate_results(query_results)
        
        assert 'mean_answer_relevancy' in aggregate
        assert 'mean_context_precision' in aggregate
        assert 'mean_faithfulness' in aggregate
        assert 'response_time_stats' in aggregate
        
        assert aggregate['total_queries'] == 2
