"""
RAG Evaluation Metrics — implemented from scratch.

This module implements core RAGAS metrics:
- Faithfulness: answer only asserts things supported by context
- Answer Relevance: answer actually addresses the question
- Context Precision: retrieved chunks are useful, not noise

Each metric is independently callable, so you can ablate
and analyze retrieval, generation, and ranking separately.

References:
  RAGAS paper: https://arxiv.org/abs/2309.15217
  Metric definitions guide: https://docs.ragas.io/
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
# import sqlite3
# from pathlib import Path
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a single metric computation."""
    
    metric_name: str
    score: float  # 0-1
    details: Dict[str, Any]  # Breakdown for analysis
    
    def to_dict(self) -> dict:
        return {
            "metric": self.metric_name,
            "score": round(self.score, 3),
            "details": self.details
        }


class Faithfulness:
    """
    Faithfulness metric: Does the answer only use supported information?
    
    Approach:
    1. Decompose answer into atomic claims
    2. For each claim, ask: "Is this supported by the context?"
    3. Score = (supported claims) / (total claims)
    
    Why it matters:
    - High faithfulness = answer doesn't hallucinate
    - Low faithfulness = model making up information
    
    Example:
        Context: "Paris is in France"
        Answer: "Paris is in France and is the capital of Europe"
        
        Claims: ["Paris is in France", "Paris is capital of Europe"]
        Supported: [True, False]
        Faithfulness: 0.5
    """
    
    def __init__(self, judge):
        """
        Initialize Faithfulness metric.
        
        Args:
            judge: LLMJudge instance for making support decisions
        """
        self.judge = judge
    
    def compute(self, 
                answer: str, 
                context: List[str]) -> MetricResult:
        """
        Compute faithfulness score.
        
        Args:
            answer (str): Generated answer
            context (List[str]): Retrieved context chunks
        
        Returns:
            MetricResult with score (0-1) and details
        """
        if not answer.strip():
            return MetricResult(
                metric_name="faithfulness",
                score=0.0,
                details={"reason": "empty answer"}
            )
        
        context_text = "\n---\n".join(context)
        
        # Step 1: Decompose answer into claims
        claims = self.judge.decompose_claims(answer)
        
        if not claims:
            logger.warning(f"Could not decompose answer into claims: {answer[:100]}")
            return MetricResult(
                metric_name="faithfulness",
                score=0.0,
                details={"reason": "could not decompose claims"}
            )
        
        logger.info(f"Decomposed into {len(claims)} claims")
        
        # Step 2: Check support for each claim
        supported_claims = []
        claim_verdicts = []
        
        for claim in claims:
            is_supported = self.judge.is_claim_supported(
                claim=claim,
                context=context_text
            )
            supported_claims.append(is_supported)
            claim_verdicts.append({
                "claim": claim,
                "supported": is_supported
            })
            
            logger.debug(f"  Claim: '{claim[:60]}...' → {is_supported}")
        
        # Step 3: Compute score
        score = sum(supported_claims) / len(claims) if claims else 0.0
        
        return MetricResult(
            metric_name="faithfulness",
            score=score,
            details={
                "num_claims": len(claims),
                "supported_claims": sum(supported_claims),
                "claim_verdicts": claim_verdicts
            }
        )


class AnswerRelevance:
    """
    Answer Relevance metric: Does the answer address the question?
    
    Approach (reverse generation):
    1. Generate N questions that could be answered by this answer
    2. Compute cosine similarity between:
       - Original question embedding
       - Generated question embeddings
    3. Score = mean(similarities)
    
    Why it matters:
    - High relevance = answer directly addresses the question
    - Low relevance = answer talks about something else
    
    Intuition:
    If your answer truly answers the question, you should be able to
    reverse-engineer the question from the answer.
    
    Example:
        Q: "What is machine learning?"
        A: "Machine learning is a subset of AI where systems learn from data"
        
        Generated Qs:
        - "Define machine learning"
        - "What is ML?"
        - "How does machine learning work?"
        
        Similarities to original Q: [0.92, 0.88, 0.85]
        Score: 0.88
    """
    
    def __init__(self, 
                 embedding_model: Optional[SentenceTransformer] = None,
                 num_generated_questions: int = 3):
        """
        Initialize AnswerRelevance metric.
        
        Args:
            embedding_model: SentenceTransformer for computing similarities
                            If None, uses e5-small
            num_generated_questions: How many questions to generate from answer
        """
        if embedding_model is None:
            embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
        
        self.embedding_model = embedding_model
        self.num_generated_questions = num_generated_questions
    
    def compute(self, 
                question: str,
                answer: str,
                judge) -> MetricResult:
        """
        Compute answer relevance score.
        
        Args:
            question (str): Original question
            answer (str): Generated answer
            judge: LLMJudge instance for generating questions
        
        Returns:
            MetricResult with score (0-1) and details
        """
        if not answer.strip():
            return MetricResult(
                metric_name="answer_relevance",
                score=0.0,
                details={"reason": "empty answer"}
            )
        
        # Step 1: Generate questions from answer
        generated_questions = judge.generate_questions(
            answer=answer,
            num_questions=self.num_generated_questions
        )
        
        if not generated_questions:
            logger.warning("Could not generate questions from answer")
            return MetricResult(
                metric_name="answer_relevance",
                score=0.0,
                details={"reason": "could not generate questions"}
            )
        
        logger.info(f"Generated {len(generated_questions)} questions from answer")
        
        # Step 2: Compute embeddings
        q_embedding = self.embedding_model.encode(question, normalize_embeddings=True)
        gen_q_embeddings = self.embedding_model.encode(
            generated_questions,
            normalize_embeddings=True
        )
        
        # Step 3: Compute cosine similarities
        similarities = util.cos_sim(q_embedding, gen_q_embeddings)[0].tolist()
        
        logger.debug(f"Similarities to original question: {similarities}")
        
        # Step 4: Average similarity
        score = np.mean(similarities) if similarities else 0.0
        
        return MetricResult(
            metric_name="answer_relevance",
            score=score,
            details={
                "generated_questions": generated_questions,
                "similarities": [round(s, 3) for s in similarities],
                "mean_similarity": round(score, 3)
            }
        )


class ContextPrecision:
    """
    Context Precision metric: Are retrieved chunks actually useful?
    
    Approach:
    1. For each chunk, determine if it contributed to the answer
    2. Chunks ranked higher should be more relevant
    3. Score = weighted precision (penalize useful chunks ranked low)
    
    Formula:
        precision@k = (number of relevant chunks in top-k) / k
        context_precision = mean(precision@k for each k)
    
    Why it matters:
    - High precision = retrieved chunks are useful, not noisy
    - Low precision = wasting context on irrelevant chunks
    - Identifies problems in retrieval ranking
    
    Example:
        Retrieved: [chunk_A, chunk_B, chunk_C, chunk_D]
        Relevant:  [True,   True,   False,  False]
        
        precision@1 = 1/1 = 1.0
        precision@2 = 2/2 = 1.0
        precision@3 = 2/3 = 0.67
        precision@4 = 2/4 = 0.5
        
        context_precision = (1.0 + 1.0 + 0.67 + 0.5) / 4 = 0.79
    """
    
    def __init__(self, judge):
        """
        Initialize ContextPrecision metric.
        
        Args:
            judge: LLMJudge instance for relevance assessment
        """
        self.judge = judge
    
    def compute(self, 
                question: str,
                context: List[str],
                answer: str) -> MetricResult:
        """
        Compute context precision score.
        
        Args:
            question (str): The question being answered
            context (List[str]): Retrieved context chunks (in ranking order)
            answer (str): Generated answer
        
        Returns:
            MetricResult with score (0-1) and details
        """
        if not context:
            return MetricResult(
                metric_name="context_precision",
                score=0.0,
                details={"reason": "no context"}
            )
        
        # Step 1: Determine relevance for each chunk
        relevances = []
        chunk_assessments = []
        
        for i, chunk in enumerate(context):
            is_relevant = self.judge.is_chunk_relevant(
                question=question,
                chunk=chunk,
                answer=answer
            )
            relevances.append(is_relevant)
            chunk_assessments.append({
                "rank": i + 1,
                "relevant": is_relevant,
                "text_preview": chunk[:80] + "..."
            })
            
            logger.debug(f"  Chunk {i+1}: {is_relevant}")
        
        # Step 2: Compute precision@k for each position
        precisions = []
        for k in range(1, len(relevances) + 1):
            num_relevant = sum(relevances[:k])
            precision_at_k = num_relevant / k
            precisions.append(precision_at_k)
        
        # Step 3: Average precision (mean of all precision@k)
        score = np.mean(precisions) if precisions else 0.0
        
        logger.info(f"Context precision@k: {[round(p, 3) for p in precisions]}")
        
        return MetricResult(
            metric_name="context_precision",
            score=score,
            details={
                "num_chunks": len(context),
                "relevant_chunks": sum(relevances),
                "precision_at_k": [round(p, 3) for p in precisions],
                "chunk_assessments": chunk_assessments
            }
        )


class RAGEvaluator:
    """
    Complete RAG evaluator combining all metrics.
    
    Usage:
        evaluator = RAGEvaluator(judge=my_judge)
        
        # Evaluate a single question
        scores = evaluator.evaluate(
            question=q,
            context=chunks,
            answer=answer
        )
        
        # Evaluate batch
        results = evaluator.evaluate_batch(
            questions=questions,
            contexts=contexts,
            answers=answers
        )
    """
    
    def __init__(self, judge, embedding_model=None):
        """
        Initialize RAG evaluator.
        
        Args:
            judge: LLMJudge instance
            embedding_model: SentenceTransformer for answer relevance
        """
        self.judge = judge
        self.faithfulness = Faithfulness(judge=judge)
        self.answer_relevance = AnswerRelevance(embedding_model=embedding_model)
        self.context_precision = ContextPrecision(judge=judge)
    
    def evaluate(self,
                 question: str,
                 context: List[str],
                 answer: str) -> Dict[str, MetricResult]:
        """
        Evaluate a single QA triple.
        
        Args:
            question (str): The question
            context (List[str]): Retrieved context chunks
            answer (str): Generated answer
        
        Returns:
            Dict mapping metric name → MetricResult
        """
        logger.info(f"Evaluating: {question[:80]}...")
        
        results = {}
        
        # Compute each metric
        results["faithfulness"] = self.faithfulness.compute(answer, context)
        results["answer_relevance"] = self.answer_relevance.compute(
            question, answer, self.judge
        )
        results["context_precision"] = self.context_precision.compute(
            question, context, answer
        )
        
        return results
    
    def evaluate_batch(self,
                      questions: List[str],
                      contexts: List[List[str]],
                      answers: List[str]) -> List[Dict[str, MetricResult]]:
        """
        Evaluate multiple QA triples.
        
        Args:
            questions: List of questions
            contexts: List of context lists
            answers: List of answers
        
        Returns:
            List of results dicts (one per QA triple)
        """
        assert len(questions) == len(contexts) == len(answers)
        
        all_results = []
        for i, (q, ctx, a) in enumerate(zip(questions, contexts, answers)):
            logger.info(f"Evaluating sample {i+1}/{len(questions)}")
            result = self.evaluate(q, ctx, a)
            all_results.append(result)
        
        return all_results
    
    def aggregate_results(self,
                         results: List[Dict[str, MetricResult]]) -> Dict[str, float]:
        """
        Aggregate results across multiple samples.
        
        Args:
            results: List of evaluation results
        
        Returns:
            Dict mapping metric name → average score
        """
        metric_names = ["faithfulness", "answer_relevance", "context_precision"]
        aggregated = {}
        
        for metric_name in metric_names:
            scores = [
                result[metric_name].score
                for result in results
                if metric_name in result
            ]
            
            if scores:
                aggregated[metric_name] = np.mean(scores)
                aggregated[f"{metric_name}_std"] = np.std(scores)
        
        return aggregated


# ============================================================================
# ABLATION-FRIENDLY API: Test each component separately
# ============================================================================

def evaluate_chunking_strategy(chunks_a: List[str],
                               chunks_b: List[str],
                               question: str,
                               answer: str,
                               judge) -> Tuple[float, float]:
    """
    Compare two chunking strategies on a single answer.
    
    For LinkedIn: "We tested two chunking strategies on the same answer"
    
    Args:
        chunks_a: Chunks from strategy A
        chunks_b: Chunks from strategy B
        question: The question
        answer: Generated answer
        judge: LLMJudge
    
    Returns:
        (context_precision_a, context_precision_b)
    """
    evaluator = RAGEvaluator(judge=judge)
    
    precision_a = evaluator.context_precision.compute(question, chunks_a, answer)
    precision_b = evaluator.context_precision.compute(question, chunks_b, answer)
    
    return precision_a.score, precision_b.score


def evaluate_retrieval_strategy(question: str,
                                context_dense: List[str],
                                context_hybrid: List[str],
                                answer: str,
                                judge) -> Dict[str, float]:
    """
    Compare dense vs hybrid retrieval on context precision.
    
    For LinkedIn: "Dense vs Hybrid retrieval on Arabic QA"
    
    Args:
        question: The question
        context_dense: Retrieved by dense search
        context_hybrid: Retrieved by hybrid (RRF)
        answer: Generated answer
        judge: LLMJudge
    
    Returns:
        Dict with scores for each strategy
    """
    evaluator = RAGEvaluator(judge=judge)
    
    precision_dense = evaluator.context_precision.compute(question, context_dense, answer)
    precision_hybrid = evaluator.context_precision.compute(question, context_hybrid, answer)
    
    return {
        "dense": precision_dense.score,
        "hybrid": precision_hybrid.score,
        "improvement": precision_hybrid.score - precision_dense.score
    }


def evaluate_generation_quality(question: str,
                                context: List[str],
                                answers: List[str],
                                judge) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple generated answers (e.g., different LLMs).
    
    For LinkedIn: "GPT-4o vs GPT-4o-mini for answer generation"
    
    Args:
        question: The question
        context: Retrieved context (same for all)
        answers: Generated by different methods
        judge: LLMJudge
    
    Returns:
        Dict mapping answer_id → scores
    """
    evaluator = RAGEvaluator(judge=judge)
    
    comparison = {}
    for i, answer in enumerate(answers):
        scores = evaluator.evaluate(question, context, answer)
        comparison[f"answer_{i}"] = {
            name: result.score
            for name, result in scores.items()
        }
    
    return comparison
