"""
LangGraph-based agentic retrieval pipeline.

Orchestrates the end-to-end flow:
1. Decompose: Break complex questions into sub-queries
2. Retrieve: Get relevant chunks for each sub-query
3. Check Sufficiency: Is the retrieved context enough?
4. Generate: Create answer from context
5. Evaluate: Score the answer (faithfulness, relevance, precision)
"""

from typing import TypedDict, List, Dict, Any, Optional
import logging

from langgraph.graph import StateGraph, END

import src.config
from src.agent.decomposer import decompose_query
from src.agent.ollama_client import generate_json, generate_text, OllamaGenerationError
from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


# ============ State Definition ============

class PipelineState(TypedDict):
    """
    The state passed through the LangGraph pipeline.
    
    Tracks question, decomposed sub-queries, retrieved context, 
    generated answer, and evaluation scores.
    """
    
    # Input
    question: str
    
    # Decomposition
    is_decomposed: bool
    sub_queries: List[str]
    
    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]  # List of {"text": str, "id": str, "score": float}
    all_retrieved_chunks: List[Dict[str, Any]]  # Merged across all sub-queries
    
    # Sufficiency check
    is_sufficient: bool
    sufficiency_reason: str
    iteration: int
    max_iterations: int
    
    # Generation
    answer: str
    answer_generated: bool
    
    # Evaluation
    evaluation_scores: Dict[str, float]  # {"faithfulness": 0.82, "answer_relevance": 0.76, ...}
    evaluation_complete: bool


# ============ Node Functions ============

def decompose_node(state: PipelineState, config: ExperimentConfig) -> PipelineState:
    """
    Node 1: Decompose complex questions into sub-queries.
    
    If agentic=False, skips decomposition (returns question as-is).
    If agentic=True and question is simple, also returns as-is.
    If agentic=True and question is complex, uses LLM to decompose.
    
    Updates state:
    - is_decomposed: whether decomposition was performed
    - sub_queries: list of atomic sub-questions
    """
    question = state["question"]
    
    # If agentic loop disabled, no decomposition
    if not config.agentic:
        logger.info("Agentic loop disabled, skipping decomposition")
        return {
            **state,
            "is_decomposed": False,
            "sub_queries": [question],
            "iteration": 0
        }
    
    # Use LLM to decompose
    logger.info(f"Decomposing question: {question}")
    sub_queries = decompose_query(
        query=question,
        language=config.language,
        model=config.llm_model,
        auto_detect=True  # Auto-detect simple vs. complex
    )
    
    is_decomposed = len(sub_queries) > 1
    logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
    
    return {
        **state,
        "is_decomposed": is_decomposed,
        "sub_queries": sub_queries,
        "iteration": 0,
        "max_iterations": config.max_iterations
    }


def retrieve_node(state: PipelineState, vector_store, config: ExperimentConfig) -> PipelineState:
    """
    Node 2: Retrieve chunks for each sub-query.
    
    For each sub-query in state["sub_queries"], calls the appropriate
    vector_store query method and merges results.
    
    Updates state:
    - retrieved_chunks: chunks for the current iteration
    - all_retrieved_chunks: merged across all sub-queries
    """
    sub_queries = state["sub_queries"]
    all_chunks = []
    
    for sub_query in sub_queries:
        logger.info(f"Retrieving for sub-query: {sub_query}"
                    f"(retriever={config.retriever_type}, top_k={config.top_k})"
        )
        
        # Call vector store's hybrid_query (already includes dense + bm25 + RRF)
        if config.retriever_type == "dense":
            chunks = vector_store.query(sub_query, top_k=config.top_k)
        elif config.retriever_type == "bm25":
            chunks = vector_store.bm25_query(sub_query, top_k=config.top_k)
        elif config.retriever_type == "hybrid":
            chunks = vector_store.hybrid_query(sub_query, top_k=config.top_k, k=config.rrf_k)
        else:
            raise ValueError(f"Unknown retriever_type: {config.retriever_type!r}")        
        
        all_chunks.extend(chunks)
        
        logger.info(f"Retrieved {len(chunks)} chunks")
    
    # Deduplicate by chunk ID (in case same chunk retrieved for multiple sub-queries)
    seen_ids = set()
    deduplicated = []
    for chunk in all_chunks:
        if chunk["id"] not in seen_ids:
            deduplicated.append(chunk)
            seen_ids.add(chunk["id"])
    
    logger.info(f"Total unique chunks after dedup: {len(deduplicated)}")
    
    return {
        **state,
        "retrieved_chunks": deduplicated,
        "all_retrieved_chunks": deduplicated
    }


def check_sufficiency_node(state: PipelineState, config: ExperimentConfig) -> PipelineState:
    """
    Node 3: Check if retrieved context is sufficient to answer the question.
    
    If agentic=False, always consider sufficient (skip check).
    If agentic=True, use LLM to assess whether context is adequate.
    
    If NOT sufficient and iterations < max_iterations:
        - Update iteration counter
        - Set is_sufficient=False → will loop back to retrieve with reformulated query
    
    If sufficient or max iterations reached:
        - Set is_sufficient=True → proceed to generation
    
    Updates state:
    - is_sufficient: bool
    - sufficiency_reason: str explaining the decision
    - iteration: incremented if will retry
    """
    
    # If agentic disabled, always sufficient
    if not config.agentic:
        logger.info("Agentic loop disabled, skipping sufficiency check")
        return {
            **state,
            "is_sufficient": True,
            "sufficiency_reason": "agentic disabled"
        }
    
    question = state["question"]
    chunks_text = "\n---\n".join([c["text"] for c in state["all_retrieved_chunks"]])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)
    
    # Create sufficiency check prompt
    sufficiency_prompt = f"""You are evaluating whether retrieved context is sufficient to answer a question.

Question: {question}

Retrieved context:
{chunks_text}

Answer these two questions:
1. Can this question be answered from the provided context?
2. What information is missing (if any)?

Respond in JSON format:
{{"is_sufficient": true/false, "missing_info": "what is missing, or 'nothing' if complete"}}"""
    
    try:
        parsed = generate_json(
            prompt=sufficiency_prompt,
            model=config.llm_model,
            max_tokens=200,
            temperature=0.1,
        )

        if parsed is None:
            raise OllamaGenerationError("Model did not return parseable JSON")

        is_sufficient = parsed.get("is_sufficient", False)
        missing = parsed.get("missing_info", "unknown")
        
        sufficiency_reason = f"LLM assessment: sufficient={is_sufficient}, missing={missing}"
        logger.info(f"Sufficiency check: {sufficiency_reason}")
        
    except Exception as e:
        logger.error(f"Sufficiency check failed: {e}, assuming sufficient")
        is_sufficient = True
        sufficiency_reason = "check failed, defaulting to sufficient"
    
    # Decide whether to retry
    can_retry = iteration < max_iterations - 1
    will_retry = (not is_sufficient) and can_retry
    
    if will_retry:
        logger.info(f"Marking for retry (iteration {iteration + 1}/{max_iterations})")
        iteration += 1
    
    return {
        **state,
        "is_sufficient": is_sufficient or not can_retry,  # Force sufficient if max iterations reached
        "sufficiency_reason": sufficiency_reason,
        "iteration": iteration
    }


def generate_node(state: PipelineState, config: ExperimentConfig) -> PipelineState:
    """
    Node 4: Generate answer from retrieved context.
    
    Calls LLM with question + retrieved chunks to produce an answer.
    
    Updates state:
    - answer: the generated answer
    - answer_generated: True
    """
    question = state["question"]
    chunks = state["all_retrieved_chunks"]
    
    # Format context
    context_str = "\n---\n".join([
        f"[Source: {c['id']}]\n{c['text']}"
        for c in chunks
    ])
    
    # Create generation prompt
    generation_prompt = f"""Answer the following question using ONLY the provided context.
If the context doesn't contain enough information, say so.

Question: {question}

Context:
{context_str}

Answer:"""
    
    logger.info("Generating answer from context")
    
    try:
        answer = generate_text(
            prompt=generation_prompt,
            model=config.llm_model,
            max_tokens=500,
            temperature=0.7,
        ).strip()
        logger.info(f"Generated answer: {answer[:200]}...")
        
    except OllamaGenerationError as e:
        logger.error(f"Generation failed: {e}")
        answer = "Error: Could not generate answer"
    
    return {
        **state,
        "answer": answer,
        "answer_generated": True
    }


def evaluate_node(state: PipelineState) -> PipelineState:
    """
    Node 5: Evaluate the answer using metrics.
    
    Placeholder for now — in the full implementation, this would call
    the evaluation module to compute faithfulness, answer relevance, etc.
    
    For this version, just log that evaluation is needed.
    
    Updates state:
    - evaluation_scores: dict of metric names → scores
    - evaluation_complete: True
    """
    logger.info("Evaluation step (placeholder — full metrics TBD)")
    
    # In the full implementation:
    # from src.evaluation.metrics import RAGEvaluator
    # evaluator = RAGEvaluator(judge=my_judge)
    # scores = evaluator.evaluate(question, [c["text"] for c in state["all_retrieved_chunks"]], state["answer"])
    
    
    # For now, return placeholder scores
    scores = {
        "faithfulness": 0.0,  # Placeholder
        "answer_relevance": 0.0,  # Placeholder
        "context_precision": 0.0  # Placeholder
    }
    
    return {
        **state,
        "evaluation_scores": scores,
        "evaluation_complete": True
    }


# ============ Conditional Routing ============

def should_retry(state: PipelineState) -> str:
    """
    Determine whether to retry retrieval or proceed to generation.
    
    Returns:
    - "retrieve" if context is insufficient and we haven't hit max iterations
    - "generate" if context is sufficient or max iterations reached
    """
    if state["is_sufficient"]:
        return "generate"
    else:
        # For retry, would need to reformulate query and retrieve again
        # For now, just proceed to generation
        return "generate"


# ============ Pipeline Builder ============

class ArRAGPipeline:
    """
    LangGraph-based agentic RAG pipeline.
    
    Orchestrates: decompose → retrieve → check sufficiency → generate → evaluate
    """
    
    def __init__(self, 
                 config: ExperimentConfig,
                 vector_store):
        """
        Initialize the pipeline.
        
        Args:
            config (ExperimentConfig): Experiment configuration
            vector_store: Vector store instance (must have hybrid_query method)
        """
        self.config = config
        self.vector_store = vector_store
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Construct the LangGraph StateGraph."""
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node(
            "decompose",
            lambda state: decompose_node(state, self.config)
        )
        
        workflow.add_node(
            "retrieve",
            lambda state: retrieve_node(state, self.vector_store, self.config)
        )
        
        workflow.add_node(
            "check_sufficiency",
            lambda state: check_sufficiency_node(state, self.config)
        )
        
        workflow.add_node(
            "generate",
            lambda state: generate_node(state, self.config)
        )
        
        workflow.add_node(
            "evaluate",
            lambda state: evaluate_node(state)
        )
        
        # Define flow
        workflow.set_entry_point("decompose")
        
        # decompose → retrieve
        workflow.add_edge("decompose", "retrieve")
        
        # retrieve → check_sufficiency
        workflow.add_edge("retrieve", "check_sufficiency")
        
        # check_sufficiency → generate or retrieve (retry)
        # For now, always go to generate (retry logic TBD)
        workflow.add_edge("check_sufficiency", "generate")
        
        # generate → evaluate
        workflow.add_edge("generate", "evaluate")
        
        # evaluate → END
        workflow.add_edge("evaluate", END)
        
        return workflow.compile()
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the pipeline on a single question.
        
        Args:
            question (str): The question to answer
        
        Returns:
            Dict containing: question, sub_queries, answer, retrieved_chunks, evaluation_scores
        
        Example:
            >>> pipeline = ArRAGPipeline(config, vector_store)
            >>> result = pipeline.run("What is machine learning?")
            >>> print(result["answer"])
        """
        logger.info(f"Running pipeline for question: {question}")
        
        # Initialize state
        initial_state: PipelineState = {
            "question": question,
            "is_decomposed": False,
            "sub_queries": [],
            "retrieved_chunks": [],
            "all_retrieved_chunks": [],
            "is_sufficient": False,
            "sufficiency_reason": "",
            "iteration": 0,
            "max_iterations": self.config.max_iterations,
            "answer": "",
            "answer_generated": False,
            "evaluation_scores": {},
            "evaluation_complete": False
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Return relevant outputs
        return {
            "question": final_state["question"],
            "sub_queries": final_state["sub_queries"],
            "is_decomposed": final_state["is_decomposed"],
            "retrieved_chunks": final_state["all_retrieved_chunks"],
            "answer": final_state["answer"],
            "evaluation_scores": final_state["evaluation_scores"],
            "iterations": final_state["iteration"]
        }
    
    def run_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Run the pipeline on multiple questions.
        
        Args:
            questions (List[str]): List of questions
        
        Returns:
            List of result dicts (one per question)
        """
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            result = self.run(question)
            results.append(result)
        
        return results
