"""
Testing module for ArRAG-Eval system.
Contains evaluation metrics and testing functions for retrieval systems.
"""

from typing import List, Set, Dict, Union


def recall_at_k(retrieved_items: List[Union[str, int]], relevant_items: Set[Union[str, int]], k: int) -> float:
    """
    Calculate Recall@k metric for retrieval evaluation.
    
    Recall@k measures what fraction of the relevant items appear in the top k retrieved results.
    Formula: Recall@k = (number of relevant items in top k results) / (total number of relevant items)
    
    Args:
        retrieved_items: List of retrieved items in ranked order
        relevant_items: Set of relevant items for the query
        k: The cutoff rank position
        
    Returns:
        float: Recall@k value between 0 and 1
        
    Raises:
        ValueError: If k is negative or if relevant_items is empty
        
    Example:
        >>> retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        >>> relevant = {'doc2', 'doc4', 'doc5'}
        >>> recall_at_k(retrieved, relevant, k=3)
        0.3333333333333333  # Only doc2 is in top 3, so 1/3
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    
    if not relevant_items:
        raise ValueError("relevant_items cannot be empty")
    
    if k == 0:
        return 0.0
    
    # Get top k retrieved items
    top_k_items = set(retrieved_items[:k])
    
    # Count how many relevant items are in top k
    relevant_in_top_k = len(top_k_items & relevant_items)
    
    # Calculate recall
    recall = relevant_in_top_k / len(relevant_items)
    
    return recall


def recall_at_ks(
    retrieved_items: List[Union[str, int]], 
    relevant_items: Set[Union[str, int]], 
    ks: List[int] = None
) -> Dict[int, float]:
    """
    Calculate Recall@k for multiple k values.
    
    Args:
        retrieved_items: List of retrieved items in ranked order
        relevant_items: Set of relevant items for the query
        ks: List of k values to compute recall for. Defaults to [1, 5, 10]
        
    Returns:
        dict: Dictionary mapping k values to their corresponding recall scores
        
    Example:
        >>> retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        >>> relevant = {'doc2', 'doc4', 'doc5'}
        >>> recall_at_ks(retrieved, relevant, ks=[1, 3, 5])
        {1: 0.0, 3: 0.3333333333333333, 5: 1.0}
    """
    if ks is None:
        ks = [1, 5, 10]
    
    results = {}
    for k in ks:
        results[k] = recall_at_k(retrieved_items, relevant_items, k)
    
    return results
