# accuracy.py
"""Exact- and top-k match metrics using the <recipe> delimiter."""

from typing import List
from difflib import SequenceMatcher

def compute_soft_accuracy(split_preds: List[str], gold: List[str]) -> float:
    """
    Compute soft accuracy by averaging the highest similarity between each predicted name
    and any of the gold names using SequenceMatcher.
    
    Args:
        split_preds: List of predicted recipe names
        gold: List of gold/reference recipe names
        
    Returns:
        float: Average similarity score between predictions and gold names
    """
    if not split_preds or not gold:
        return 0.0
    
    # Clean predictions and gold to avoid empty strings
    clean_preds = [p.strip() for p in split_preds if p.strip()]
    clean_gold = [g.strip() for g in gold if g.strip()]
    
    if not clean_preds or not clean_gold:
        return 0.0

    def best_match_score(pred: str, gold_list: List[str]) -> float:
        return max(SequenceMatcher(None, pred.lower(), g.lower()).ratio() for g in gold_list)

    total_score = sum(best_match_score(p, clean_gold) for p in clean_preds)
    return total_score / len(clean_preds)