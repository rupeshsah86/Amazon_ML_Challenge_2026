import numpy as np
import pandas as pd
from typing import Dict, List, Set, Union

def calculate_entity_f05(pred_ids: Set[str], true_ids: Set[str]) -> float:
    r"""
    Calculate F0.5 score for a single Source 1 entity.
    
    Precision = |pred \cap true| / |pred|  (1.0 if both empty, 0.0 if pred non-empty & true empty)
    Recall    = |pred \cap true| / |true|  (1.0 if both empty, 0.0 if true non-empty & pred empty)
    F0.5      = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    if len(pred_ids) == 0 and len(true_ids) == 0:
        return 1.0
    if len(pred_ids) == 0 or len(true_ids) == 0:
        return 0.0
    
    intersection = len(pred_ids.intersection(true_ids))
    if intersection == 0:
        return 0.0
    
    precision = intersection / len(pred_ids)
    recall = intersection / len(true_ids)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def calculate_macro_f05(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> float:
    """
    Calculate Macro F0.5 across all Source 1 entities in ground_truth.
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = calculate_entity_f05(pred_set, true_set)
        scores.append(score)
    return float(np.mean(scores))

def parse_matched_ids(val: Union[str, float]) -> Set[str]:
    """Parse comma/space separated matched IDs into a set."""
    if pd.isna(val) or val is None:
        return set()
    val_str = str(val).strip()
    if not val_str:
        return set()
    # Replace commas with spaces and split
    ids = val_str.replace(',', ' ').split()
    return set(s.strip() for s in ids if s.strip())
