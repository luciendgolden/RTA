import csv
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

from shared.logging import logger

def load_word_vectors_from_text(filepath: str) -> Dict[str, np.ndarray]:
    word_vectors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # split into word and vector component
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float64)
            word_vectors[word] = vector
    return word_vectors

def log_best_results(
    title: str,
    query: Dict,
    hits: List[Dict],
):
    if not hits:
        logger.debug(f"No {title} results to display.")
        return

    sort_key = "_cosine" if any("_cosine" in hit for hit in hits) else "_score"
    
    try:
        hits.sort(key=lambda x: x.get(sort_key, 0.0), reverse=True)
    except KeyError:
        logger.debug(f"Warning: No valid {sort_key} found in hits for sorting.")
        return

    logger.debug(f"Top 10 {title} results (based on {len(hits)} results):")
    for hit in hits[:10]:
        match = hit.get("_source", {})
        score = hit.get(sort_key, 0.0)
        query_sentence = query.get("sentence", "N/A")
        match_sentence = match.get("sentence", "N/A")
        
        logger.debug(
            f"similarity={score:.3f} «{query_sentence}» ↔ «{match_sentence}»"
        )
        
        highlight = hit.get("highlight", {})
        if highlight.get("sentence"):
            logger.debug(f"«{highlight['sentence'][0]}»")
    
    logger.debug("-" * 20)