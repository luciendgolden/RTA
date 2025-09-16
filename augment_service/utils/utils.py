import csv
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def log_top_results(
    title: str,
    query: Dict,
    hits: List[Dict],
    log_n: int,
    score_key: str = '_score',
    score_format: str = '.2f',
):
    if not hits:
        logger.info(f"No {title} results to display.")
        return

    log_n = min(log_n, len(hits))

    logger.info("-" * 20)
    logger.info(f"Top {log_n}/{len(hits)} {title}: '{query['sentence']}'")
    for i, hit in enumerate(hits[:log_n], 1):
        score = hit[score_key]
        logger.info(f"{i}. {hit['_source']['sentence']} ({score_key}: {score:{score_format}})")