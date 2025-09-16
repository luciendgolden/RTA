from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np
import logging
from typing import Dict, List, Optional, Set, Tuple

from elasticsearch import Elasticsearch

from augment_service.config.config import RTAConfig
from augment_service.core.elasticsearch import exact_token_match_search, knn_search, lexical_search
from augment_service.preprocessor.text_preprocessor import TextPreProcessor
from augment_service.utils.utils import log_top_results

logger = logging.getLogger(__name__)

# https://aman.ai/primers/ai/token-sampling/
def top_k_sampling(
    hits: List[Dict],
    top_k: int = 10,
    temperature: float = 1.0,
) -> Dict:
    if not hits:
        logger.debug("No candidates available for top-k sampling")
        return None
    
    sorted_candidates = sorted(hits, key=lambda x: x['_rrf_score'], reverse=True)
    
    k = min(top_k, len(sorted_candidates))
    if k == 0:
        logger.debug("No valid candidates for top-k sampling")
        return None
    
    top_k_candidates = sorted_candidates[:k]
    
    # top-k sampling without usage consideration
    scores = np.array([candidate['_rrf_score'] for candidate in top_k_candidates])
    exp_scores = np.exp(scores / temperature)
    probabilities = exp_scores / np.sum(exp_scores)
    
    selected_idx = np.random.choice(len(top_k_candidates), p=probabilities)
    selected_candidate = top_k_candidates[selected_idx]
    
    logger.debug(
        f"Selected candidate (top-{k} sampling): "
        f"_rrf_score={selected_candidate['_rrf_score']:.3f}, prob={probabilities[selected_idx]:.3f} "
        f"«{selected_candidate['_source']['sentence']}»"
    )
    
    return selected_candidate

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# find lexical longest common subsequence
def crossover_lcs(
    preprocessor: TextPreProcessor,
    s1: List[str],
    s2: List[str]
) -> Tuple[List[str], List[str]]:
    s1_normalized = [token.strip() for token in s1]
    s2_normalized = [token.strip() for token in s2]
    
    if s1_normalized == s2_normalized or not s1_normalized or not s2_normalized:
        logger.debug("identical or empty sentences, skipping crossover")
        return None, None
    
    matcher = SequenceMatcher(None, s1_normalized, s2_normalized)
    matching_blocks = matcher.get_matching_blocks()
    matching_blocks = [m for m in matching_blocks if m.size > 0]
    
    if not matching_blocks:
        logger.debug("No matching blocks found")
        return None, None
    
    match = max(matching_blocks, key=lambda m: m.size)
    
    logger.debug(f"Matching block: {match}")
    logger.debug(f"s1 match (normalized): {s1_normalized[match.a: match.a + match.size]}")
    logger.debug(f"s2 match (normalized): {s2_normalized[match.b: match.b + match.size]}")
    
    pivot1 = match.a + match.size
    pivot2 = match.b + match.size
    
    crossover_1 = s1[:pivot1] + s2[pivot2:]
    crossover_2 = s2[:pivot2] + s1[pivot1:]
    
    logger.debug(f"Original 1: {preprocessor.detokenize(s1)}")
    logger.debug(f"Crossover 1: {preprocessor.detokenize(crossover_1)}")
    logger.debug(f"Original 2: {preprocessor.detokenize(s2)}")
    logger.debug(f"Crossover 2: {preprocessor.detokenize(crossover_2)}")
    
    return crossover_1, crossover_2

# find semantic context window
def crossover_context_window(
    config: RTAConfig,
    s1: List[str], # query sentence tokens
    s2: List[str], # matching sentence tokens
    idf_dict: Dict[str, float],
    word_to_vec: Dict[str, np.ndarray],
) -> Optional[Tuple[List[str], List[str]]]:
    s1_normalized = [token.lower().strip() for token in s1 if token.strip()]
    s2_normalized = [token.lower().strip() for token in s2 if token.strip()]
    
    # edge cases
    if s1_normalized == s2_normalized:
        logger.debug("Sequences are identical after normalization, returning None")
        return None
    if not s1_normalized or not s2_normalized:
        logger.debug("One or both sequences are empty after normalization, returning None")
        return None

    if len(s1_normalized) < config.context_window_size or len(s2_normalized) < config.context_window_size:
        logger.debug(f"Sentences too short for crossover with window size {config.context_window_size}")
        return None
    
    def find_best_segment(s1: List[str], s2: List[str], window_size: int):
        best_score = -1
        best_pivot = None
        best_s1_window = None
        best_s2_window = None
        for i in range(len(s1) - window_size + 1):
            s1_window = s1[i:i + window_size]
            #print("\n")
            #print(f"s1_window({i}:{i + window_size}) {s1_window}")
            for j in range(len(s2) - window_size + 1):
                s2_window = s2[j:j + window_size]
                #print(f"s2_window({j}:{j + window_size}) {s2_window}")
                # average similarity weighted by IDF
                similarities = []
                idf_weights = []
                for t1, t2 in zip(s1_window, s2_window):
                    if t1 in word_to_vec and t2 in word_to_vec:
                        sim = cosine_similarity(word_to_vec[t1], word_to_vec[t2])
                        idf_weight = (idf_dict[t1] + idf_dict[t2])
                        similarities.append(sim * idf_weight)
                        # collect idf weights between the two tokens
                        idf_weights.append(idf_weight)
                if similarities:
                    avg_sim = sum(similarities) / sum(idf_weights) if idf_weights else 0
                    if avg_sim > best_score:
                        #print(f"{s1_window} {s2_window} {avg_sim:.2f}")
                        best_score = avg_sim # S_ctx(i,j)
                        max_sim_idx = similarities.index(max(similarities)) # k*
                        
                        best_pivot = (i + max_sim_idx, j + max_sim_idx) # i^*, j^*
                        best_s1_window = s1_window
                        best_s2_window = s2_window
        return best_pivot, best_score, best_s1_window, best_s2_window

    pivot, score, best_s1_window, best_s2_window = find_best_segment(s1_normalized, s2_normalized, config.context_window_size)
    #print(f"s1['{s1_normalized[pivot[0]]}']")
    #print(f"s2['{s2_normalized[pivot[1]]}']")

    if pivot is None or score < config.context_window_threshold:
        logger.debug(f"No valid pivot found (best score {score:.4f} < threshold {config.context_window_threshold})")
        return None
    
    pivot1, pivot2 = pivot
    logger.debug(f"Window size {config.context_window_size}: Best score = {score:.4f}, Window = w1{best_s1_window} w2{best_s2_window}, Pivot = {pivot} s1['{s1_normalized[pivot[0]]}'] s2['{s2_normalized[pivot[1]]}']")

    # Perform crossover after pivot element
    #crossover_1 = s1[:pivot1 + 1] + s2[pivot2 + 1:]
    #crossover_2 = s2[:pivot2 + 1] + s1[pivot1 + 1:]
    
    # Perform crossover at pivot element
    crossover_1 = s1[:pivot1] + s2[pivot2:]
    crossover_2 = s2[:pivot2] + s1[pivot1:]

    #logger.debug(f"Original 1: {config.preprocessor.detokenize(s1)}")
    #logger.debug(f"Crossover 1: {config.preprocessor.detokenize(crossover_1)}")
    #logger.debug(f"Original 2: {config.preprocessor.detokenize(s2)}")
    #logger.debug(f"Crossover 2: {config.preprocessor.detokenize(crossover_2)}")

    # ensure the crossover produces new sentences
    if crossover_1 == s1 or crossover_2 == s2:
        logger.debug("Crossover produced sequences identical to originals, returning None")
        return None
    
    if crossover_1 == s2 and crossover_2 == s1:
        logger.debug("Crossover produced sequences identical to originals in reverse order, returning None")
        return None
    
    return crossover_1, crossover_2

def find_candidates(
    es_client: Elasticsearch,
    config: RTAConfig,
    query: Dict,
    log_n: int,
) -> List[Dict]:
    # lexical search
    bm25_hits = lexical_search(es_client, query, config.index_name)
    
    if config.verbosity >= 2 and bm25_hits:
        log_top_results(
            title="BM25 results",
            query=query,
            hits=bm25_hits,
            log_n=log_n,
        )
    
    # kNN search with cosine similarity
    knn_hits = knn_search(es_client, query, config.index_name)

    if config.verbosity >= 2 and knn_hits:
        log_top_results(
            title="kNN results",
            query=query,
            hits=knn_hits,
            log_n=log_n,
        )

    if not bm25_hits and not knn_hits:
        return []

    # Reciprocal Rank Fusion (RRF)
    k = config.rank_constant
    rrf_scores = defaultdict(float)

    # BM25 hits: add 1 / (k + rank) for each document
    for rank, hit in enumerate(bm25_hits, 1):
        doc_id = hit['_id']
        rrf_scores[doc_id] += 1 / (k + rank)

    # kNN hits: add 1 / (k + rank) for each document
    for rank, hit in enumerate(knn_hits, 1):
        doc_id = hit['_id']
        rrf_scores[doc_id] += 1 / (k + rank)

    all_hits = {hit['_id']: hit for hit in bm25_hits + knn_hits}
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    rrf_results = []
    for doc_id, rrf_score in sorted_docs:
        hit = all_hits[doc_id]
        hit['_rrf_score'] = rrf_score
        rrf_results.append(hit)
    
    return rrf_results