from difflib import SequenceMatcher
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from elasticsearch import Elasticsearch

from augment_service.config.settings import SLIDING_WINDOW_SIM_THRESHOLD, USIF_SIM_THRESHOLD, SLIDING_WINDOW_SIZE, VERBOSITY
from augment_service.core.bigram_model import BigramModel
from augment_service.core.elasticsearch import exact_token_match_search, find_by_sentidx, lexical_search
from augment_service.preprocessor.text_preprocessor import TextPreProcessor
from augment_service.uSIF.usif import uSIF
from augment_service.utils.utils import log_best_results
from shared.logging import logger

# https://aman.ai/primers/ai/token-sampling/
import numpy as np
import logging

logger = logging.getLogger(__name__)

def top_k_sampling(
    query: Dict,
    candidates: List[Dict],
    top_k: int = 10,
    temperature: float = 1.0,
) -> Dict:
    if not candidates:
        logger.debug("No candidates available for top-k sampling")
        return None
    
    sorted_candidates = sorted(candidates, key=lambda x: x['_cosine'], reverse=True)
    
    k = min(top_k, len(sorted_candidates))
    if k == 0:
        logger.debug("No valid candidates for top-k sampling")
        return None
    
    top_k_candidates = sorted_candidates[:k]
    
    scores = np.array([candidate['_cosine'] for candidate in top_k_candidates])
    exp_scores = np.exp(scores / temperature)
    probabilities = exp_scores / np.sum(exp_scores)
    
    selected_idx = np.random.choice(len(top_k_candidates), p=probabilities)
    selected_candidate = top_k_candidates[selected_idx]
    
    logger.debug(
        f"Selected candidate (top-{k} sampling): "
        f"cosine={selected_candidate['_cosine']:.3f}, prob={probabilities[selected_idx]:.3f} "
        f"«{query['sentence']}» ↔ «{selected_candidate['_source']['sentence']}»"
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

# find longest common subsequence
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
    preprocessor: TextPreProcessor,
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
    
    if len(s1_normalized) < SLIDING_WINDOW_SIZE or len(s2_normalized) < SLIDING_WINDOW_SIZE:
        logger.debug(f"Sentences too short for crossover with window size {SLIDING_WINDOW_SIZE}")
        return None
    
    def find_best_segment(s1: List[str], s2: List[str], window_size: int):
        best_score = -1
        best_pivot = None
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
                        idf_weight = (idf_dict.get(t1, 1.0) + idf_dict.get(t2, 1.0)) / 2
                        similarities.append(sim * idf_weight)
                        # collect idf weights between the two tokens
                        idf_weights.append(idf_weight)
                if similarities:
                    avg_sim = sum(similarities) / sum(idf_weights) if idf_weights else 0
                    if avg_sim > best_score:
                        #print(f"{avg_sim:.2f}")
                        best_score = avg_sim
                        max_sim_idx = similarities.index(max(similarities))
                        best_pivot = (i + max_sim_idx, j + max_sim_idx)
        return best_pivot, best_score
    
    pivot1, pivot2 = None, None
    
    pivot, score = find_best_segment(s1_normalized, s2_normalized, SLIDING_WINDOW_SIZE)
    
    if pivot is None or score < SLIDING_WINDOW_SIM_THRESHOLD:
        logger.debug(f"No valid pivot found (best score {score:.4f} < threshold {SLIDING_WINDOW_SIM_THRESHOLD})")
        return None
    
    pivot1, pivot2 = pivot
    logger.debug(f"Window size {SLIDING_WINDOW_SIZE}: Best score = {score:.4f}, Pivot = {pivot} s1['{s1_normalized[pivot[0]]}'] s2['{s2_normalized[pivot[1]]}']")
    
    # check valid pivot based on sim threshold
    if score < SLIDING_WINDOW_SIM_THRESHOLD:
        logger.debug(f"No valid pivot found (best score {score:.4f} < threshold {SLIDING_WINDOW_SIM_THRESHOLD})")
        return None
    
    # Perform crossover
    crossover_1 = s1[:pivot1] + s2[pivot2:]
    crossover_2 = s2[:pivot2] + s1[pivot1:]
    
    logger.debug(f"Original 1: {preprocessor.detokenize(s1)}")
    logger.debug(f"Crossover 1: {preprocessor.detokenize(crossover_1)}")
    logger.debug(f"Original 2: {preprocessor.detokenize(s2)}")
    logger.debug(f"Crossover 2: {preprocessor.detokenize(crossover_2)}")
    
    # ensure the crossover produces new sentences
    if crossover_1 == s1 or crossover_2 == s2:
        logger.debug("Crossover produced sequences identical to originals, returning None")
        return None
    
    return crossover_1, crossover_2

def topk_one_point_crossover(
    es_client: Elasticsearch,
    preprocessor: TextPreProcessor,
    index_name: str,
    sent_tokens: List[str],
    idf_dict: Dict[str, float],
    word_to_vec: Dict[str, np.ndarray],
    augmented_indices: Set[int]
) -> Optional[Tuple[List[str],List[str], int]]:
    logger.debug("=" * 60)
    logger.debug(f"Current tokens: {sent_tokens}")
    logger.debug(f"Total Sentence tokens: {len(sent_tokens)}")
    
    sent_hit = exact_token_match_search(es_client, sent_tokens, index_name)
    
    if not sent_hit:
        logger.debug("No exact match found for the sentence tokens, returning None")
        return None, None, None
    
    if sent_hit[0]['_source']['sentence_tokens'] != sent_tokens:
        logger.debug("No exact match found for the sentence tokens, returning None")
        return None, None, None
    
    query = sent_hit[0]['_source']
    bm25_hits = lexical_search(es_client, query, index_name)
    if VERBOSITY >= 2:
        log_best_results("BM25 lexical search", query, bm25_hits)
    
    # rerank with uSIF
    rerank_candidates: List[Dict] = []
    for hit in bm25_hits:
        hit_embedding = np.array(hit['_source']['sentence_embedding'])
        if np.all(hit_embedding == 0):
            continue
        cosine = np.dot(query["sentence_embedding"], hit_embedding) / (
                    np.linalg.norm(query["sentence_embedding"]) * np.linalg.norm(hit_embedding)
                )
        if cosine >= USIF_SIM_THRESHOLD:
            hit['_cosine'] = cosine
            rerank_candidates.append(hit)
    
    if not rerank_candidates:
        logger.debug(f"No candidates found after uSIF rerank since no candidate reached the similarity threshold >= {USIF_SIM_THRESHOLD}")
        return None, None, None
    
    if VERBOSITY >= 2:
        log_best_results("uSIF rerank", query, rerank_candidates)
        
    # filter out candidates that are already augmented
    rerank_candidates = [candidate for candidate in rerank_candidates
                         if candidate['_source']['sentence_idx'] not in augmented_indices]

    if not rerank_candidates:
        logger.debug("No unused candidates available after filtering augmented indices")
        return None, None, None

    rerank_candidates.sort(key=lambda x: x['_cosine'], reverse=True)
    
    MAX_CROSSOVER_RETRIES = len(rerank_candidates)
    
    for retry in range(MAX_CROSSOVER_RETRIES):
        top_k_sampling_candidate = top_k_sampling(
            query=query,
            candidates=rerank_candidates,
        )
        
        if not top_k_sampling_candidate:
            logger.debug("No candidate selected from top-k sampling")
            return None, None, None
        
        s1=query['sentence_tokens']
        s2=top_k_sampling_candidate['_source']['sentence_tokens']
        
        # testing purposes, we can use the first candidate
        #s2=rerank_candidates[0]['_source']['sentence_tokens']
        #sent_idx2 = rerank_candidates[0]['_source']['sentence_idx']
        
        # save sent_idx2 in order to know which result sentence we need to replace
        sent_idx2 = top_k_sampling_candidate['_source']['sentence_idx']
        
        result = crossover_context_window(
            preprocessor=preprocessor,
            s1=s1, 
            s2=s2, 
            idf_dict=idf_dict, 
            word_to_vec=word_to_vec,
        )
        
        if result is None:
            logger.debug(f"Retry {retry + 1}/{MAX_CROSSOVER_RETRIES}: Crossover failed, trying next candidate")
            continue
        
        crossover_1, crossover_2 = result
        
        if crossover_1 == s2 and crossover_2 == s1:
            logger.debug(f"Retry {retry + 1}/{MAX_CROSSOVER_RETRIES}: Crossover produced identical sentences, trying next candidate")
            continue
        
        return crossover_1, crossover_2, sent_idx2
    
    logger.debug("No valid crossover found after all retries")
    return None, None, None