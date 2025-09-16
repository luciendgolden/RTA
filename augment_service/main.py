import argparse
import copy
import json
import os
import random
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from elasticsearch import Elasticsearch
from typing import List, Dict, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import logging

from augment_service.config.config import RTAConfig
from augment_service.config.logging import configure_logging
from augment_service.core.rta import crossover_context_window, find_candidates, top_k_sampling
from augment_service.preprocessor.text_preprocessor import TextPreProcessor
from augment_service.uSIF.usif import WordProbabilities, WordVectors, uSIF
from augment_service.utils.metrics import compute_idf, ppl_corpus, print_corpus_stats, self_bleu
from augment_service.utils.utils import log_top_results
from augment_service.core.elasticsearch import exact_token_match_search, index_to_elasticsearch, knn_search, lexical_search

logger = logging.getLogger(__name__)

class RTAService:
    def __init__(self, config: RTAConfig):
        self.config = config
        self.synthetic_corpus = []
        self.usage_counts = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._load_autoregressive_model()

        # initialize Elasticsearch
        self.es = self._initialize_elasticsearch()
        # load corpus into List[List[str]]
        self.corpus = self._load_corpus(self.config.input_file)
        # index sentences in Elasticsearch
        self._index_sentences(self.corpus)

    @contextmanager
    def timer(self, task_name: str):
        start_time = time.time()
        logger.info(f"Starting {task_name}...")
        yield
        elapsed_time = time.time() - start_time
        
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        time_parts = []
        if hours >= 1:
            time_parts.append(f"{int(hours)} hour{'s' if hours > 1 else ''}")
        if minutes >= 1:
            time_parts.append(f"{int(minutes)} minute{'s' if minutes > 1 else ''}")
        if seconds >= 0:
            time_parts.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")
        
        formatted_time = ", ".join(time_parts)
        logger.info(f"Finished {task_name}")
        logger.info(f"Elapsed time: {formatted_time}")

    def _load_autoregressive_model(self) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        model_id = 'openai-community/gpt2'
        model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        return model, tokenizer

    def _initialize_elasticsearch(self) -> Elasticsearch:
        es = Elasticsearch(
            hosts=["http://localhost:9200"],
        )
        if not es.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")
        return es
    
    def _load_corpus(self, corpus_file_path: str) -> List[List[str]]:
        with self.timer(f"Corpus preprocessing {self.config.input_file}"):
            corpus = self.config.preprocessor.preprocess_corpus(corpus_file_path)
            stats = print_corpus_stats(corpus)
            avg_sentence_length = int(stats['avg_sentence_length'])
            
            # init usif
            self.word_vectors = WordVectors(source=self.config.embeddings_file)
            self.word_prob = WordProbabilities(count_file=self.config.vocab_file)
            self.usif = uSIF(vec=self.word_vectors, prob=self.word_prob, n=avg_sentence_length)
            logger.info(f"Initialized uSIF with n={avg_sentence_length}")
            
            # init idf dict
            self.idf_dict = compute_idf(corpus)
            logger.info(f"Computed IDF dictionary with {len(self.idf_dict)} unique words")
            
            return corpus
        
    def _index_sentences(self, corpus: List[List[str]]) -> None:
        with self.timer(f"Indexing sentences to Elasticsearch ({self.config.index_name})"):
            index_to_elasticsearch(
                es_client=self.es,
                preprocessor=self.config.preprocessor,
                corpus=corpus,
                usif_model=self.usif,
                index_name=self.config.index_name,
    )

    # keeping variation sets relevant with only considering query sentences from the original corpus..
    def augment(self) -> List[List[str]]:
        with self.timer(f"Augmenting corpus with ratio {self.config.ratio}"):
            # TODO: for testing purposes
            #self.corpus = self.corpus[:10]
           
            total_tokens = sum(len([token for token in sentence if token.strip()]) for sentence in self.corpus)
            logger.info(f"Total tokens in corpus: {total_tokens:,}")
            target_add_tokens = int(total_tokens * self.config.ratio)
            logger.info(f"Target tokens for augmentation: {target_add_tokens:,}")
            
            original_corpus = copy.deepcopy(self.corpus)
            augmented_corpus = copy.deepcopy(self.corpus)
            total_aug_tokens = 0
           
            original_len = len(original_corpus)
            usage_counts = {i: 0 for i in range(len(original_corpus))}
            max_usage = max(int(self.config.ratio), 1)
            num_rounds = max(int(self.config.ratio), 1)
            
            pos_map = {i: i for i in range(original_len)}
            
            per_round_target = target_add_tokens // num_rounds
            logger.info(f"Number of rounds: {num_rounds}, Target per round: {per_round_target:,}")
            current_round_aug_tokens = 0
           
            for round_num in range(num_rounds):
                insertions = []
                not_punc = re.compile('.*[A-Za-z0-9].*')
                valid_indices = [
                    i for i, sentence in enumerate(original_corpus)
                    if not_punc.match(self.config.preprocessor.detokenize(sentence))
                    and len(sentence) >= self.config.context_window_size
                    and usage_counts[i] < max_usage
                ]

                logger.info(f"Round {round_num + 1}/{num_rounds} - Available sentences for augmentation: {len(valid_indices)} out of {original_len} original sentence items")
               
                with tqdm(total=per_round_target, desc=f"Augmenting sentences (Round {round_num + 1})") as pbar:
                    while current_round_aug_tokens < per_round_target and valid_indices:
                        # ------ weighted query selection ------
                        weights = [1 / (usage_counts[i] + 1) for i in valid_indices]
                        query_idx = random.choices(valid_indices, weights=weights)[0]
                        logger.debug(f"Selected sentence index for augmentation: {query_idx}")
                        query_tokens = original_corpus[query_idx]
                        query_hits = exact_token_match_search(self.es, query_tokens, self.config.index_name)
                       
                        if not query_hits:
                            logger.debug(f"No exact match found for sentence index {query_idx}, skipping")
                            valid_indices.remove(query_idx)
                            continue
                        
                        if query_hits[0]['_source']['sentence_tokens'] != query_tokens:
                            logger.debug("No exact match found for the sentence tokens")
                            valid_indices.remove(query_idx)
                            continue
                        
                        query = query_hits[0]['_source']
                       

                        # ------ find matching lexical/semantic candidates ------
                        rrf_results = find_candidates(
                            es_client=self.es,
                            config=self.config,
                            query=query,
                            log_n=self.config.top_k,
                        )
                        
                        if not rrf_results:
                            logger.debug("No candidates found")
                            valid_indices.remove(query_idx)
                            continue

                        if self.config.verbosity >= 2:
                            log_top_results(
                                title="RRF results",
                                query=query,
                                hits=rrf_results,
                                log_n=self.config.top_k,
                                score_key='_rrf_score',
                                score_format='.3f'
                            )
                        
                        # ------ candidate usage penalization (usage < max_usage e.g. only add candidates that haven't been used when r=1.0) ------
                        available_candidates = []
                        top_k_limit = min(self.config.top_k, len(rrf_results))
                        logger.debug(f"Available candidates before usage filter: len={len(rrf_results)}, top_k_limit={top_k_limit}")
                        for candidate in rrf_results[:top_k_limit]:
                            cand_idx = candidate['_source']['sentence_idx']
                            usage = usage_counts.get(cand_idx, 0)
                            if usage < max_usage:
                                candidate['_rrf_score'] = candidate['_rrf_score'] / (usage + 1)
                                available_candidates.append(candidate)
                        
                        if not available_candidates:
                            logger.debug("No valid candidates after usage filter")
                            valid_indices.remove(query_idx)
                            continue
                        logger.debug(f"Available candidates after usage filter: {len(available_candidates)}")
                        
                        # ------ recombitext augmentation ------
                        MAX_CROSSOVER_RETRIES = len(available_candidates)
                        result = None
                        
                        for retry in range(MAX_CROSSOVER_RETRIES):
                            top_k_sampling_candidate = top_k_sampling(
                                hits=available_candidates,
                                top_k=self.config.top_k,
                                temperature=self.config.temperature,
                            )
                           
                            if not top_k_sampling_candidate:
                                logger.debug("No valid candidates after top-k sampling")
                                break
                           
                            s1=query['sentence_tokens']
                            s2=top_k_sampling_candidate['_source']['sentence_tokens']
                            candidate_idx = top_k_sampling_candidate['_source']['sentence_idx']
                            result = crossover_context_window(
                                config=self.config,
                                s1=s1,
                                s2=s2,
                                idf_dict=self.idf_dict,
                                word_to_vec=self.word_vectors.vectors,
                            )
                           
                            if result is None:
                                logger.debug(f"Retry {retry + 1}/{MAX_CROSSOVER_RETRIES}: Crossover failed, trying next candidate")
                                available_candidates = [c for c in available_candidates if c['_id'] != top_k_sampling_candidate['_id']]
                                continue
                           
                            xo_query, xo_candidate = result
                            break
                       
                        if result is None:
                            logger.debug("Crossover attempt failed")
                            valid_indices.remove(query_idx)
                            continue
                        
                        self.synthetic_corpus.append(xo_query)
                        self.synthetic_corpus.append(xo_candidate)

                        # ------ save positions ------
                        current_query_pos = pos_map[query_idx]
                        insertions.append((current_query_pos + 1, xo_candidate))
                        if candidate_idx < original_len:
                            current_candidate_pos = pos_map[candidate_idx]
                            insertions.append((current_candidate_pos + 1, xo_query))
                        else:
                            insertions.append((len(augmented_corpus), xo_query))
                       
                        # ------ successful augmentation ------
                        added_tokens = len(xo_query) + len(xo_candidate)
                        total_aug_tokens += added_tokens
                        current_round_aug_tokens += added_tokens

                        # ------ usage count ------
                        usage_counts[query_idx] += 1
                        if candidate_idx in usage_counts:
                            usage_counts[candidate_idx] += 1
                        
                        if usage_counts[query_idx] >= max_usage:
                            valid_indices.remove(query_idx)
                        # if uncommented, the candidate dont get considered as query anymore (leading to early stops)
                        #if candidate_idx in usage_counts and candidate_idx in valid_indices and usage_counts[candidate_idx] >= max_usage: 
                        #    valid_indices.remove(candidate_idx)
                        
                        # ------ update progress ------
                        pbar.update(added_tokens)
                
                # ------ round completed, do insertions ------
                insertions.sort(key=lambda x: x[0], reverse=True)
                for pos, new_sent in insertions:
                    augmented_corpus.insert(pos, new_sent)
                    for orig_idx in list(pos_map.keys()):
                        if pos_map[orig_idx] >= pos:
                            pos_map[orig_idx] += 1
                    
                logger.info(f"Current round augmented tokens: {current_round_aug_tokens:,}")
                current_round_aug_tokens = 0

            # ------ finalization ------
            self.usage_counts = usage_counts
            logger.info(f"Total augmented tokens: {total_aug_tokens:,} out of {target_add_tokens:,}")

            return augmented_corpus

    def search_query_and_augment(self, query_text: str) -> Tuple[List[str], List[str]]:
        with self.timer(f"Searching and augmenting '{query_text}'"):
            query_tokens = self.config.preprocessor.preprocess_text(query_text)
            if not query_tokens:
                logger.warning("Query resulted in no tokens after preprocessing")
                return []
            
            # sentence embedding
            query_embedding = self.usif.embed([query_text])[0]
            
            # query for Elasticsearch
            query = {
                'sentence_idx': -1,
                'sentence': query_text,
                'sentence_tokens': query_tokens,
                'sentence_embedding': query_embedding
            }

            rrf_results = find_candidates(
                es_client=self.es,
                config=self.config,
                query=query,
                log_n=self.config.top_k,
            )
            
            log_top_results(
                title="RRF results",
                query=query,
                hits=rrf_results,
                log_n=self.config.top_k,
                score_key='_rrf_score',
                score_format='.3f'
            )

            available_candidates = rrf_results[:]
            MAX_CROSSOVER_RETRIES = len(available_candidates)
            result = None
            for retry in range(MAX_CROSSOVER_RETRIES):
                top_k_sampling_candidate = top_k_sampling(
                    hits=rrf_results,
                    top_k=self.config.top_k,
                    temperature=self.config.temperature,
                )
                
                if not top_k_sampling_candidate:
                    logger.debug("No candidate selected from top-k sampling")
                    return
                
                s1=query['sentence_tokens']
                s2=top_k_sampling_candidate['_source']['sentence_tokens']
                
                result = crossover_context_window(
                    config=self.config,
                    s1=s1, 
                    s2=s2, 
                    idf_dict=self.idf_dict,
                    word_to_vec=self.word_vectors.vectors,
                )
                
                if result is None:
                    logger.debug(f"Retry {retry + 1}/{MAX_CROSSOVER_RETRIES}: Crossover failed, trying next candidate")
                    available_candidates = [c for c in available_candidates if c['_id'] != top_k_sampling_candidate['_id']]
                    continue
                
                crossover_1, crossover_2 = result
                
                logger.info("-" * 20)
                logger.info(f"Retry {retry + 1}/{MAX_CROSSOVER_RETRIES}: Crossover successful")
                logger.info(f"Reference Sentence: {self.config.preprocessor.detokenize(s1)}")
                logger.info(f"Candidate Sentence: {self.config.preprocessor.detokenize(s2)}")
                logger.info(f"Aug1: {self.config.preprocessor.detokenize(crossover_1)}")
                logger.info(f"Aug2: {self.config.preprocessor.detokenize(crossover_2)}")

                break
            
            if result is None:
                logger.info("All crossover attempts failed")
                return []
                
            return result
        
    def evaluate(self) -> None:
        original_sentences = [self.config.preprocessor.detokenize(sentence) for sentence in self.corpus]
        synthetic_sentences = [self.config.preprocessor.detokenize(sentence) for sentence in self.synthetic_corpus]
        
        #print(original_sentences)
        #print(synthetic_sentences)
        
        # save usage counts in self.config.output_dir
        usage_counts_path = self.config.output_dir / f"{self.config.corpus_name}.json"
        with open(usage_counts_path, "w") as f:
            json.dump(self.usage_counts, f, indent=4)
        logger.info(f"Saved usage counts to {usage_counts_path}")
        
        # perplexity (ppl)
        with self.timer("Evaluating Perplexity (PPL)"):
            # ppl training data
            ppl_training = ppl_corpus(original_sentences, self.model, self.tokenizer)
            logger.info(f"Perplexity of training data: {ppl_training:.2f}")
            # ppl synthetic data
            ppl_synthetic = ppl_corpus(synthetic_sentences, self.model, self.tokenizer)
            logger.info(f"Perplexity of synthetic data: {ppl_synthetic:.2f}")

        # self-BLEU
        with self.timer("Evaluating Self-BLEU"):
            # self-BLEU for original sentences as baseline
            avg_original_self_bleu = self_bleu(original_sentences)
            logger.info(f"Average Self-BLEU (Baseline): {avg_original_self_bleu:.4f}")
            
            # self-BLEU for synthetic sentences
            avg_synthetic_self_bleu = self_bleu(synthetic_sentences)
            logger.info(f"Average Self-BLEU (Augmented): {avg_synthetic_self_bleu:.4f}")

        return

def postclean(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    
    text = multiple_spaces_ex.sub(' ', text) # multiple spaces
    text = space_before_punctuation_ex.sub(r'\1', text) # space before punctuation
    
    text = re.sub(r'\n\n', '', text) # multipl newlines
    text = re.sub(r'\s*\n\s*', '\n', text) # newlines with spaces
    return text

def main():
    parser = argparse.ArgumentParser(description="RecombiText Augmentation (RTA)")
    parser.add_argument("--query",type=str,default=None,help="Query search string")
    parser.add_argument("--input_file",type=str,default=None,help="Input file")
    parser.add_argument("--output_dir",type=str,default=None,help="Output directory")
    parser.add_argument("--ratio",type=float,default=1.0,help="Synth:Original ratio")
    parser.add_argument("--context_window_size",type=int,default=3,help="Context window size")
    parser.add_argument("--context_window_threshold",type=float,default=0.8,help="Context window threshold")
    parser.add_argument("--rank_constant",type=int,default=60,help="RRF Rank constant")
    parser.add_argument("--embeddings_file",type=str,default=None,help="Path to embeddings file")
    parser.add_argument("--vocab_file",type=str,default=None,help="Path to vocabulary file")
    parser.add_argument("--top_k",type=int,default=10,help="Top K candidates for sampling")
    parser.add_argument("--temperature",type=float,default=1.0,help="Temperature for sampling")
    parser.add_argument("--do_eval",action="store_true",help="Whether to evaluate the data")
    parser.add_argument("--verbosity",type=int,default=1,help="Verbosity level") # 0: ERROR, 1: INFO, 2: DEBUG
    args = parser.parse_args()
    
    configure_logging(verbosity=args.verbosity)
    
    if args.input_file is None:
        raise ValueError("Input file is required.")

    if args.embeddings_file is None:
        raise ValueError("Embeddings file is required.")

    if args.vocab_file is None:
        raise ValueError("Vocabulary file is required.")

    corpus_name = Path(args.input_file).stem
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    config = RTAConfig(
        query=args.query,
        input_file=args.input_file,
        corpus_name=corpus_name,
        output_dir=output_dir,
        ratio=args.ratio,
        context_window_size=args.context_window_size,
        context_window_threshold=args.context_window_threshold,
        rank_constant=args.rank_constant,
        embeddings_file=args.embeddings_file,
        vocab_file=args.vocab_file,
        verbosity=args.verbosity,
        index_name=f"sent_docs_{corpus_name}",
        top_k=args.top_k,
        temperature=args.temperature
    )

    try:
        service = RTAService(config)
        
        if args.query:
            service.search_query_and_augment(args.query)
            return

        if args.output_dir:
            # augment training data
            augmented_corpus = service.augment()
            print_corpus_stats(augmented_corpus)
            #full_corpus = [''.join(sentence) for sentence in augmented_corpus]
            #print(full_corpus)

            # save augmented corpus
            text = ' '.join([''.join(sentence) for sentence in augmented_corpus])
            cleaned_text = postclean(text)
            new_file = config.output_dir / f"{corpus_name}.aug"
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            with open(new_file, "w") as f:
                f.write(cleaned_text)
            logger.info(f"Saved augmented corpus to {new_file}")

            if args.do_eval:
                service.evaluate()

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()