import argparse
import os
import random
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from elasticsearch import Elasticsearch
from typing import List, Dict, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from augment_service.config.settings import (
    AUG_PATH,
    SLIDING_WINDOW_SIZE,
    USIF_SIM_THRESHOLD,
    EVAL_PATH,
    GLOVE_FILE,
    GLOVE_MODEL_PATH,
    VERBOSITY,
)
from augment_service.core.one_point_crossover import cosine_similarity, topk_one_point_crossover
from augment_service.preprocessor.text_preprocessor import TextPreProcessor
from augment_service.uSIF.usif import WordProbabilities, WordVectors, uSIF
from augment_service.utils.metrics import calculate_perplexity_corpus, compute_idf, evaluate_pairs, print_corpus_stats
from augment_service.utils.utils import load_word_vectors_from_text, log_best_results
from augment_service.core.elasticsearch import index_to_elasticsearch, lexical_search
from shared.logging import logger

@dataclass
class AugmentationConfig:
    sliding_window_size: int = SLIDING_WINDOW_SIZE
    sim_threshold: float = USIF_SIM_THRESHOLD
    verbosity: int = VERBOSITY
    augmentation_percentage: float = 0.0
    index_name: str = ''
    preprocessor: TextPreProcessor = TextPreProcessor()

class AugmentationService:
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.es = self._initialize_elasticsearch()
        self.model, self.tokenizer = self._load_autoregressive_model()
        self.preprocessor = self.config.preprocessor
        self.word_to_vec = load_word_vectors_from_text(os.path.join(GLOVE_MODEL_PATH, GLOVE_FILE))
        logger.info(f"Loaded {len(self.word_to_vec)} word vectors from {GLOVE_FILE}")

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
        logger.info(f"{task_name} completed in {formatted_time}")

    def _initialize_elasticsearch(self) -> Elasticsearch:
        es = Elasticsearch(
            hosts=["http://localhost:9200"],
        )
        if not es.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")
        return es

    def _load_autoregressive_model(self) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
        model_id = 'openai-community/gpt2'
        model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        return model, tokenizer
    
    def load_corpus(self, corpus_file_path: str) -> List[List[str]]:
        with self.timer("Corpus preprocessing"):
            corpus_raw = self.preprocessor.load_corpus(corpus_file_path)
            corpus = self.preprocessor.preprocess_corpus(corpus_file_path)
            stats = print_corpus_stats(corpus)
            avg_sentence_length = int(stats['avg_sentence_length'])
            
            # init usif
            word_prob = WordProbabilities(corpus=corpus_raw)
            word_vectors = WordVectors(self.word_to_vec)
            self.usif = uSIF(word_vectors, word_prob, n=avg_sentence_length)
            logger.info(f"Initialized uSIF with n={avg_sentence_length}")
            
            return corpus

    def index_sentences(self, corpus: List[List[str]]) -> None:
        with self.timer(f"Indexing sentences to Elasticsearch ({self.config.index_name})"):
            index_to_elasticsearch(
                es_client=self.es,
                preprocessor=self.preprocessor,
                corpus=corpus,
                usif_model=self.usif,
                index_name=self.config.index_name,
            )

    def augment_corpus(self, corpus: List[List[str]]) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        original_corpus = corpus.copy()
        mixed_corpus = corpus.copy()
        
        total_tokens = sum(len(sentence) for sentence in original_corpus)
        budget = int(self.config.augmentation_percentage * total_tokens)
        logger.info(f"Total tokens in corpus: {total_tokens}, Augmentation budget: {budget} tokens (based on {self.config.augmentation_percentage:.2%} from {total_tokens})")

        augmented_sentences = []
        picked_sentences = []
        total_aug_tokens = 0
        
        # indices that have been augmented
        augmented_indices = set()
        
        # only get incides where sentences are not just punctuation
        not_punc = re.compile('.*[A-Za-z0-9].*')
        # filter valid indices and ensure sentence items have enough tokens
        valid_indices = [
            i for i, sentence in enumerate(original_corpus)
            if not_punc.match(self.preprocessor.detokenize(sentence))
            and len(sentence) >= SLIDING_WINDOW_SIZE
        ]
        
        logger.info(f"Available sentences for augmentation: {len(valid_indices)} out of {len(original_corpus)} total sentence items in corpus")
        
        idf_dict = compute_idf(original_corpus)
        
        with self.timer(f"Augmenting sentences with one-point crossover"):
            with tqdm(total=budget, desc=f"Augmenting sentences with one-point crossover") as pbar:
                    while total_aug_tokens < budget and valid_indices:
                        sent_idx = random.choice(valid_indices)
                        logger.debug(f"Selected sentence index for augmentation: {sent_idx}")
                        tokens = original_corpus[sent_idx]
                        
                        aug1_toks, aug2_toks, sent_idx2 = topk_one_point_crossover(
                            es_client=self.es,
                            preprocessor=self.preprocessor,
                            index_name=self.config.index_name,
                            sent_tokens=tokens,
                            idf_dict=idf_dict,
                            word_to_vec=self.word_to_vec,
                            augmented_indices=augmented_indices,
                        )
                        
                        # check if topk_one_point_crossover returne None
                        if aug1_toks is None or aug2_toks is None or sent_idx2 is None:
                            logger.debug(f"No augmentation possible for sentence {sent_idx}")
                            valid_indices.remove(sent_idx)
                            continue

                        picked_sentences.append(original_corpus[sent_idx])
                        picked_sentences.append(original_corpus[sent_idx2])
                        augmented_sentences.append(aug1_toks)
                        augmented_sentences.append(aug2_toks)

                        if self.config.verbosity >= 2:
                            #sentences 1
                            original_text1 = self.preprocessor.detokenize(original_corpus[sent_idx])
                            augmented_text1 = self.preprocessor.detokenize(aug1_toks)
                            
                            # evaluate sentences 1
                            ppl_original1 = calculate_perplexity_corpus([original_text1], self.model, self.tokenizer)
                            ppl_augmented1 = calculate_perplexity_corpus([augmented_text1], self.model, self.tokenizer)
                            logger.debug(f"Original1: {original_text1} PPL: {ppl_original1:.2f}")
                            logger.debug(f"Augmented1: {augmented_text1} PPL: {ppl_augmented1:.2f}")
                            
                            # sentences 2
                            original_text2 = self.preprocessor.detokenize(original_corpus[sent_idx2])
                            augmented_text2 = self.preprocessor.detokenize(aug2_toks)
                            
                            # evaluate sentences 2
                            ppl_original2 = calculate_perplexity_corpus([original_text2], self.model, self.tokenizer)
                            ppl_augmented2 = calculate_perplexity_corpus([augmented_text2], self.model, self.tokenizer)
                            logger.debug(f"Original2: {original_text2} PPL: {ppl_original2:.2f}")
                            logger.debug(f"Augmented2: {augmented_text2} PPL: {ppl_augmented2:.2f}")
                        
                        # replace
                        mixed_corpus[sent_idx] = aug1_toks
                        mixed_corpus[sent_idx2] = aug2_toks
                        
                        # update tracking
                        aug_token_count = len(aug1_toks) + len(aug2_toks)
                        total_aug_tokens += aug_token_count
                        pbar.update(aug_token_count)
                        
                        # remove sent_idx from valid_indices and update augmented_indices
                        valid_indices.remove(sent_idx)
                        if sent_idx2 in valid_indices:
                            valid_indices.remove(sent_idx2)
                        
                        augmented_indices.add(sent_idx)
                        augmented_indices.add(sent_idx2)
        
        logger.info(f"Augmented {len(augmented_sentences)} sentences, total augmented tokens: {total_aug_tokens}")
        print_corpus_stats(mixed_corpus)
        
        return picked_sentences, augmented_sentences, mixed_corpus

    def post_clean(self, lines):
        new_lines = []
        for line in lines:
            new_line = line.strip()
            if new_line:
                new_lines.append(new_line + '\n')
        return new_lines

    def post_clean_mixed(self, lines):
        new_lines = []
        for line in lines:
            if line != '\n':
                new_lines.append(line)
        return new_lines

    def save_results(self, augmented_sentences: List[List[str]], mixed_corpus: List[List[str]], corpus_file_path: str) -> None:
        if not augmented_sentences:
            logger.info("No augmented sentences to save")
            return

        total_aug_tokens = sum(len(sentence) for sentence in augmented_sentences)
        token_display = (
            f"{total_aug_tokens // 1_000_000}M" if total_aug_tokens >= 1_000_000
            else f"{total_aug_tokens // 1_000}K" if total_aug_tokens >= 1_000
            else str(total_aug_tokens)
        )
        
        # directory structure based on corpus file path
        corpus_path = Path(corpus_file_path)
        corpus_dir = corpus_path.parent.name
        
        aug_percentage = int(self.config.augmentation_percentage * 100)
        aug_subdir = f"{aug_percentage}pct"
        output_dir = os.path.join(AUG_PATH, corpus_dir, aug_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        # consistent filename
        base_filename = f"{self.config.sliding_window_size}sw.{token_display}.{corpus_path.name}"
        augmented_file_path = os.path.join(output_dir, f"{base_filename}.aug.txt")
        mixed_file_path = os.path.join(output_dir, f"{base_filename}.mixed.txt")

        """
        # Save augmented sentences to file
        with open(augmented_file_path, 'w', encoding='utf-8') as f:
            detokenized_sentences = [self.preprocessor.detokenize(sentence) for sentence in augmented_sentences]
            cleaned_sentences = self.post_clean(detokenized_sentences)
            for text in cleaned_sentences:
                f.write(text)
        logger.info(f"Saved augmented corpus to {augmented_file_path}")
        """
        
        with open(mixed_file_path, 'w', encoding='utf-8') as f:
            detokenized_sentences = [self.preprocessor.detokenize(sentence) for sentence in mixed_corpus]
            for text in detokenized_sentences:
                f.write(text)
        
        with open(mixed_file_path, 'r') as f:
            lines = f.readlines()
            
        lines = self.post_clean_mixed(lines)
        
        with open(mixed_file_path, 'w') as f:
            f.writelines(lines)
        
        logger.info(f"Saved mixed corpus to {mixed_file_path}")

    def evaluate_results(self, picked_sentences: List[List[str]], augmented_sentences: List[List[str]], corpus_file_path: str) -> None:
        picked_texts = [self.preprocessor.detokenize(s) for s in picked_sentences]
        augmented_texts = [self.preprocessor.detokenize(s) for s in augmented_sentences]

        with self.timer("Evaluating perplexity"):
            ppl_picked = calculate_perplexity_corpus(picked_texts, self.model, self.tokenizer)
            ppl_augmented = calculate_perplexity_corpus(augmented_texts, self.model, self.tokenizer)
            logger.info(f"Picked sentences PPL: {ppl_picked:.2f}")
            logger.info(f"Augmented sentences PPL: {ppl_augmented:.2f}")

        with self.timer("Evaluating BLEU scores"):
            pairs = list(zip(picked_texts, augmented_texts))
            bleus = evaluate_pairs(pairs)
            integrated_pairs = [(i, p[0].strip(), p[1].strip(), b) for i, (p, b) in enumerate(zip(pairs, bleus))]
            df = pd.DataFrame(integrated_pairs, columns=['#', 'Original', 'Augmented', 'BLEU_Score'])
            
            # total augmented tokens for naming
            total_aug_tokens = sum(len(sentence) for sentence in augmented_sentences)
            token_display = (
                f"{total_aug_tokens // 1_000_000}M" if total_aug_tokens >= 1_000_000
                else f"{total_aug_tokens // 1_000}K" if total_aug_tokens >= 1_000
                else str(total_aug_tokens)
            )
            
            # directory structure based on corpus file path
            corpus_path = Path(corpus_file_path)
            corpus_dir = corpus_path.parent.name
            aug_percentage = int(self.config.augmentation_percentage * 100)
            eval_subdir = f"{aug_percentage}pct"
            output_dir = os.path.join(EVAL_PATH, corpus_dir, eval_subdir)
            os.makedirs(output_dir, exist_ok=True)
            
            # consistent filename
            base_filename = f"{self.config.sliding_window_size}sw.{token_display}.{corpus_path.name}.bleu.csv"
            output_path = os.path.join(output_dir, base_filename)
            
            df.to_csv(
                output_path, 
                index=False, 
                header=True, 
                encoding='utf-8'
            )
            logger.info(f"Saved BLEU scores to {output_path}")

    def search_query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        with self.timer(f"Processing search query '{query_text}'"):
            query_tokens = self.preprocessor.preprocess_text(query_text)
            if not query_tokens:
                logger.warning("Query resulted in no tokens after preprocessing")
                return []
            
            # uSIF embedding
            query_sentence = self.preprocessor.detokenize(query_tokens)
            query_embedding = self.usif.embed([query_sentence])[0]
            
            # query for Elasticsearch
            query = {
                'sentence_idx': -1,
                'sentence': query_sentence,
                'sentence_tokens': query_tokens,
                'sentence_embedding': query_embedding
            }
            
            # lexical search
            bm25_hits = lexical_search(self.es, query, self.config.index_name)
            if not bm25_hits:
                logger.info("No search results found for the query")
                return []
            
            if self.config.verbosity >= 2:
                log_best_results("BM25 lexical search", query, bm25_hits)
            
            # rerank using uSIF
            rerank_candidates = []
            for hit in bm25_hits:
                hit_embedding = np.array(hit['_source']['sentence_embedding'])
                if np.all(hit_embedding == 0):
                    continue
                cosine = cosine_similarity(query_embedding, hit_embedding)
                if cosine >= self.config.sim_threshold:
                    hit['_cosine'] = cosine
                    rerank_candidates.append(hit)
            
            if not rerank_candidates:
                logger.info("No candidates found after uSIF reranking")
                return []
            
            if self.config.verbosity >= 2:
                log_best_results("uSIF rerank", query, rerank_candidates)
            
            selected_candidates = sorted(rerank_candidates, key=lambda x: x['_cosine'], reverse=True)[:top_k]
            
            results = [
                {
                    'sentence': candidate['_source']['sentence'],
                    'cosine_score': candidate['_cosine'],
                    'sentence_idx': candidate['_source']['sentence_idx']
                }
                for candidate in selected_candidates
            ]
            
            logger.info(f"Top {top_k} search results for query: '{query_text}'")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result['sentence']} (Cosine: {result['cosine_score']:.3f})")
            
            return results

def parse_arguments():
    parser = argparse.ArgumentParser(description="Augment text corpus using Elasticsearch and n-gram search")
    parser.add_argument(
        "--query", 
        type=str, 
        default=None, 
        help="Query string to search in the corpus (if provided, corpus will not be augmented)"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to the corpus file to be augmented"
    )    
    parser.add_argument(
        "--aug_percentage",
        type=float,
        default=0.1,
        help="Percentage of tokens to augment (e.g., 0.5 for 50%)"
    )
    return parser.parse_args()

def main():
    # Uncomment for reproducibility
    # random.seed(42)
    
    args = parse_arguments()
    corpus_name = Path(args.corpus).stem
    config = AugmentationConfig(
        augmentation_percentage=args.aug_percentage,
        index_name=f"sent_docs_{corpus_name}"
    )

    service = None
    try:
        service = AugmentationService(config)
        corpus_file_path = args.corpus
        
        corpus = service.load_corpus(corpus_file_path)
        service.index_sentences(corpus)
        
        if args.query:
            service.search_query(args.query)
            return
            
        picked_sentences, augmented_sentences, mixed_corpus = service.augment_corpus(corpus)
        
        if len(augmented_sentences) > 0:
            service.save_results(augmented_sentences, mixed_corpus, corpus_file_path)
            service.evaluate_results(picked_sentences, augmented_sentences, corpus_file_path)

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()