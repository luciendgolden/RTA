from collections import Counter, defaultdict
import math
import random
from typing import Dict, List
import evaluate
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Self-BLEU
def is_valid_sentence(s: str) -> bool:
    cleaned = ''.join(' ' if not c.isprintable() else c for c in s).strip()
    return len(cleaned) > 0

def self_bleu(sentences: List[str], sample_size: int = 1000, seed: int = 42, num_runs: int = 3):
    bleu = evaluate.load('bleu')
    
    n = len(sentences)
    logger.info(f"Calculating Self-BLEU for {n} sentences with sample size {sample_size}, seed {seed}, and {num_runs} runs")
    
    if n < 2:
        logger.info("Skipping Self-BLEU: Corpus too small (< 2 sentences)")
        return 0.0
    
    run_averages = []
    for run in range(num_runs):
        current_seed = seed + run
        random.seed(current_seed)
        
        if sample_size is not None and sample_size < n:
            sampled_sentences = random.sample(sentences, sample_size)
            logger.info(f"Run {run+1}/{num_runs}: Sampling {sample_size} sentences for Self-BLEU (original: {n})")
        else:
            sampled_sentences = sentences
        
        self_bleu_scores = []
        for i, pred in tqdm(enumerate(sampled_sentences), total=len(sampled_sentences), desc=f"Self-BLEU Run {run+1}/{num_runs}"):
            if not is_valid_sentence(pred):
                self_bleu_scores.append(0.0)
                continue
            references = [sampled_sentences[j] for j in range(len(sampled_sentences)) if j != i]
            references = [ref for ref in references if is_valid_sentence(ref)]
            if not references:
                self_bleu_scores.append(0.0)
                continue
            result = bleu.compute(predictions=[pred], references=[references])
            self_bleu_scores.append(result['bleu'])
        
        run_avg = np.mean(self_bleu_scores) if self_bleu_scores else 0.0
        logger.info(f"Run {run+1}/{num_runs} Self-BLEU Average: {run_avg:.4f}")
        run_averages.append(run_avg)
    
    avg_self_bleu = np.mean(run_averages) if run_averages else 0.0
    return avg_self_bleu

# IDF                                                    
def compute_idf(corpus: List[List[str]]) -> Dict[str, float]:
    df: Dict[str, int] = defaultdict(int)
    for sent in corpus:
        for w in set(sent):
            w = w.strip().lower()
            df[w] += 1
    n_docs = len(corpus) or 1
    return {w: math.log(n_docs / freq) for w, freq in df.items()}

# PPL – Perplexity                                                
def ppl_text(text: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> float:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item() * input_ids.size(1)
        total_length += input_ids.size(1)
    
    avg_loss = total_loss / total_length
    perplexity = math.exp(avg_loss)
    return perplexity

def ppl_corpus(corpus: List[str], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> float:
    model.eval()
    encodings = tokenizer(" ".join(corpus), return_tensors="pt")
    
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    
    return ppl.item()


# Corpus Statistics                                               
def print_corpus_stats(corpus: List[List[str]]) -> Dict[str, float]:
    # merge corpus to 1 string text
    text = ''.join([''.join(sentence) for sentence in corpus])
    total_words = len(text.split())

    # count total non-empty stripped tokens
    total_tokens = sum(len([token for token in sentence if token.strip()]) for sentence in corpus)

    # calculate sentence lengths, excluding empty tokens
    striped_sentences = [len([token for token in sentence if token.strip()]) for sentence in corpus]
    total_sentences = len(striped_sentences)
    avg_sentence_length = np.mean(striped_sentences) if striped_sentences else 0
    std_sentence_length = np.std(striped_sentences) if striped_sentences else 0
    
    # Strip tokens for vocabulary and frequency analysis
    striped_tokens = [token.strip() for sentence in corpus for token in sentence if token.strip()]
    vocab_size = len(set(striped_tokens))
    
    # Word-Token Ratio (WTR)
    wtr = total_tokens/total_words
    
    # Type-Token Ratio (TTR)
    ttr = vocab_size / total_tokens if total_tokens > 0 else 0
    
    # Word frequency and rare words (RWORDS, hapax legomena)
    word_freq = Counter(striped_tokens)
    most_common_words = word_freq.most_common(5)
    rwords = len([word for word, count in word_freq.items() if count == 1])
    
    # Average word length (excluding punctuation)
    word_lengths = [len(token) for token in striped_tokens if token.isalnum()]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0
    
    # Top 5 bigrams
    bigrams = []
    for sentence in corpus:
        cleaned_sentence = [token.strip() for token in sentence if token.strip()]
        bigrams.extend((cleaned_sentence[i], cleaned_sentence[i+1]) for i in range(len(cleaned_sentence)-1))
    bigram_freq = Counter(bigrams)
    most_common_bigrams = bigram_freq.most_common(5)
    
    logger.info("-"*20)
    logger.info(f"Corpus Statistics:")
    logger.info(f"# Sentences: {total_sentences}")
    logger.info(f"# Words: {total_words:,}")
    logger.info(f"# Tokens: {total_tokens:,}")
    logger.info(f"⌀ Sentence Length: {avg_sentence_length:.2f}")
    logger.info(f"σ Sentence Length: {std_sentence_length:.2f}")
    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Word-Token Ratio (WTR): {wtr:.2f}")
    logger.info(f"Type-Token Ratio (TTR): {ttr:.4f}")
    logger.info(f"Rare Words (RWORDS): {rwords:,}")
    logger.info(f"Average Word Length: {avg_word_length:.2f} characters")
    logger.info(f"Top 5 Most Common Words: {most_common_words}")
    logger.info(f"Top 5 Most Common Bigrams: {most_common_bigrams}")
    logger.info("-"*20)
    
    return {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "avg_sentence_length": avg_sentence_length,
        "std_sentence_length": std_sentence_length,
        "vocab_size": vocab_size,
        "ttr": ttr,
        "rwords": rwords,
        "avg_word_length": avg_word_length,
        "most_common_words": most_common_words,
        "most_common_bigrams": most_common_bigrams
    }