from collections import Counter, defaultdict
import math
from typing import Dict, List, Sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import spacy
from shared.logging import logger

from shared.logging import logger

# ---------------------------------------------------------------------------
# BLEU helpers                                                               
# ---------------------------------------------------------------------------

_chencherry = SmoothingFunction()

def sentence_bleu_score(candidate: Sequence[str], reference: Sequence[str]) -> float:
    return sentence_bleu(
        [reference],
        candidate,
        smoothing_function=_chencherry.method1,
    )

def corpus_bleu_score(candidates: List[Sequence[str]], references: List[Sequence[str]]) -> float:
    if len(candidates) != len(references):
        raise ValueError("Candidates & references length mismatch")
    return corpus_bleu([[ref] for ref in references], candidates, smoothing_function=_chencherry.method1)

def evaluate_pairs(pairs: list[tuple[str, str]]) -> List[float]:
    scores: List[float] = []
    for orig, aug in pairs:
        scores.append(sentence_bleu_score(list(aug), list(orig)))
    return scores

# ---------------------------------------------------------------------------
# Self‑BLEU – diversity estimate                                            
# ---------------------------------------------------------------------------

def self_bleu(sentences: List[Sequence[str]]) -> float:
    if len(sentences) < 2:
        return 0.0
    scores: List[float] = []
    for idx, sent in enumerate(sentences):
        refs = sentences[:idx] + sentences[idx + 1 :]
        score = corpus_bleu([[ref] for ref in refs], [sent], smoothing_function=_chencherry.method1)
        scores.append(score)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Type‑Token Ratio & IDF                                                    
# ---------------------------------------------------------------------------

def ttr(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_idf(corpus: List[List[str]]) -> Dict[str, float]:
    df: Dict[str, int] = defaultdict(int)
    for sent in corpus:
        for w in set(sent):
            w = w.strip().lower()
            df[w] += 1
    n_docs = len(corpus) or 1
    return {w: math.log(n_docs / freq) for w, freq in df.items()}

# ---------------------------------------------------------------------------
# PPL – Perplexity                                                
# ---------------------------------------------------------------------------
def calculate_perplexity(text: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> float:
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

def calculate_perplexity_corpus(corpus: List[str], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> float:
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


# ---------------------------------------------------------------------------
# Corpus Statistics                                               
# ---------------------------------------------------------------------------
def print_corpus_stats(corpus: List[List[str]]) -> Dict[str, float]:
    num_sentences = len(corpus)
    
    # count non-empty tokens after stripping whitespace
    num_tokens = sum(len([token for token in sentence if token.strip()]) for sentence in corpus)
    
    # calculate sentence lengths, excluding empty tokens
    sentence_lengths = [len([token for token in sentence if token.strip()]) for sentence in corpus]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    std_sentence_length = np.std(sentence_lengths) if sentence_lengths else 0
    
    # Strip tokens for vocabulary and frequency analysis
    all_tokens = [token.strip() for sentence in corpus for token in sentence if token.strip()]
    vocabulary_size = len(set(all_tokens))
    
    # Type-Token Ratio (TTR)
    type_token_ratio = vocabulary_size / num_tokens if num_tokens > 0 else 0
    
    # Word frequency and rare words (RWORDS, hapax legomena)
    word_freq = Counter(all_tokens)
    most_common_words = word_freq.most_common(5)
    rwords = len([word for word, count in word_freq.items() if count == 1])
    
    # Average word length (excluding punctuation)
    word_lengths = [len(token) for token in all_tokens if token.isalnum()]
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
    logger.info(f"Total Sentence Items: {num_sentences}")
    logger.info(f"Total Tokens: {num_tokens}")
    logger.info(f"Average Sentence Length: {avg_sentence_length:.2f} tokens")
    logger.info(f"Std Dev of Sentence Length: {std_sentence_length:.2f} tokens")
    logger.info(f"Vocabulary Size: {vocabulary_size}")
    logger.info(f"Type-Token Ratio (TTR): {type_token_ratio:.4f}")
    logger.info(f"Rare Words (RWORDS): {rwords}")
    logger.info(f"Average Word Length: {avg_word_length:.2f} characters")
    logger.info(f"Top 5 Most Common Words: {most_common_words}")
    logger.info(f"Top 5 Most Common Bigrams: {most_common_bigrams}")
    logger.info("-"*20)
    
    return {
        "num_sentences": num_sentences,
        "num_tokens": num_tokens,
        "avg_sentence_length": avg_sentence_length,
        "std_sentence_length": std_sentence_length,
        "vocabulary_size": vocabulary_size,
        "type_token_ratio": type_token_ratio,
        "rwords": rwords,
        "avg_word_length": avg_word_length,
        "most_common_words": most_common_words,
        "most_common_bigrams": most_common_bigrams
    }