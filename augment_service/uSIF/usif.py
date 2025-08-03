import logging
from typing import Dict, List, Optional, Set, Union
import numpy as np
import re
from sklearn.decomposition import TruncatedSVD
import spacy

from shared.logging import logger

# https://github.com/kawine/usif
class WordProbabilities:
    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        min_freq: Optional[int] = 1,
        count_file: Optional[str] = None
    ):
        self.nlp = spacy.blank("en")
        self.prob = {}
        self.min_prob = 1e-6
        
        self.counts = {}
        total = 0.0
        
        if count_file:
            # load from file (as seen in uSIF word2prob)
            with open(count_file, 'r') as f:
                for line in f:
                    word, count = line.strip().split()
                    word = word.lower()
                    count = int(count)
                    self.counts[word] = self.counts.get(word, 0) + count
        elif corpus:
            docs = self.nlp.pipe(corpus)
            
            # compute from corpus and tokenize first
            for doc in docs:
                for token in doc:
                    token = token.text.lower()
                    self.counts[token] = self.counts.get(token, 0) + 1
        else:
            raise ValueError("Must provide either a corpus or a count file.")
        
        if min_freq > 1:
            for token in list(self.counts.keys()):
                if self.counts[token] < min_freq:
                    del self.counts[token]
        
        total = sum(self.counts.values())
        
        # normalize probabilities
        if total > 0:
            self.prob = {k: v / total for k, v in self.counts.items()}
            self.min_prob = min(self.prob.values(), default=1e-6)

    def __getitem__(self, word: str) -> float:
        return self.prob.get(word.lower(), self.min_prob)
        #return self.prob.get(word, self.min_prob)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.prob
        #return word in self.prob

    def __len__(self) -> int:
        return len(self.prob)

    def vocab(self):
        return iter(self.prob.keys())

class WordVectors:
    def __init__(self, source: Union[str, Dict[str, np.ndarray]]):
        self.vectors = {}
        
        if isinstance(source, str):
            self._load_from_file(source)
        elif isinstance(source, dict):
            self._load_from_dict(source)
        else:
            raise ValueError("Source must be a file path (str) or a Dict[str, np.ndarray]")
        
        self.embedding_dim = list(self.vectors.values())[0].shape[0]

    # load from file (as seen in uSIF word2vec)
    def _load_from_file(self, vector_fn: str):
        for line in open(vector_fn):
            line = line.strip().split()
                
            word = line[0]
            try:
                embedding = np.array([float(val) for val in line[1:]], dtype=np.float32)
                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                    continue
                self.vectors[word] = embedding
            except ValueError:
                continue

    def _load_from_dict(self, vecs: Dict[str, np.ndarray]):
        for word, embedding in vecs.items():
                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                    continue
                self.vectors[word] = embedding.astype(np.float32)

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[word]

    def __contains__(self, word: str) -> bool:
        return word in self.vectors


class uSIF:
    # sentence embedding model with uSIF logic
    def __init__(self, 
                 vec: WordVectors, 
                 prob: WordProbabilities,
                 tokenizer: Optional[callable] = None,
                 n: int = 11, 
                 m: int = 5, 
                 ):
        """Initialize a sent2vec object.

		Variable names (e.g., alpha, a) all carry over from the paper.

		Args:
			vec: word2vec object
			prob: word2prob object
			n: expected random walk length. This is the avg sentence length, which
				should be estimated from a large representative sample. For STS
				tasks, n ~ 11. n should be a positive integer.
			m: number of common discourse vectors (in practice, no more than 5 needed)
		"""
        self.nlp = spacy.blank("en")
        self.vec = vec
        self.m = m
        self.tokenizer = tokenizer
        self.oov_counter = {'oov_count': 0, 'total_tokens': 0}
        self.oov_words = {}

        if not (isinstance(n, int) and n > 0):
            raise TypeError("n should be a positive integer")
        
        vocab_size = float(len(prob))
        threshold = 1 - (1 - 1/vocab_size) ** n if vocab_size > 0 else 0
        alpha = len([w for w in prob.vocab() if prob[w] > threshold]) / vocab_size if vocab_size > 0 else 0.5
        Z = 0.5 * vocab_size if vocab_size > 0 else 1.0
        self.a = (1 - alpha) / (alpha * Z) if alpha > 0 else 1e-6
        self.weight = lambda word: (self.a / (0.5 * self.a + prob[word]))
        
        self.fallback_vector = np.ones(self.vec.embedding_dim) * self.a
    
    # Vectorize a given sentence
    def _to_vec(self, sentence: str) -> np.ndarray:
        sentence = sentence.lower()
        doc = self.nlp(sentence)
        # non-punctuation
        tokens = [token.text for token in doc if not token.is_punct]
        valid_tokens = [t for t in tokens if t in self.vec]

        # collect OOV tokens
        oov_tokens = [t for t in tokens if t not in self.vec]
        for token in oov_tokens:
            self.oov_words[token] = self.oov_words.get(token, 0) + 1
        
        self.oov_counter['oov_count'] += len(tokens) - len(valid_tokens)
        self.oov_counter['total_tokens'] += len(tokens)
        
        if not valid_tokens:
            return self.fallback_vector.copy()
        
        v_t = np.array([self.vec[t] for t in valid_tokens])
        
        # Normalize each vector
        v_t = v_t * (1.0 / np.linalg.norm(v_t, axis=0))
        
        # Apply weights
        v_t = np.array([self.weight(t) * v_t[i, :] for i, t in enumerate(valid_tokens)])
        
        # Mean of weighted vectors
        embedding = np.mean(v_t, axis=0)  
        
        # Check for NaN or Inf values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return self.fallback_vector.copy()
        
        return embedding

    # Embed a list of sentences
    def embed(self, sentences: List[str]) -> np.ndarray:
        self.oov_counter = {'oov_count': 0, 'total_tokens': 0}
        vectors = [self._to_vec(s) for s in sentences]
        
        if self.oov_counter['total_tokens'] > 0:
            oov_rate = self.oov_counter['oov_count'] / self.oov_counter['total_tokens']
            #print(f"OOV rate: {oov_rate:.2%}")
        
        if self.m == 0 or len(sentences) < 2:
            return np.array(vectors)
        
        vectors = np.array(vectors)
        
        # uSIF principal‑component removal
        svd = TruncatedSVD(n_components=min(self.m, len(vectors)-1), random_state=0).fit(vectors)
        proj = lambda a, b: a.dot(b) * b / np.dot(b, b)
        for i in range(min(self.m, svd.components_.shape[0])):
            lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
            pc = svd.components_[i]
            vectors = vectors - [lambda_i * proj(v_s, pc) for v_s in vectors]
        
        return np.array(vectors)
    
    def get_oov_words(self) -> Dict[str, int]:
        return dict(sorted(self.oov_words.items()))
    
    def get_embedding_dim(self) -> int:
        return self.vec.embedding_dim
    