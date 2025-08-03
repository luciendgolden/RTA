from pathlib import Path
from typing import List
import spacy
from tqdm import tqdm

from shared.logging import logger

class TextPreProcessor:
    def __init__(self):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        logger.debug(self.nlp.pipe_names)
    
    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize a list of tokens into the original sentence."""
        return ''.join(tokens)
    
    def preprocess_text(self, text: str) -> list[str]:
        doc = self.nlp(text)
        tokens = [tok.text_with_ws for tok in doc]
        
        return tokens
            
    def preprocess_corpus(self, corpus_path: str, use_sent_tokenize: bool = True) -> List[List[str]]:
        """Preprocess a corpus"""
        processed_corpus = []
        
        if Path(corpus_path).is_file():
            with open(corpus_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            raise ValueError(f"Invalid file path: {corpus_path}")
        
        for line in tqdm(lines, desc="Processing corpus"):
            if use_sent_tokenize:
                docs = self.nlp(line)
                for doc in docs.sents:
                    #tokens = [tok.text for tok in doc]
                    tokens = [tok.text_with_ws for tok in doc]
                    if tokens:
                        processed_corpus.append(tokens)
            else:
                sent_text = [line]
                processed_corpus.append(sent_text)
                
        logger.info([self.detokenize(sentence_tokens) for sentence_tokens in processed_corpus[:10]])
        
        return processed_corpus
    
    def load_corpus(self, corpus_path: str) -> List[str]:
        """Load a corpus from a file path."""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        return [line.strip() for line in lines if line.strip()]