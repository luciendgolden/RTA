from dataclasses import dataclass
from pathlib import Path

from augment_service.preprocessor.text_preprocessor import TextPreProcessor


@dataclass
class RTAConfig:
    query: str = None
    input_file: str = None
    corpus_name: str = None
    output_dir: Path = None
    ratio: float = 1.0
    context_window_size: int = 3
    context_window_threshold: float = 0.6
    rank_constant: int = 60
    embeddings_file: str = None
    vocab_file: str = None
    verbosity: int = 1
    index_name: str = '',
    top_k: int = 10,
    temperature: float = 1.0,
    preprocessor: TextPreProcessor = TextPreProcessor()