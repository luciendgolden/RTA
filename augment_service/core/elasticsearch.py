import hashlib
import math
import os
import pickle
import time
import zlib
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
from typing import Dict, List
from tqdm import tqdm
from augment_service.preprocessor.text_preprocessor import TextPreProcessor
from augment_service.uSIF.usif import uSIF

from shared.logging import logger

def setup_index(es_client: Elasticsearch, index_name: str, embedding_dim: int):
    settings = {
        "settings": {
            "similarity": {"default": {"type": "BM25"}},
            "analysis": {
                "analyzer": {
                    "my_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": []
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "sentence_idx": {"type": "integer"},
                "sentence": {
                    "type": "text",
                    "analyzer": "my_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                "sentence_tokens": {"type": "keyword"},
                "token_count": {"type": "integer"},
                "sentence_embedding": {
                    "type": "dense_vector", 
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
    es_client.indices.create(index=index_name, body=settings)

def index_to_elasticsearch(
    es_client: Elasticsearch,
    preprocessor: TextPreProcessor,
    corpus: List[List[str]],
    usif_model: uSIF,
    index_name: str,
):
    embedding_dim = usif_model.get_embedding_dim() if usif_model else None
    
    setup_index(es_client, index_name, embedding_dim)
    docs = []
    s_strings = []
    total_sentences = len(corpus)
    seen_tokens = set()
    
    # prepare documents
    with tqdm(total=total_sentences, desc="Preparing sentences for Elasticsearch") as pbar:
        for sent_idx, sentence in enumerate(corpus):
            s_tokens = sentence
            tokens_tuple = tuple(s_tokens)
            
            if tokens_tuple not in seen_tokens:
                seen_tokens.add(tokens_tuple)
                s_str = preprocessor.detokenize(s_tokens)
                doc = {
                    'sentence_idx': sent_idx,
                    'sentence': s_str,
                    'sentence_tokens': s_tokens,
                    "token_count": len(s_tokens)
                }
                docs.append(doc)
                s_strings.append(s_str)
            pbar.update(1)
    
    # generate embeddings with uSIF
    if usif_model:
        embeddings = usif_model.embed(s_strings)
        for doc, embedding in zip(docs, embeddings):
            doc['sentence_embedding'] = embedding.tolist()
    
    # actions for bulk indexing
    actions = [{
        '_index': index_name,
        "_id": f"{doc['sentence_idx']}",
        '_source': doc
    } for doc in docs]
    
    if not actions:
        logger.info("No sentences to index, skipping Elasticsearch indexing.")
        return
    
    # Bulk indexing with streaming_bulk
    try:
        total_actions = len(actions)
        success = 0
        failed = []
        with tqdm(total=total_actions, desc="Indexing unique documents", unit="doc") as pbar:
            for ok, result in streaming_bulk(
                client=es_client, 
                actions=actions
            ):
                if ok:
                    success += 1
                else:
                    failed.append(result)
                pbar.update(1)

        logger.info(f"Indexed {success} documents successfully, {len(failed)} failed")
        es_client.indices.refresh(index=index_name)
        
        if failed:
            logger.error(f"First 5 failed actions: {failed[:5]}")
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {e}")
        raise

def lexical_search(es_client: Elasticsearch, query: Dict, index_name: str) -> List[Dict]:
    search_body = {
        "size": 100,
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "sentence": query['sentence']
                        }
                    }
                ],
                "must_not": [{"term": {"sentence_idx": query["sentence_idx"]}}],
            }
        },
        "highlight": {
            "fields": {
                "sentence": {
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                    "fragment_size": 0
                }
            }
        },
        "collapse": {
            "field": "sentence.keyword"
        }
    }
    
    response = es_client.search(index=index_name, body=search_body)
    logger.debug(f"Total hits for bm25 lexical search: {response['hits']['total']['value']}")
    hits = response['hits']['hits']
    return hits

def knn_search(es_client: Elasticsearch, query: Dict, index_name: str) -> List[Dict]:
    search_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "field": "sentence_embedding",
                            "query_vector": query["sentence_embedding"],
                            "k": 10,
                            "num_candidates": 100
                        }
                    }
                ],
                "must_not": [{"term": {"sentence_idx": query["sentence_idx"]}}],
            }
        },
    }
    
    response = es_client.search(index=index_name, body=search_body)
    hits = response['hits']['hits']
    return hits

def find_by_sentidx(es_client: Elasticsearch, sent_idx: int, index_name: str) -> List[Dict]:
    search_body = {
        "query":{
            "term":{
                "sentence_idx": sent_idx
            }
        }
    }
    
    response = es_client.search(index=index_name, body=search_body)
    hits = response['hits']['hits']
    return hits

def exact_token_match_search(es_client: Elasticsearch, sent_tokens: List[str], index_name: str) -> List[Dict]:
    must_clauses = [{"term": {"sentence_tokens": token}} for token in sent_tokens]
    
    search_body = {
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": {
                    "term": {
                        "token_count": len(sent_tokens)
                    }
                }
            }
        }
    }
    
    response = es_client.search(index=index_name, body=search_body)
    hits = response['hits']['hits']
    exact_hits = [hit for hit in hits if hit['_source']['sentence_tokens'] == sent_tokens]
    return exact_hits