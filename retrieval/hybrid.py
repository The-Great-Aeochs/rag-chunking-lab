"""
Hybrid Search (Dense + Sparse)

Combines dense vector search (semantic similarity) with BM25 sparse
search (keyword matching) using Reciprocal Rank Fusion (RRF).

Why hybrid? Dense search understands meaning ("LLM as a judge" finds
evaluation methodology) but misses exact keywords. BM25 finds exact
terms but misses semantic similarity. Together they cover both cases.
"""

from retrieval.bm25_search import BM25Search
from retrieval.query_rewriter import reciprocal_rank_fusion


class HybridSearch:
    def __init__(self, chunks, store, embed_query_fn):
        self.store = store
        self.embed_query_fn = embed_query_fn
        self.bm25 = BM25Search(chunks)

    def search(self, query, k=5, dense_k=20, sparse_k=20):
        """Retrieve using both dense and sparse, merge with RRF.

        dense_k/sparse_k: how many candidates to fetch from each before
        merging. Should be larger than final k to give RRF enough signal.
        """
        qe = self.embed_query_fn(query)
        dense_results = self.store.search(qe, k=dense_k)
        sparse_results = self.bm25.search(query, k=sparse_k)

        merged = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        return merged[:k]
