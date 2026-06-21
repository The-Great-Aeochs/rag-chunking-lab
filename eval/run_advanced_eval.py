"""
Advanced Retrieval Evaluation

Compares retrieval strategies against the baseline:
  - Baseline: Dense search (embed query -> vector search)
  - RAG Fusion: Multi-query + Reciprocal Rank Fusion
  - HyDE: Hypothetical Document Embedding
  - Hybrid: Dense + BM25 with RRF
  - Reranker: Dense retrieval + Cohere cross-encoder reranking

Uses Section-wise chunker + Qdrant as the fixed infrastructure.
Measures Recall@5 and MRR across the golden set.

Usage:
    python eval/run_advanced_eval.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from shared.loader import load_all_pdfs
from shared.embedder import embed_texts, embed_query
from chunking import section_wise
from vectordb.qdrant_store import QdrantStore
from eval.metrics import recall_at_k, reciprocal_rank
from retrieval.bm25_search import BM25Search
from retrieval.hybrid import HybridSearch
from retrieval.query_rewriter import rag_fusion_search, hyde_search
from retrieval.reranker import retrieve_and_rerank

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")


def evaluate_retrieval(search_fn, golden, label):
    """Run a search function against the golden set and return metrics."""
    recalls, mrrs, latencies = [], [], []

    for item in golden:
        t0 = time.perf_counter()
        try:
            results = search_fn(item["question"])
        except Exception as e:
            print(f"  [{label}] Error on '{item['question'][:40]}...': {e}")
            results = []
        latency = (time.perf_counter() - t0) * 1000

        recalls.append(recall_at_k(results, item["evidence"], k=5))
        mrrs.append(reciprocal_rank(results, item["evidence"], k=5))
        latencies.append(latency)

    n = len(golden)
    return {
        "recall@5": sum(recalls) / n if n else 0,
        "mrr": sum(mrrs) / n if n else 0,
        "avg_latency_ms": sum(latencies) / n if n else 0,
    }


def main():
    pages = load_all_pdfs(PAPERS_DIR)
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print(f"Pages: {len(pages)} | Golden questions: {len(golden)}")
    print("Chunker: Section-wise | Store: Qdrant\n")

    chunks = section_wise.chunk(pages, chunk_size=800, chunk_overlap=80)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]

    store = QdrantStore(collection_name="adv_eval", dimension=dim)
    store.add(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks ({dim}d embeddings)\n")

    strategies = {}

    # 1. Baseline: Dense search
    strategies["Dense (baseline)"] = lambda q: store.search(embed_query(q), k=5)

    # 2. BM25 only
    bm25 = BM25Search(chunks)
    strategies["BM25 (sparse)"] = lambda q: bm25.search(q, k=5)

    # 3. Hybrid: Dense + BM25
    hybrid = HybridSearch(chunks, store, embed_query)
    strategies["Hybrid (Dense+BM25)"] = lambda q: hybrid.search(q, k=5)

    # 4. RAG Fusion (requires OpenAI)
    if os.environ.get("OPENAI_API_KEY"):
        strategies["RAG Fusion"] = lambda q: rag_fusion_search(q, store, embed_query, k=5)[0]

    # 5. HyDE (requires OpenAI)
    if os.environ.get("OPENAI_API_KEY"):
        strategies["HyDE"] = lambda q: hyde_search(q, store, embed_query, k=5)[0]

    # 6. Reranker (requires Cohere)
    if os.environ.get("COHERE_API_KEY"):
        strategies["Reranker (Cohere)"] = lambda q: retrieve_and_rerank(q, store, embed_query, retrieve_k=20, final_k=5)

    # 7. Hybrid + Reranker (requires both)
    if os.environ.get("COHERE_API_KEY"):
        from retrieval.reranker import rerank
        def hybrid_rerank(q):
            candidates = hybrid.search(q, k=20, dense_k=20, sparse_k=20)
            return rerank(q, candidates, top_n=5)
        strategies["Hybrid + Reranker"] = hybrid_rerank

    header = f"{'Strategy':<25} {'Recall@5':>10} {'MRR':>8} {'Latency':>12}"
    print(header)
    print("-" * len(header))

    for label, search_fn in strategies.items():
        scores = evaluate_retrieval(search_fn, golden, label)
        print(
            f"{label:<25} {scores['recall@5']:>10.2%} {scores['mrr']:>8.3f} "
            f"{scores['avg_latency_ms']:>10.1f}ms"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
