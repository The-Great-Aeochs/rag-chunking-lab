"""
Cross-Encoder Reranker (Cohere)

After initial retrieval (broad recall), a cross-encoder reranks results
by processing query + document together. Cross-encoders are more accurate
than bi-encoders (embedding models) because they see the full interaction
between query and document, but they're too slow to run on the entire corpus.

Pattern: retrieve broadly (top 20) -> rerank to find the best 5.
"""

import os
import time
import cohere

_last_call_time = 0


def _get_client():
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set -- add it to .env")
    return cohere.ClientV2(api_key=api_key)


def _rate_limit():
    """Cohere trial keys allow 10 calls/min. Wait if needed."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < 6.5:
        time.sleep(6.5 - elapsed)
    _last_call_time = time.time()


def rerank(query, results, top_n=5, model="rerank-v3.5"):
    """Rerank search results using Cohere's cross-encoder.

    Args:
        query: the user's question
        results: list of dicts with "text" and "metadata" from initial retrieval
        top_n: how many to return after reranking
        model: Cohere rerank model name

    Returns:
        Reranked list of result dicts with updated scores
    """
    if not results:
        return []

    _rate_limit()
    client = _get_client()
    documents = [r["text"] for r in results]

    response = client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
    )

    reranked = []
    for item in response.results:
        original = results[item.index]
        reranked.append({
            "text": original["text"],
            "metadata": original["metadata"],
            "score": float(item.relevance_score),
        })
    return reranked


def retrieve_and_rerank(query, store, embed_query_fn, retrieve_k=20, final_k=5):
    """Two-stage retrieval: broad vector search -> cross-encoder rerank.

    Retrieves retrieve_k candidates, then reranks to find the best final_k.
    """
    qe = embed_query_fn(query)
    candidates = store.search(qe, k=retrieve_k)
    return rerank(query, candidates, top_n=final_k)
