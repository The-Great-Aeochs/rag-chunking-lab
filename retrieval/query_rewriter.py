"""
Query Rewriting Techniques

RAG Fusion: Generate multiple query variants, search with each, combine
            results using Reciprocal Rank Fusion (RRF).

HyDE:       Generate a hypothetical answer, embed that instead of the
            original query. The hypothetical doc is closer in embedding
            space to real chunks than a short question is.
"""

import os
import openai


def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)


def generate_query_variants(question, n=4):
    """Generate n alternative phrasings of the question using an LLM."""
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} different search queries that could help answer "
                f"this question. Each query should approach it from a different "
                f"angle or use different keywords. Return only the queries, "
                f"one per line, no numbering.\n\n"
                f"Question: {question}"
            ),
        }],
        temperature=0.7,
    )
    variants = [
        line.strip() for line in response.choices[0].message.content.strip().split("\n")
        if line.strip()
    ]
    return variants[:n]


def reciprocal_rank_fusion(*ranked_lists, k=60):
    """Combine multiple ranked result lists using RRF.

    Each list is a list of dicts with "text", "metadata", "score".
    Returns a single merged list sorted by RRF score.
    k=60 is the standard constant from the original RRF paper.
    """
    scores = {}
    items = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["text"][:200]
            if key not in items:
                items[key] = item
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [
        {**items[key], "score": scores[key]}
        for key in sorted_keys
    ]


def rag_fusion_search(question, store, embed_query_fn, k=5, n_variants=4):
    """RAG Fusion: multi-query search with RRF merging.

    1. Generate n query variants
    2. Search with original + all variants
    3. Merge results using Reciprocal Rank Fusion
    """
    variants = generate_query_variants(question, n=n_variants)
    all_queries = [question] + variants

    ranked_lists = []
    for q in all_queries:
        qe = embed_query_fn(q)
        results = store.search(qe, k=k)
        ranked_lists.append(results)

    merged = reciprocal_rank_fusion(*ranked_lists, k=60)
    return merged[:k], variants


def generate_hypothetical_document(question, chunk_size=800):
    """Generate a hypothetical answer passage for HyDE."""
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Write a detailed passage (~{chunk_size} characters) that "
                f"directly answers this question. Write it as if it were an "
                f"excerpt from a research paper or textbook. Do not say "
                f"'I don't know' — generate a plausible, detailed answer.\n\n"
                f"Question: {question}"
            ),
        }],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def hyde_search(question, store, embed_query_fn, k=5, chunk_size=800):
    """HyDE: search using a hypothetical document embedding.

    1. Generate a hypothetical answer passage
    2. Embed that passage (not the original question)
    3. Search with the hypothetical embedding
    """
    hypo_doc = generate_hypothetical_document(question, chunk_size)
    hypo_embedding = embed_query_fn(hypo_doc)
    results = store.search(hypo_embedding, k=k)
    return results, hypo_doc
