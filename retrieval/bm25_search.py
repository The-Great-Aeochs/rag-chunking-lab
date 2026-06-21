"""
BM25 Sparse Search

Keyword-based retrieval using BM25 (Best Matching 25). Unlike dense
vector search which captures semantic meaning, BM25 matches on exact
keywords with TF-IDF weighting. Good at finding specific terms that
embedding models might miss.
"""

from rank_bm25 import BM25Okapi


class BM25Search:
    def __init__(self, chunks):
        self.chunks = chunks
        self.corpus = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(scores[idx]),
            })
        return results
