"""
Semantic Chunker

Splits text into sentences, embeds groups of sentences, and places
chunk boundaries where the embedding similarity between consecutive
groups drops below a threshold (mean - 1 std of all similarities).

This is the "smart" chunker: boundaries align with topic shifts
rather than character counts. Trade-off: slower (requires embedding
every sentence group during chunking, not just at index time).
"""

import re
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from shared.embedder import embed_texts, cosine_similarity


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.strip()) > 10]


def _group_sentences(sentences, window=3):
    groups = []
    for i in range(len(sentences)):
        start = max(0, i - window // 2)
        end = min(len(sentences), i + window // 2 + 1)
        groups.append(" ".join(sentences[start:end]))
    return groups


def chunk(
    pages,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    similarity_threshold=None,
):
    full_text = "\n".join(p["page_content"] for p in pages)
    source = pages[0]["metadata"]["source"] if pages else "unknown"
    base_meta = {k: v for k, v in pages[0]["metadata"].items()} if pages else {}

    sentences = _split_sentences(full_text)
    if len(sentences) < 3:
        return [{"text": full_text, "metadata": {**base_meta, "chunker": "semantic"}}]

    groups = _group_sentences(sentences, window=3)
    embeddings = embed_texts(groups)

    similarities = []
    for i in range(len(embeddings) - 1):
        similarities.append(cosine_similarity(embeddings[i], embeddings[i + 1]))

    if similarity_threshold is None:
        mean = np.mean(similarities)
        std = np.std(similarities)
        similarity_threshold = mean - std

    breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < similarity_threshold]
    breakpoints = [0] + breakpoints + [len(sentences)]

    results = []
    for i in range(len(breakpoints) - 1):
        start, end = breakpoints[i], breakpoints[i + 1]
        chunk_text = " ".join(sentences[start:end]).strip()
        if len(chunk_text) < 20:
            continue

        meta = {**base_meta, "chunker": "semantic"}

        if len(chunk_text) > chunk_size * 2:
            pieces = []
            words = chunk_text.split()
            current = ""
            for w in words:
                candidate = current + " " + w if current else w
                if len(candidate) > chunk_size:
                    if current:
                        pieces.append(current)
                    current = w
                else:
                    current = candidate
            if current:
                pieces.append(current)
            for p in pieces:
                results.append({"text": p.strip(), "metadata": meta})
        else:
            results.append({"text": chunk_text, "metadata": meta})
    return results
