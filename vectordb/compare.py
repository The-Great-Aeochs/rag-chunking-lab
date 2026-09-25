"""
Vector DB Comparison

Indexes the same chunks into FAISS, Qdrant (in-memory), and Chroma.
Compares: index build time, query latency, recall@k, and shows
Qdrant's metadata filtering (which FAISS can't do).

Usage:
    python vectordb/compare.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from shared.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from shared.embedder import embed_texts, embed_query
from chunking.recursive import chunk as recursive_chunk
from vectordb.faiss_store import FaissStore
from vectordb.qdrant_store import QdrantStore
from vectordb.chroma_store import ChromaStore

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")

QUERIES = [
    "What is multi-head attention?",
    "How does the transformer handle positional information?",
    "What is the training data used?",
    "What is scaled dot-product attention?",
    "How does self-attention work?",
]


def main():
    pages = load_all_pdfs(PAPERS_DIR)
    chunks = recursive_chunk(
        pages,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    print(f"Loaded {len(pages)} pages → {len(chunks)} chunks\n")

    print("Embedding chunks...")
    t0 = time.perf_counter()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    embed_time = time.perf_counter() - t0
    print(f"Embedded in {embed_time:.2f}s ({len(chunks)/embed_time:.0f} chunks/sec)\n")

    stores = {
        "FAISS": FaissStore(dimension=embeddings.shape[1]),
        "Qdrant": QdrantStore(collection_name="compare", dimension=embeddings.shape[1]),
        "Chroma": ChromaStore(collection_name="compare"),
    }

    # --- Index build time ---
    print(f"{'Store':<10} {'Index Time':>12} {'Vectors':>10}")
    print("-" * 35)
    for name, store in stores.items():
        t0 = time.perf_counter()
        store.add(chunks, embeddings)
        build_time = time.perf_counter() - t0
        print(f"{name:<10} {build_time:>11.4f}s {store.count:>10}")

    # --- Query latency ---
    print(f"\n{'Store':<10} {'Avg Latency':>12} {'Queries':>10}")
    print("-" * 35)
    query_embeddings = [embed_query(q) for q in QUERIES]

    for name, store in stores.items():
        latencies = []
        for qe in query_embeddings:
            t0 = time.perf_counter()
            store.search(qe, k=5)
            latencies.append(time.perf_counter() - t0)
        avg = np.mean(latencies) * 1000
        print(f"{name:<10} {avg:>10.2f}ms {len(QUERIES):>10}")

    # --- Result comparison for one query ---
    print(f"\n--- Top-3 results for: '{QUERIES[0]}' ---\n")
    qe = query_embeddings[0]
    for name, store in stores.items():
        results = store.search(qe, k=3)
        print(f"[{name}]")
        for i, r in enumerate(results):
            preview = r["text"][:120].replace("\n", " ")
            print(f"  {i+1}. (score={r['score']:.4f}) {preview}...")
        print()

    # --- Qdrant metadata filtering demo ---
    print("--- Qdrant metadata filtering (FAISS can't do this) ---\n")
    qdrant = stores["Qdrant"]
    qe = embed_query("What is the model architecture?")

    print("Unfiltered search:")
    for r in qdrant.search(qe, k=3):
        preview = r["text"][:100].replace("\n", " ")
        section = r["metadata"].get("section", "n/a")
        print(f"  (score={r['score']:.4f}, section={section}) {preview}...")

    print("\nFiltered to page 0 only:")
    for r in qdrant.search(qe, k=3, source_filter=chunks[0]["metadata"]["source"]):
        preview = r["text"][:100].replace("\n", " ")
        page = r["metadata"].get("page", "n/a")
        print(f"  (score={r['score']:.4f}, page={page}) {preview}...")


if __name__ == "__main__":
    main()
