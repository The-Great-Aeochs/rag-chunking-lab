"""
Full Evaluation: Chunker × VectorDB matrix

Runs every combination of chunker and vector store against the golden set.
Prints a matrix of recall@5 and MRR so you can see which combo wins.

Usage:
    python eval/run_eval.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from shared.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from shared.embedder import embed_texts, embed_query
from chunking import recursive, character, section_wise, semantic
from vectordb.faiss_store import FaissStore
from vectordb.qdrant_store import QdrantStore
from vectordb.chroma_store import ChromaStore
from eval.metrics import evaluate_store

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")


def main():
    pages = load_all_pdfs(PAPERS_DIR)
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print(f"Pages: {len(pages)} | Golden questions: {len(golden)}\n")

    chunkers = {
        "Recursive": (recursive.chunk, {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP}),
        "Character": (character.chunk, {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP}),
        "Section-wise": (section_wise.chunk, {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP}),
        "Semantic": (semantic.chunk, {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP}),
    }

    store_builders = {
        "FAISS": lambda dim: FaissStore(dimension=dim),
        "Qdrant": lambda dim: QdrantStore(collection_name="eval", dimension=dim),
        "Chroma": lambda dim: ChromaStore(collection_name="eval"),
    }

    header = f"{'Chunker':<14} {'VectorDB':<10} {'Chunks':>7} {'Recall@5':>10} {'MRR':>8} {'Idx(ms)':>9} {'Qry(ms)':>9}"
    print(header)
    print("-" * len(header))

    for cname, (chunk_fn, ckwargs) in chunkers.items():
        t0 = time.perf_counter()
        chunks = chunk_fn(pages, **ckwargs)
        chunk_time = time.perf_counter() - t0

        if not chunks:
            print(f"{cname:<14} — no chunks produced, skipping")
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        dim = embeddings.shape[1]

        for sname, builder in store_builders.items():
            store = builder(dim)

            t0 = time.perf_counter()
            store.add(chunks, embeddings)
            idx_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            scores = evaluate_store(store, embed_query, golden, k=5)
            qry_time = (time.perf_counter() - t0) * 1000 / len(golden)

            print(
                f"{cname:<14} {sname:<10} {len(chunks):>7} "
                f"{scores['recall@5']:>10.2%} {scores['mrr']:>8.3f} "
                f"{idx_time:>8.1f}ms {qry_time:>8.2f}ms"
            )

    print("\nDone. Try changing chunk_size and rerun.")


if __name__ == "__main__":
    main()
