"""FAISS indexes used by the ANN teaching lab."""

import time

import faiss
import numpy as np


INDEX_NAMES = ["Flat (exact)", "HNSW", "IVF Flat", "IVF-PQ"]


def normalize_vectors(vectors):
    normalized = np.asarray(vectors, dtype="float32").copy()
    faiss.normalize_L2(normalized)
    return normalized


def _pq_subquantizers(dimension, preferred=48):
    for candidate in (preferred, 64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
        if candidate <= dimension and dimension % candidate == 0:
            return candidate
    return 1


def build_index(
    name,
    vectors,
    *,
    hnsw_m=16,
    ef_construction=80,
    ef_search=32,
    nlist=16,
    nprobe=4,
    pq_bits=4,
):
    """Build one normalized inner-product index and return it with its settings."""
    vectors = normalize_vectors(vectors)
    count, dimension = vectors.shape
    recommended_nlist = max(1, count // 39)
    effective_nlist = max(1, min(int(nlist), count, recommended_nlist))
    effective_nprobe = max(1, min(int(nprobe), effective_nlist))

    if name == "Flat (exact)":
        index = faiss.IndexFlatIP(dimension)
        settings = "exact exhaustive search"
    elif name == "HNSW":
        index = faiss.IndexHNSWFlat(dimension, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(ef_construction)
        index.hnsw.efSearch = int(ef_search)
        settings = f"M={int(hnsw_m)}, efConstruction={int(ef_construction)}, efSearch={int(ef_search)}"
    elif name == "IVF Flat":
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dimension),
            dimension,
            effective_nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        settings = f"nlist={effective_nlist}, nprobe={effective_nprobe}"
    elif name == "IVF-PQ":
        subquantizers = _pq_subquantizers(dimension)
        # FAISS recommends roughly 39 training vectors per codeword. Reduce
        # the bit depth for this small classroom corpus to avoid a misleading
        # under-trained quantizer while preserving the compression lesson.
        training_bits = int(np.floor(np.log2(max(count // 39, 2))))
        effective_pq_bits = max(1, min(int(pq_bits), training_bits))
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatIP(dimension),
            dimension,
            effective_nlist,
            subquantizers,
            effective_pq_bits,
            faiss.METRIC_INNER_PRODUCT,
        )
        settings = (
            f"nlist={effective_nlist}, nprobe={effective_nprobe}, "
            f"m={subquantizers}, bits={effective_pq_bits}"
        )
    else:
        raise ValueError(f"Unknown ANN index: {name}")

    started = time.perf_counter()
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    build_ms = (time.perf_counter() - started) * 1000

    if hasattr(index, "nprobe"):
        index.nprobe = effective_nprobe

    return index, settings, build_ms


def search_index(index, query_vectors, k, repeats=3):
    queries = normalize_vectors(query_vectors)
    index.search(queries[:1], k)

    started = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        scores, indices = index.search(queries, k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    latency_ms = elapsed_ms / (max(1, int(repeats)) * len(queries))
    return scores, indices, latency_ms


def ann_recall(exact_indices, candidate_indices):
    recalls = []
    for exact, candidate in zip(exact_indices, candidate_indices):
        expected = {int(i) for i in exact if i >= 0}
        found = {int(i) for i in candidate if i >= 0}
        recalls.append(len(expected & found) / len(expected) if expected else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def serialized_size_bytes(index):
    return len(faiss.serialize_index(index))
