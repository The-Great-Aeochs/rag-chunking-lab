# Advanced RAG Assignment: Parent-Child and Multi-Hop Retrieval

## Objective

Extend the existing RAG lab with a pipeline that retrieves small child chunks, expands them into larger parent sections, and answers questions requiring evidence from multiple sections or documents.

## Part 1: Parent-Child Retrieval

1. Use Section-wise chunks as parent documents.
2. Split each parent into 250-400 character child chunks.
3. Add `parent_id`, `child_id`, `source`, `section`, and `page` metadata.
4. Embed and index the child chunks.
5. Retrieve child chunks, but send their parent sections to the generator.
6. Deduplicate parents before constructing the final context.

Suggested module: `retrieval/parent_child.py`

## Part 2: Multi-Hop Retrieval

Detect questions containing multiple information needs and decompose them into focused subqueries.

Example:

> Compare how Transformers remove recurrence with how HNSW avoids exhaustive vector search.

Search each subquery independently, combine the rankings with Reciprocal Rank Fusion, and retrieve evidence for every part of the question.

Suggested module: `retrieval/query_decomposer.py`

## Part 3: Grounded Answers

The final pipeline must:

- Cite the source and section for major claims.
- Display generated subqueries and retrieved parent IDs.
- Avoid citing documents that were not retrieved.
- Return `Insufficient evidence` when the corpus cannot support an answer.

## Benchmark

Create `eval/advanced_golden_set.json` with at least 20 questions:

| Question type | Minimum |
|---|---:|
| Exact or lexical | 5 |
| Semantic paraphrase | 5 |
| Multi-hop within one document | 3 |
| Multi-hop across documents | 3 |
| Metadata-constrained | 2 |
| Unanswerable | 2 |

Compare these pipelines:

1. Dense baseline
2. Hybrid retrieval
3. Rewrite -> Hybrid -> Reranker
4. Parent-Child Hybrid
5. Full Parent-Child Multi-Hop pipeline

Report:

- Evidence Recall@5 and MRR
- Multi-hop evidence coverage
- Median latency
- A brief comparison of quality and speed

Keep the corpus, embedding model, Top-K, and reranker candidate count fixed. Compare the full pipeline with one ablation that removes either decomposition, parent expansion, or reranking.

## Deliverables

- Working implementation with tests
- A short `BENCHMARK_REPORT.md` containing the results table and two failure cases
