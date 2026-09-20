# RAG Chunking Lab

Hands-on comparison of chunking strategies and vector databases for RAG.

## Setup

```bash
pip install -r requirements.txt
```

Drop PDFs into `papers/` (Attention Is All You Need is included).

## Usage

```bash
# Compare all 4 chunking methods
python chunking/compare.py

# Compare FAISS vs Qdrant vs Chroma
python vectordb/compare.py

# Full evaluation matrix: every chunker × every vector DB
python eval/run_eval.py

# End-to-end RAG pipeline
python rag/pipeline.py "What is multi-head attention?"
python rag/pipeline.py "What optimizer is used?" --chunker recursive --store faiss
```

## Structure

```
chunking/         4 chunking strategies (recursive, character, section, semantic)
vectordb/         3 vector stores (FAISS, Qdrant, Chroma)
eval/             Golden set + metrics (recall@k, MRR)
rag/              End-to-end retrieval pipeline
shared/           PDF loader + embedding model
papers/           Demo PDFs
```

## Chunking Methods

| Method | How it splits | Strength | Weakness |
|--------|--------------|----------|----------|
| Recursive | Paragraph → line → space → char fallback | Consistent sizes | No semantic awareness |
| Character | Single separator, then fixed-width fallback | Simple, bounded chunks | Fallback can cut through words |
| Section-wise | Regex header detection | Preserves paper structure | Pattern-dependent |
| Semantic | Embedding similarity drops | Topic-aligned boundaries | Slow (embeds during chunking) |

## Experiments to try

- Change `chunk_size` from 800 to 200 and rerun eval
- Add more PDFs and see which chunker scales best
- Edit `eval/golden_set.json` with your own questions
