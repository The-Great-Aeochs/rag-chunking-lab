# RAG Chunking Lab

Hands-on comparison of chunking strategies and vector databases for RAG.

## Run locally

```bash
git clone https://github.com/The-Great-Aeochs/rag-chunking-lab.git
cd rag-chunking-lab

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
python app.py
```

Open http://127.0.0.1:7860 in a browser. To use a different port:

```bash
GRADIO_SERVER_PORT=7861 python app.py
```

The first run downloads the embedding model. Drop additional PDFs into
`papers/`; *Attention Is All You Need* is already included.

## Choose the generation model

The provider is controlled by `.env`. OpenAI remains the default.

### OpenAI

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o
```

### Local Hugging Face Qwen

To run an open model on the local machine without an OpenAI API key:

```dotenv
LLM_PROVIDER=huggingface
HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

Restart `python app.py` after changing `.env`. On the first Qwen request,
Transformers downloads several GB of model files from Hugging Face and caches
them locally. Generation uses CUDA, Apple Metal (MPS), or CPU, in that order;
CPU generation is slower. The selected provider is used for RAG answers, RAG
Fusion query variants, and HyDE passages.

`Qwen/Qwen2.5-1.5B-Instruct` is a public Apache-2.0 model, so this setup does
not require a Hugging Face token. It is a useful classroom-sized example, but
its answers will generally be weaker than larger hosted models.

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
python rag/pipeline.py "What optimizer is used?" --chunker Recursive --store FAISS
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
