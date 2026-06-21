"""
Chroma Vector Store

Lightweight embedded vector database. Persistence + metadata filtering
without a separate server process. Good middle ground between FAISS and
a full database like Qdrant.

Indexing: HNSW (Hierarchical Navigable Small World) via the hnswlib library.
Same graph-based ANN algorithm as Qdrant — multi-layer navigable graph with
O(log n) search. Configured with hnsw:space = "cosine". At small scale the
performance difference vs. brute-force (FAISS Flat) is negligible; HNSW
shines at 100K+ vectors where brute-force becomes too slow.
"""

import chromadb
#hnswlib is the library that chroma uses for the index

class ChromaStore:
    def __init__(self, collection_name="chunks", persist_dir=None):
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.EphemeralClient()
        self.collection_name = collection_name

        existing = [c.name for c in self.client.list_collections()]
        if collection_name in existing:
            self.client.delete_collection(collection_name)

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks, embeddings):
        offset = self.collection.count()
        ids = [str(offset + i) for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]

        clean_metadatas = []
        for m in metadatas:
            clean_metadatas.append({
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in m.items()
            })

        emb_list = [e.tolist() if hasattr(e, "tolist") else e for e in embeddings]

        self.collection.add(
            ids=ids,
            embeddings=emb_list,
            documents=documents,
            metadatas=clean_metadatas,
        )

    def search(self, query_embedding, k=5, where_filter=None):
        query = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        kwargs = {"query_embeddings": [query], "n_results": k}
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": results["distances"][0][i] if results["distances"] else 0.0,
            })
        return hits

    @property
    def count(self):
        return self.collection.count()
