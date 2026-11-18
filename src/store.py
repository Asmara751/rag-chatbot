import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import build_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")

def index_documents(folder: str = "data", batch_size: int = 200):
    chunks = build_chunks(folder)
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts).tolist()
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
        )
        print(f"Indexed {min(i + batch_size, total)}/{total} chunks")

def library_size() -> int:
    """How many chunks are in the built-in (persistent) library."""
    return collection.count()

def search(question: str, k: int = 3):
    n = collection.count()
    if n == 0:
        return []
    q_emb = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=min(k, n))
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))

def build_memory_index(chunks: list[dict]) -> dict:
    """Embed chunks and keep them in memory (used for uploaded files)."""
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return {"chunks": chunks, "emb": np.asarray(emb)}

def search_in_memory(question: str, index: dict, k: int = 6):
    q = model.encode([question], normalize_embeddings=True)[0]
    sims = index["emb"] @ q
    top = np.argsort(-sims)[:k]
    results = []
    for i in top:
        c = index["chunks"][i]
        results.append((c["text"], {"source": c["source"], "page": c["page"]}, float(1 - sims[i])))
    return results

if __name__ == "__main__":
    index_documents()
    for text, meta, dist in search("What is memory ballooning?"):
        print(f"\n[{meta['source']} p.{meta['page']}] distance={dist:.3f}")
        print(text[:200])