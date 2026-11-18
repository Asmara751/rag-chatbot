import chromadb
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

def search(question: str, k: int = 3):
    q_emb = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))

if __name__ == "__main__":
    index_documents()
    for text, meta, dist in search("What is memory ballooning?"):
        print(f"\n[{meta['source']} p.{meta['page']}] distance={dist:.3f}")
        print(text[:200])