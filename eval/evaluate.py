import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "src")
from ingest import load_pdfs, chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")
pages = load_pdfs("data")
questions = json.load(open("eval/questions.json", encoding="utf-8"))
KS = (1, 3, 6)

def first_ranks(chunk_size, overlap):
    chunks = []
    for p in pages:
        for c in chunk_text(p["text"], chunk_size, overlap):
            chunks.append({"text": c, "source": p["source"], "page": p["page"]})
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    q_emb = model.encode([q["question"] for q in questions], normalize_embeddings=True)
    ranking = np.argsort(-(q_emb @ emb.T), axis=1)

    ranks = []
    for qi, q in enumerate(questions):
        rank = None
        for r, idx in enumerate(ranking[qi][:10], start=1):
            c = chunks[idx]
            if c["source"] == q["source"] and c["page"] in q["pages"]:
                rank = r
                break
        ranks.append(rank)
    return len(chunks), ranks

print(f"{len(questions)} questions\n")
print(f"{'size':>5} {'overlap':>7} {'chunks':>6} {'hit@1':>6} {'hit@3':>6} {'hit@6':>6} {'MRR':>6}")
for size, overlap in [(400, 80), (800, 150), (1200, 200)]:
    n_chunks, ranks = first_ranks(size, overlap)
    n = len(ranks)
    hit = {k: sum(1 for r in ranks if r and r <= k) / n for k in KS}
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"{size:>5} {overlap:>7} {n_chunks:>6} {hit[1]:>6.2f} {hit[3]:>6.2f} {hit[6]:>6.2f} {mrr:>6.2f}")
    for q, r in zip(questions, ranks):
        print(f"        rank={r if r else '>10':>3}  {q['question']}")