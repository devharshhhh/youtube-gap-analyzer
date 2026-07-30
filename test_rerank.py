from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank

topic = "machine learning basics"
query = "supervised learning algorithm"

candidates = hybrid_search(topic, query, top_k=10)

print("=== BEFORE RE-RANKING (hybrid order) ===")
for c in candidates:
    print(f"rrf={c['rrf_score']:.4f} | {c['text'][:100]}...")

reranked = rerank(query, candidates, top_k=5)

print("\n=== AFTER RE-RANKING ===")
for c in reranked:
    print(f"rerank={c['rerank_score']:.4f} | {c['text'][:100]}...")