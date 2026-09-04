from retrieval.hybrid_search import dense_search, bm25_search, hybrid_search

topic = "machine learning basics"
query = "supervised learning algorithm"

print("=== DENSE ONLY ===")
for cid, doc, meta, score in dense_search(topic, query, top_k=3):
    print(f"[{meta['source_name'][:40]}] {doc[:100]}...")

print("\n=== BM25 ONLY ===")
for cid, doc, meta, score in bm25_search(topic, query, top_k=3):
    print(f"[{meta['source_name'][:40]}] {doc[:100]}...")

print("\n=== HYBRID (RRF) ===")
for r in hybrid_search(topic, query, top_k=3):
    print(f"[{r['metadata']['source_name'][:40]}] score={r['rrf_score']:.4f} {r['text'][:100]}...")
