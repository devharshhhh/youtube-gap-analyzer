from rank_bm25 import BM25Okapi
from ingestion.embed_and_store import get_or_create_collection, embedding_model
from retrieval.redis_cache import cache_chunks, get_cached_chunks

_bm25_index_cache = {}  # topic -> BM25Okapi object (still in-memory, cheap to rebuild)


def _build_bm25_index(topic: str):
    # Try Redis first for the chunk data
    cached = get_cached_chunks(topic)
    if cached:
        ids, documents, metadatas = cached
        print(f"[Redis] Cache hit for topic '{topic}'")
    else:
        print(f"[Redis] Cache miss for topic '{topic}', fetching from ChromaDB...")
        collection = get_or_create_collection(topic)
        data = collection.get(include=["documents", "metadatas"])
        ids, documents, metadatas = data["ids"], data["documents"], data["metadatas"]
        cache_chunks(topic, ids, documents, metadatas)

    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    _bm25_index_cache[topic] = (bm25, ids, documents, metadatas)
    return _bm25_index_cache[topic]


def get_all_chunks(topic: str):
    if topic not in _bm25_index_cache:
        _build_bm25_index(topic)
    _, ids, documents, metadatas = _bm25_index_cache[topic]
    return ids, documents, metadatas


def bm25_search(topic: str, query: str, top_k: int = 10):
    if topic not in _bm25_index_cache:
        _build_bm25_index(topic)

    bm25, ids, documents, metadatas = _bm25_index_cache[topic]
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(ids, documents, metadatas, scores), key=lambda x: x[3], reverse=True)
    return ranked[:top_k]


def dense_search(topic: str, query: str, top_k: int = 10):
    collection = get_or_create_collection(topic)
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    ranked = list(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))
    return ranked


def hybrid_search(topic: str, query: str, top_k: int = 8, rrf_k: int = 60):
    dense_results = dense_search(topic, query, top_k=20)
    bm25_results = bm25_search(topic, query, top_k=20)

    rrf_scores = {}
    chunk_lookup = {}

    for rank, (chunk_id, doc, meta, _) in enumerate(dense_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + rrf_k)
        chunk_lookup[chunk_id] = (doc, meta)

    for rank, (chunk_id, doc, meta, _) in enumerate(bm25_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + rrf_k)
        chunk_lookup[chunk_id] = (doc, meta)

    ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {"id": cid, "text": chunk_lookup[cid][0], "metadata": chunk_lookup[cid][1], "rrf_score": score}
        for cid, score in ranked_ids
    ]