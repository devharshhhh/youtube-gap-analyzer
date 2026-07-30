from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list, top_k: int = 5):
    """
    candidates: list of dicts from hybrid_search(), each with a 'text' key
    Returns the same dicts, re-sorted by cross-encoder relevance score.
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]