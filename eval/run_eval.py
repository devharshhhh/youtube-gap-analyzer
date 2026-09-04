from eval.eval_set import EVAL_SET
from retrieval.hybrid_search import dense_search, bm25_search, hybrid_search
from retrieval.rerank import rerank

TOPIC = "eval_corpus"

def recall_at_k(retrieved_ids: list, correct_ids: list) -> int:
    """1 if ANY correct chunk appears in retrieved, else 0."""
    return 1 if any(cid in retrieved_ids for cid in correct_ids) else 0


def reciprocal_rank(retrieved_ids: list, correct_ids: list) -> float:
    """1/rank of the FIRST correct chunk found, or 0 if none found."""
    for i, rid in enumerate(retrieved_ids):
        if rid in correct_ids:
            return 1 / (i + 1)
    return 0.0


def evaluate_method(method_name: str, get_ids_fn, k: int = 5):
    recalls = []
    rrs = []

    for example in EVAL_SET:
        query = example["query"]
        correct_ids = example["correct_chunk_ids"]

        retrieved_ids = get_ids_fn(query)[:k]

        recalls.append(recall_at_k(retrieved_ids, correct_ids))
        rrs.append(reciprocal_rank(retrieved_ids, correct_ids))

    avg_recall = sum(recalls) / len(recalls)
    avg_mrr = sum(rrs) / len(rrs)

    print(f"\n{method_name}")
    print(f"  Recall@{k}: {avg_recall:.2%}")
    print(f"  MRR:      {avg_mrr:.4f}")
    return avg_recall, avg_mrr


def get_dense_ids(query):
    results = dense_search(TOPIC, query, top_k=10)
    return [cid for cid, doc, meta, dist in results]


def get_bm25_ids(query):
    results = bm25_search(TOPIC, query, top_k=10)
    return [cid for cid, doc, meta, score in results]


def get_hybrid_ids(query):
    results = hybrid_search(TOPIC, query, top_k=10)
    return [r["id"] for r in results]


def get_hybrid_reranked_ids(query):
    candidates = hybrid_search(TOPIC, query, top_k=10)
    reranked = rerank(query, candidates, top_k=10)
    return [r["id"] for r in reranked]


if __name__ == "__main__":
    print("=" * 50)
    print(f"Evaluating on {len(EVAL_SET)} queries")
    print("=" * 50)

    evaluate_method("Dense only", get_dense_ids)
    evaluate_method("BM25 only", get_bm25_ids)
    evaluate_method("Hybrid (RRF)", get_hybrid_ids)
    evaluate_method("Hybrid + Re-ranked", get_hybrid_reranked_ids)
