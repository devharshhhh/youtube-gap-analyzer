import json
from eval.run_eval import evaluate_method, get_dense_ids, get_bm25_ids, get_hybrid_ids, get_hybrid_reranked_ids

BASELINE_FILE = "eval/baseline_scores.json"

def save_baseline():
    results = {}

    for name, fn in [
        ("dense", get_dense_ids),
        ("bm25", get_bm25_ids),
        ("hybrid", get_hybrid_ids),
        ("hybrid_reranked", get_hybrid_reranked_ids),
    ]:
        recall, mrr = evaluate_method(name, fn)
        results[name] = {"recall_at_5": recall, "mrr": mrr}

    with open(BASELINE_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline saved to {BASELINE_FILE}")
    return results


if __name__ == "__main__":
    save_baseline()
