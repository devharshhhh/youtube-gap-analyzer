import json
import sys
from eval.run_eval import evaluate_method, get_dense_ids, get_bm25_ids, get_hybrid_ids, get_hybrid_reranked_ids

BASELINE_FILE = "eval/baseline_scores.json"
TOLERANCE = 0.05  # allow up to 5 percentage points of regression before failing


def load_baseline():
    with open(BASELINE_FILE) as f:
        return json.load(f)


def gate_check():
    baseline = load_baseline()
    failed = False

    print("=" * 50)
    print("EVAL GATE CHECK")
    print("=" * 50)

    for name, fn in [
        ("dense", get_dense_ids),
        ("bm25", get_bm25_ids),
        ("hybrid", get_hybrid_ids),
        ("hybrid_reranked", get_hybrid_reranked_ids),
    ]:
        recall, mrr = evaluate_method(name, fn)
        base_recall = baseline[name]["recall_at_5"]
        base_mrr = baseline[name]["mrr"]

        recall_drop = base_recall - recall
        mrr_drop = base_mrr - mrr

        status = "PASS"
        if recall_drop > TOLERANCE or mrr_drop > TOLERANCE:
            status = "FAIL"
            failed = True

        print(f"\n{name}: {status}")
        print(f"  Recall@5: {recall:.2%} (baseline: {base_recall:.2%}, drop: {recall_drop:.2%})")
        print(f"  MRR:      {mrr:.4f} (baseline: {base_mrr:.4f}, drop: {mrr_drop:.4f})")

    print("\n" + "=" * 50)
    if failed:
        print("GATE FAILED — quality regression detected")
        sys.exit(1)
    else:
        print("GATE PASSED — no significant regression")
        sys.exit(0)


if __name__ == "__main__":
    gate_check()
