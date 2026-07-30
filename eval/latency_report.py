import json
import statistics

LOG_FILE = "observability/query_log.jsonl"


def load_log():
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def percentile(data: list, p: float) -> float:
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p) 
    idx = min(idx, len(data_sorted) - 1)
    return data_sorted[idx]


def report():
    entries = load_log()
    if not entries:
        print("No log entries found yet — run test_instrumented_query.py a few times first.")
        return

    totals = [e["total_seconds"] for e in entries]
    hybrid_times = [e["timings_seconds"].get("hybrid_search", 0) for e in entries]
    rerank_times = [e["timings_seconds"].get("rerank", 0) for e in entries]

    print(f"Total queries logged: {len(entries)}\n")

    print("=== Total pipeline latency ===")
    print(f"  p50: {percentile(totals, 0.50):.3f}s")
    print(f"  p95: {percentile(totals, 0.95):.3f}s")
    print(f"  p99: {percentile(totals, 0.99):.3f}s")
    print(f"  avg: {statistics.mean(totals):.3f}s")

    print("\n=== Breakdown by stage (avg) ===")
    print(f"  hybrid_search: {statistics.mean(hybrid_times):.3f}s")
    print(f"  rerank:        {statistics.mean(rerank_times):.3f}s")


if __name__ == "__main__":
    report()