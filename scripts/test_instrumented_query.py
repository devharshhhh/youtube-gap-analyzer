from observability.tracker import Timer, log_query
from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank

topic = "eval_corpus"
query = "how to make my first game as a begineer"

timings = {}

with Timer("hybrid_search", timings):
    candidates = hybrid_search(topic, query, top_k=10)

with Timer("rerank", timings):
    results = rerank(query, candidates, top_k=5)

entry = log_query(topic, query, timings)

print(f"hybrid_search: {timings['hybrid_search']:.3f}s")
print(f"rerank:        {timings['rerank']:.3f}s")
print(f"TOTAL:         {entry['total_seconds']:.3f}s")