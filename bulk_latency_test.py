from observability.tracker import Timer, log_query
from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank

topic = "eval_corpus"

queries = [
    "how does a neural network recognize handwritten digits",
    "what is supervised learning",
    "how to install python",
    "pandas dataframe indexing",
    "what is data science",
    "correlation between two variables",
    "reinforcement learning agent environment",
    "deep learning brain inspired",
    "python data structures list set",
    "linear regression coefficients",
    "who is bhagat singh",
    "how i can start learning coding",
    "how tomake my very first game",
    "unreal or unity, whch one is good for begineers",
    "top niche to create content on youtube",
    "highest earning artist from youtube",
    "what is the future of AI",
]

for query in queries:
    timings = {}
    with Timer("hybrid_search", timings):
        candidates = hybrid_search(topic, query, top_k=10)
    with Timer("rerank", timings):
        rerank(query, candidates, top_k=5)

    entry = log_query(topic, query, timings)
    print(f"{query[:40]:40s} total={entry['total_seconds']:.3f}s")