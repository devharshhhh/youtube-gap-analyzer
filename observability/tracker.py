import time
import json
import os
from datetime import datetime

LOG_FILE = "observability/query_log.jsonl"


class Timer:
    """Context manager for timing a pipeline stage."""
    def __init__(self, name: str, results: dict):
        self.name = name
        self.results = results

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.results[self.name] = elapsed


def log_query(topic: str, query: str, timings: dict, token_counts: dict = None):
    """Append one query's timing (and optional token/cost data) to a log file."""
    os.makedirs("observability", exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "query": query,
        "timings_seconds": timings,
        "total_seconds": sum(timings.values()),
        "token_counts": token_counts or {},
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry