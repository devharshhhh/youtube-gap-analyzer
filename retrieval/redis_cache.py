import redis
import json

_redis_client = None


def get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    return _redis_client


def cache_chunks(topic: str, ids: list, documents: list, metadatas: list, ttl_seconds: int = 3600):
    """Cache chunk data in Redis with an expiry (default 1 hour)."""
    client = get_redis_client()
    key = f"chunks:{topic}"
    payload = json.dumps({"ids": ids, "documents": documents, "metadatas": metadatas})
    client.setex(key, ttl_seconds, payload)


def get_cached_chunks(topic: str):
    """Returns (ids, documents, metadatas) or None if not cached."""
    client = get_redis_client()
    key = f"chunks:{topic}"
    payload = client.get(key)
    if payload is None:
        return None
    data = json.loads(payload)
    return data["ids"], data["documents"], data["metadatas"]