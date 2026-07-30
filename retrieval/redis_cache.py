import redis
import json

_redis_client = None
_redis_available = None


def get_redis_client():
    global _redis_client, _redis_available
    if _redis_available is False:
        return None
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True, socket_connect_timeout=2)
            _redis_client.ping()
            _redis_available = True
        except Exception:
            _redis_available = False
            return None
    return _redis_client


def cache_chunks(topic: str, ids: list, documents: list, metadatas: list, ttl_seconds: int = 3600):
    client = get_redis_client()
    if client is None:
        return  # Redis unavailable, silently skip caching
    key = f"chunks:{topic}"
    payload = json.dumps({"ids": ids, "documents": documents, "metadatas": metadatas})
    client.setex(key, ttl_seconds, payload)


def get_cached_chunks(topic: str):
    client = get_redis_client()
    if client is None:
        return None  # Redis unavailable, treat as cache miss
    key = f"chunks:{topic}"
    payload = client.get(key)
    if payload is None:
        return None
    data = json.loads(payload)
    return data["ids"], data["documents"], data["metadatas"]