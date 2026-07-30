from ingestion.chunker import fixed_size_chunk
from ingestion.embed_and_store import store_chunks
from agents.youtube_agent import get_top_videos

# One shared "topic" collection so all this content is searchable together —
# gives real distractor chunks across related-but-different subjects.
SHARED_TOPIC = "eval_corpus"

SEARCH_QUERIES = [
    "machine learning basics",
    "deep learning explained",
    "python programming tutorial",
    "data science for beginners",
]

for query in SEARCH_QUERIES:
    print(f"\n--- Fetching videos for: {query} ---")
    videos = get_top_videos(query, max_results=4)

    for v in videos:
        if not v["transcript"]:
            print(f"Skipped (no content): {v['title']}")
            continue

        chunks = fixed_size_chunk(v["transcript"])
        store_chunks(SHARED_TOPIC, chunks, source_type="youtube", source_name=v["title"])
        print(f"Stored {len(chunks)} chunks from: {v['title']}")

print("\nDone. Run eval/list_chunks.py (pointed at 'eval_corpus') to see the full corpus.")