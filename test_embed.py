from ingestion.chunker import fixed_size_chunk
from ingestion.embed_and_store import store_chunks, query_collection
from agents.youtube_agent import get_top_videos

topic = "machine learning basics"
videos = get_top_videos(topic, max_results=3)

for v in videos:
    chunks = fixed_size_chunk(v["transcript"])
    store_chunks(topic, chunks, source_type="youtube", source_name=v["title"])
    print(f"Stored {len(chunks)} chunks from: {v['title']}")

print("\n--- Test query ---")
results = query_collection(topic, "what is supervised learning")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"\n[{meta['source_name']}] {doc[:150]}...")