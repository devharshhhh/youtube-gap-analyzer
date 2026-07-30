from ingestion.chunker import fixed_size_chunk, semantic_chunk
from agents.youtube_agent import get_top_videos

videos = get_top_videos("machine learning basics", max_results=3)  # try 3 videos instead of 1

for v in videos:
    print(f"Title: {v['title']}")
    print(f"Transcript length: {len(v['transcript'])} characters\n")

# Use whichever video actually has a transcript
transcript = next((v["transcript"] for v in videos if v["transcript"]), None)

if not transcript:
    print("None of these videos had a usable transcript. Try a different topic or more results.")
else:
    print("=== FIXED-SIZE CHUNKS ===")
    for i, c in enumerate(fixed_size_chunk(transcript)):
        print(f"[{i}] {c[:120]}...\n")

    print("=== SEMANTIC CHUNKS ===")
    for i, c in enumerate(semantic_chunk(transcript)):
        print(f"[{i}] {c[:120]}...\n")