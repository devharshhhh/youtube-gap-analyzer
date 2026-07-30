from agents.research_agent import research_topic
from agents.youtube_agent import get_top_videos
from agents.gap_analysis import analyze_gaps
from agents.script_agent import generate_script
from ingestion.chunker import fixed_size_chunk
from ingestion.embed_and_store import store_chunks
from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank


def run_pipeline(topic: str, want_script: bool = False) -> dict:
    print(f"Researching: {topic}")
    research = research_topic(topic)

    print("Analyzing YouTube coverage...")
    videos = get_top_videos(topic)

    print("Chunking and storing content...")
    research_chunks = fixed_size_chunk(research["research_text"])
    store_chunks(topic, research_chunks, source_type="research", source_name="web_research")

    for v in videos:
        if v["transcript"]:
            video_chunks = fixed_size_chunk(v["transcript"])
            store_chunks(topic, video_chunks, source_type="youtube", source_name=v["title"])

    print("Retrieving most relevant content...")
    candidates = hybrid_search(topic, query=topic, top_k=15)
    retrieved_chunks = rerank(topic, candidates, top_k=10)

    print("Finding content gaps...")
    gap_result = analyze_gaps(topic, retrieved_chunks)
    brief = gap_result["brief"]

    total_cost = research["cost"]["total_cost_usd"] + gap_result["cost"]["total_cost_usd"]

    result = {
        "brief": brief,
        "sources": research["sources"],
        "videos": videos,
        "retrieved_chunks": retrieved_chunks,
        "cost_breakdown": {
            "research": research["cost"],
            "gap_analysis": gap_result["cost"],
        },
    }

    if want_script:
        print("Generating script...")
        script_result = generate_script(brief)
        result["script"] = script_result["script"]
        result["cost_breakdown"]["script"] = script_result["cost"]
        total_cost += script_result["cost"]["total_cost_usd"]

    result["total_cost_usd"] = round(total_cost, 6)

    return result


if __name__ == "__main__":
    topic = input("Enter a video topic: ")
    result = run_pipeline(topic, want_script=True)

    print("\n--- BRIEF ---\n")
    print(result["brief"])

    print("\n--- SCRIPT ---\n")
    print(result.get("script", "(not generated)"))

    print("\n--- COST BREAKDOWN ---")
    for stage, cost in result["cost_breakdown"].items():
        print(f"  {stage}: ${cost['total_cost_usd']:.6f}")
    print(f"  TOTAL: ${result['total_cost_usd']:.6f}")