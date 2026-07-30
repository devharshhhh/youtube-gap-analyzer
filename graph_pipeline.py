from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.research_agent import research_topic
from agents.youtube_agent import get_top_videos
from agents.gap_analysis import analyze_gaps
from agents.script_agent import generate_script
from ingestion.chunker import fixed_size_chunk
from ingestion.embed_and_store import store_chunks
from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank


# ---- Shared state passed between every node ----
class PipelineState(TypedDict):
    topic: str
    want_script: bool
    research: Optional[dict]
    videos: Optional[list]
    retrieved_chunks: Optional[list]
    brief: Optional[str]
    script: Optional[str]
    cost_breakdown: dict
    total_cost_usd: float


# ---- Node: Research Agent ----
def research_node(state: PipelineState) -> PipelineState:
    print("[Research Agent] Searching the web...")
    research = research_topic(state["topic"])
    state["research"] = research
    state["cost_breakdown"]["research"] = research["cost"]
    return state


# ---- Node: Search Agent (YouTube search + transcript/whisper fallback) ----
def youtube_node(state: PipelineState) -> PipelineState:
    print("[Search Agent] Fetching YouTube videos...")
    videos = get_top_videos(state["topic"])
    state["videos"] = videos
    return state


# ---- Node: Chunking Agent ----
def chunk_node(state: PipelineState) -> PipelineState:
    print("[Chunking Agent] Splitting content into chunks...")
    topic = state["topic"]

    research_chunks = fixed_size_chunk(state["research"]["research_text"])
    store_chunks(topic, research_chunks, source_type="research", source_name="web_research")

    for v in state["videos"]:
        if v["transcript"]:
            video_chunks = fixed_size_chunk(v["transcript"])
            store_chunks(topic, video_chunks, source_type="youtube", source_name=v["title"])

    return state


# ---- Node: Retrieval Agent (hybrid search + re-rank) ----
def retrieve_node(state: PipelineState) -> PipelineState:
    print("[Retrieval Agent] Running hybrid search + re-ranking...")
    topic = state["topic"]
    candidates = hybrid_search(topic, query=topic, top_k=15)
    reranked = rerank(topic, candidates, top_k=10)
    state["retrieved_chunks"] = reranked
    return state


# ---- Node: Gap Agent ----
def gap_analysis_node(state: PipelineState) -> PipelineState:
    print("[Gap Agent] Analyzing content gaps...")
    result = analyze_gaps(state["topic"], state["retrieved_chunks"])
    state["brief"] = result["brief"]
    state["cost_breakdown"]["gap_analysis"] = result["cost"]
    return state


# ---- Node: Script Agent (conditional — only runs if requested) ----
def script_node(state: PipelineState) -> PipelineState:
    print("[Script Agent] Generating script...")
    result = generate_script(state["brief"])
    state["script"] = result["script"]
    state["cost_breakdown"]["script"] = result["cost"]
    return state


# ---- Conditional routing: skip script node entirely if not requested ----
def route_after_gap_analysis(state: PipelineState) -> str:
    return "script_node" if state["want_script"] else "finalize_node"


# ---- Node: Finalize (compute total cost) ----
def finalize_node(state: PipelineState) -> PipelineState:
    total = sum(c["total_cost_usd"] for c in state["cost_breakdown"].values())
    state["total_cost_usd"] = round(total, 6)
    return state


# ---- Build the graph ----
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("research_node", research_node)
    graph.add_node("youtube_node", youtube_node)
    graph.add_node("chunk_node", chunk_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("gap_analysis_node", gap_analysis_node)
    graph.add_node("script_node", script_node)
    graph.add_node("finalize_node", finalize_node)

    graph.set_entry_point("research_node")
    graph.add_edge("research_node", "youtube_node")
    graph.add_edge("youtube_node", "chunk_node")
    graph.add_edge("chunk_node", "retrieve_node")
    graph.add_edge("retrieve_node", "gap_analysis_node")

    # Conditional branch: script generation is optional
    graph.add_conditional_edges(
        "gap_analysis_node",
        route_after_gap_analysis,
        {"script_node": "script_node", "finalize_node": "finalize_node"}
    )
    graph.add_edge("script_node", "finalize_node")
    graph.add_edge("finalize_node", END)

    return graph.compile()


# ---- Entry point, mirrors main.py's run_pipeline() ----
from observability.tracker import Timer, log_query

def run_pipeline_graph(topic: str, want_script: bool = False) -> dict:
    app = build_graph()
    initial_state: PipelineState = {
        "topic": topic,
        "want_script": want_script,
        "research": None,
        "videos": None,
        "retrieved_chunks": None,
        "brief": None,
        "script": None,
        "cost_breakdown": {},
        "total_cost_usd": 0.0,
    }

    timings = {}
    with Timer("full_pipeline", timings):
        final_state = app.invoke(initial_state)

    # Aggregate token counts across all LLM calls for logging
    total_input_tokens = sum(c["input_tokens"] for c in final_state["cost_breakdown"].values())
    total_output_tokens = sum(c["output_tokens"] for c in final_state["cost_breakdown"].values())

    token_counts = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_cost_usd": final_state["total_cost_usd"],
    }

    log_query(topic, topic, timings, token_counts=token_counts)

    return final_state


if __name__ == "__main__":
    topic = input("Enter a video topic: ")
    result = run_pipeline_graph(topic, want_script=True)

    print("\n--- BRIEF ---\n")
    print(result["brief"])

    print("\n--- SCRIPT ---\n")
    print(result.get("script", "(not generated)"))

    print("\n--- COST BREAKDOWN ---")
    for stage, cost in result["cost_breakdown"].items():
        print(f"  {stage}: ${cost['total_cost_usd']:.6f}")
    print(f"  TOTAL: ${result['total_cost_usd']:.6f}")