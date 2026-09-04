import streamlit as st

from agents.research_agent import research_topic
from agents.youtube_agent import get_top_videos
from agents.gap_analysis import analyze_gaps
from agents.script_agent import generate_script
from ingestion.chunker import fixed_size_chunk
from ingestion.embed_and_store import store_chunks
from retrieval.hybrid_search import hybrid_search
from retrieval.rerank import rerank

st.set_page_config(page_title="TubeScope", page_icon="🎬", layout="wide")


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


# ---------------------------------------------------------------------------
# Streamlit UI — this is what was missing. The old file only had a
# `if __name__ == "__main__":` block that called input(), which blocks
# forever on a server with no attached terminal (that's why the deployed
# app sat on a blank page with a clean-looking log). Every code path below
# uses st.* widgets instead, so Streamlit actually has something to render.
# ---------------------------------------------------------------------------

st.title("🎬 TubeScope")
st.caption("AI-powered YouTube content gap analyzer")

with st.form("analyze_form"):
    topic = st.text_input("Video topic", placeholder="e.g. beginner woodworking tools")
    want_script = st.checkbox("Also generate a script", value=False)
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    if not topic.strip():
        st.warning("Enter a topic first.")
        st.stop()

    with st.spinner("Researching, retrieving, and finding content gaps..."):
        try:
            result = run_pipeline(topic, want_script=want_script)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    st.subheader("Content gap brief")
    st.write(result["brief"])

    if want_script:
        st.subheader("Generated script")
        st.write(result.get("script", "(not generated)"))

    with st.expander("Sources used"):
        for s in result.get("sources", []):
            st.write(s)

    with st.expander("YouTube videos analyzed"):
        for v in result.get("videos", []):
            st.write(v.get("title", "Untitled"))

    st.subheader("Cost breakdown")
    for stage, cost in result["cost_breakdown"].items():
        st.write(f"**{stage}**: ${cost['total_cost_usd']:.6f}")
    st.write(f"**Total: ${result['total_cost_usd']:.6f}**")
else:
    st.info("Enter a topic above and click Analyze to get started.")
