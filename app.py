import streamlit as st
from main import run_pipeline
from agents.script_agent import generate_script
from eval.faithfulness import faithfulness_score

st.set_page_config(page_title="Content Research Assistant", layout="centered")
st.title("🎬 Content Research Assistant")
st.caption("Research any topic, see what YouTube already covers, and find your unique angle.")

topic = st.text_input("Enter your video topic")

if st.button("Research", type="primary") and topic:
    with st.spinner("Researching topic, analyzing YouTube, and retrieving relevant content..."):
        result = run_pipeline(topic, want_script=False)
    st.session_state["result"] = result
    st.session_state["topic"] = topic
    # Clear any stale script/faithfulness from a previous topic
    st.session_state.pop("script_result", None)
    st.session_state.pop("faithfulness", None)

if "result" in st.session_state:
    result = st.session_state["result"]

    st.markdown("## Content Brief")
    st.markdown(result["brief"])

    st.divider()

    # --- Cost breakdown ---
    with st.expander(f"💰 Cost: ${result['total_cost_usd']:.6f} total"):
        for stage, cost in result["cost_breakdown"].items():
            st.markdown(f"- **{stage}**: ${cost['total_cost_usd']:.6f} "
                        f"({cost['input_tokens']} in / {cost['output_tokens']} out tokens)")

    # --- Sources used ---
    with st.expander(f"🔗 {len(result['sources'])} web sources used"):
        for s in result["sources"]:
            st.markdown(f"- [{s['title']}]({s['url']})")

    # --- YouTube videos analyzed ---
    with st.expander(f"📺 {len(result['videos'])} YouTube videos analyzed"):
        for v in result["videos"]:
            source_tag = {
                "transcript": "📝 transcript",
                "whisper_transcription": "🎙️ whisper transcription",
            }.get(v.get("content_source"), "⚠️ title/description only")
            st.markdown(f"- **{v['title']}** — {v['channel']} ({source_tag})")

    # --- Retrieved chunks with scores (shows the retrieval work) ---
    with st.expander(f"🔍 {len(result['retrieved_chunks'])} retrieved chunks used for this brief"):
        for c in result["retrieved_chunks"]:
            st.markdown(
                f"**[{c['metadata']['source_type']}]** "
                f"rerank_score={c.get('rerank_score', 0):.2f} — "
                f"{c['text'][:150]}..."
            )

    st.divider()

    # --- Faithfulness check (on demand, since it costs extra API calls) ---
    if st.button("✅ Check faithfulness (verify brief is grounded in sources)"):
        with st.spinner("Checking each claim against retrieved sources..."):
            faith = faithfulness_score(result["brief"], result["retrieved_chunks"])
        st.session_state["faithfulness"] = faith

    if "faithfulness" in st.session_state:
        faith = st.session_state["faithfulness"]
        st.metric("Faithfulness Score", f"{faith['score']:.1%}",
                   help="Fraction of claims in the brief traceable to retrieved sources")
        with st.expander("Per-claim breakdown"):
            for c in faith["claims"]:
                icon = "✅" if c["supported"] else "❌"
                st.markdown(f"{icon} {c['claim']}")

    st.divider()

    # --- Script generation (on demand) ---
    if st.button("📝 Generate Script"):
        with st.spinner("Writing script..."):
            script_result = generate_script(result["brief"])
        st.session_state["script_result"] = script_result

    if "script_result" in st.session_state:
        script_result = st.session_state["script_result"]
        st.markdown("## Script")
        st.text_area("Generated script", script_result["script"], height=400)
        st.caption(f"Script generation cost: ${script_result['cost']['total_cost_usd']:.6f}")