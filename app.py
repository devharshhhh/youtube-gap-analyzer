import streamlit as st
from pipeline import run_pipeline
from agents.script_agent import generate_script
from eval.faithfulness import faithfulness_score

st.set_page_config(page_title="TubeScope — YouTube Gap Analyzer", page_icon="🎬", layout="centered")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; max-width: 760px; }

    .ts-hero {
        background: linear-gradient(135deg, #FF3B3B 0%, #7A1FFF 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 28px;
    }
    .ts-hero h1 { margin: 0 0 4px 0; font-size: 1.7rem; color: white; }
    .ts-hero p { margin: 0; opacity: 0.9; font-size: 0.95rem; }

    .ts-step-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.75rem;
        font-weight: 700;
        color: #7A1FFF;
        margin-bottom: 4px;
    }

    div.stButton > button[kind="primary"] {
        background: #FF3B3B;
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #E02F2F;
    }

    .ts-badge {
        display: inline-block;
        background: #F1EEFF;
        color: #7A1FFF;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px 4px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — project identity, so this reads like a project, not just a form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 TubeScope")
    st.caption("Analyze YouTube like a researcher, not a viewer.")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. You give it a topic\n"
        "2. It researches the web + top YouTube videos\n"
        "3. It finds what's *missing* from existing coverage\n"
        "4. You turn that gap into a script"
    )
    st.markdown("---")
    st.markdown(
        '<span class="ts-badge">RAG</span>'
        '<span class="ts-badge">Hybrid Search</span>'
        '<span class="ts-badge">Whisper</span>'
        '<span class="ts-badge">LangGraph</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("[View source on GitHub](https://github.com/devharshhhh/youtube-gap-analyzer)")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="ts-hero">
    <h1>🎬 TubeScope</h1>
    <p>Find what every top video on your topic is missing — then write the script that fills it.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Step 1 — input
# ---------------------------------------------------------------------------
st.markdown('<div class="ts-step-label">Step 1 · Your topic</div>', unsafe_allow_html=True)
topic = st.text_input(
    "Topic", placeholder="e.g. beginner woodworking tools", label_visibility="collapsed"
)
go = st.button("🔍 Find the gap", type="primary")

if go and topic:
    with st.spinner("Researching the web and top YouTube videos, then comparing coverage..."):
        result = run_pipeline(topic, want_script=False)
    st.session_state["result"] = result
    st.session_state["topic"] = topic
    # Clear any stale script/faithfulness from a previous topic
    st.session_state.pop("script_result", None)
    st.session_state.pop("faithfulness", None)
elif go and not topic:
    st.warning("Enter a topic first.")

# ---------------------------------------------------------------------------
# Step 2 — the gap, front and center, with the script CTA right below it.
# This is the fix: script generation used to live at the very bottom, after
# five expanders of diagnostics. Now it's the very next thing you see after
# the answer to "what's missing" — because that's the natural next question.
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state["result"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ts-step-label">Step 2 · The gap we found</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(result["brief"])

    st.markdown('<div class="ts-step-label" style="margin-top:20px;">Step 3 · Turn it into a script?</div>', unsafe_allow_html=True)
    st.caption("This gap is your unique angle. Generate a script built specifically around it.")

    if st.button("📝 Generate script from this gap", type="primary"):
        with st.spinner("Writing a script around the gap..."):
            script_result = generate_script(result["brief"])
        st.session_state["script_result"] = script_result

    if "script_result" in st.session_state:
        script_result = st.session_state["script_result"]
        with st.container(border=True):
            st.markdown("**Your script**")
            st.text_area("Generated script", script_result["script"], height=350, label_visibility="collapsed")
            st.download_button(
                "⬇️ Download script (.txt)",
                script_result["script"],
                file_name=f"{st.session_state.get('topic', 'script').replace(' ', '_')}.txt",
            )
        st.caption(f"Script generation cost: ${script_result['cost']['total_cost_usd']:.6f}")

    # -----------------------------------------------------------------
    # Everything below is *evidence*, not the main flow — collapsed by
    # default so it doesn't compete with the brief/script for attention.
    # -----------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 Research details — sources, cost, and how grounded the brief is"):

        st.markdown(f"**Total cost so far:** ${result['total_cost_usd']:.6f}")
        for stage, cost in result["cost_breakdown"].items():
            st.markdown(
                f"- **{stage}**: ${cost['total_cost_usd']:.6f} "
                f"({cost['input_tokens']} in / {cost['output_tokens']} out tokens)"
            )

        st.markdown("---")
        st.markdown(f"**{len(result['sources'])} web sources used**")
        for s in result["sources"]:
            st.markdown(f"- [{s['title']}]({s['url']})")

        st.markdown("---")
        st.markdown(f"**{len(result['videos'])} YouTube videos analyzed**")
        for v in result["videos"]:
            source_tag = {
                "transcript": "📝 transcript",
                "whisper_transcription": "🎙️ whisper transcription",
            }.get(v.get("content_source"), "⚠️ title/description only")
            st.markdown(f"- **{v['title']}** — {v['channel']} ({source_tag})")

        st.markdown("---")
        st.markdown(f"**{len(result['retrieved_chunks'])} retrieved chunks used for this brief**")
        for c in result["retrieved_chunks"]:
            st.markdown(
                f"**[{c['metadata']['source_type']}]** "
                f"rerank_score={c.get('rerank_score', 0):.2f} — "
                f"{c['text'][:150]}..."
            )

        st.markdown("---")
        if st.button("✅ Check faithfulness (verify brief is grounded in sources)"):
            with st.spinner("Checking each claim against retrieved sources..."):
                faith = faithfulness_score(result["brief"], result["retrieved_chunks"])
            st.session_state["faithfulness"] = faith

        if "faithfulness" in st.session_state:
            faith = st.session_state["faithfulness"]
            st.metric(
                "Faithfulness Score", f"{faith['score']:.1%}",
                help="Fraction of claims in the brief traceable to retrieved sources",
            )
            for c in faith["claims"]:
                icon = "✅" if c["supported"] else "❌"
                st.markdown(f"{icon} {c['claim']}")
