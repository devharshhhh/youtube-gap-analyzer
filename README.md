![Eval Gate](https://github.com/devharshhhh/youtube-gap-analyzer/actions/workflows/eval-gate.yml/badge.svg)

# TubeScope - AI-Powered YouTube Gap Analyzer

> Analyze YouTube content like a researcher, not a viewer.

TubeScope is an AI-powered research assistant that helps YouTube creators discover content opportunities by analyzing existing videos, extracting transcripts, identifying knowledge gaps, and generating evidence-backed video ideas and scripts.

Instead of relying on intuition, TubeScope uses Retrieval-Augmented Generation (RAG), hybrid search, and LLM reasoning to answer one question:

> **"What valuable information is missing from the top YouTube videos on this topic?"**

---

## Demo

*(GIF/video coming soon)*

---

## Why?

Most creators spend hours:
- Watching competitor videos
- Taking notes
- Comparing information
- Reading articles
- Looking for unique angles

TubeScope automates this workflow. It searches YouTube, extracts or transcribes video content, performs hybrid semantic + keyword retrieval, analyzes what has already been covered, identifies missing information, and generates research-backed content suggestions — grounded in retrieved evidence, not the model's free-floating memory.

---

## Features

- Search YouTube videos by topic (YouTube Data API)
- Fetch official YouTube caption transcripts, with multi-language fallback
- Automatically transcribe videos using local Whisper when no captions are available
- Hybrid Retrieval (dense semantic search + BM25 keyword search, combined via Reciprocal Rank Fusion)
- Cross-encoder re-ranking for precision on top candidates
- Evidence-backed Gap Analysis (LLM compares retrieved research vs. retrieved YouTube coverage)
- AI-generated Content Briefs and Video Scripts, grounded in retrieved chunks
- Custom faithfulness scoring (LLM-as-judge groundedness check)
- Cost tracking per LLM call (tokens + USD)
- Latency benchmarking (p50/p95/p99)
- Redis-backed chunk caching (with graceful fallback if Redis is unavailable)
- Automated retrieval-quality eval gate (recall@k, MRR) running in CI on every push
- Observability dashboard (separate Streamlit app) for latency, cost, and retrieval quality

---

## Architecture

```
                   User
                     │
                     ▼
              Streamlit Interface
                     │
                     ▼
         Graph Pipeline (LangGraph)
                     │
      ┌──────────────┼───────────────┐
      ▼              ▼               │
 YouTube Agent   Research Agent      │
      │              │               │
      ▼              ▼               │
Captions/Whisper  Web Search         │
      │              │               │
      └──────┬───────┘               │
             ▼                       │
         Chunking                    │
             ▼                       │
   Embeddings + ChromaDB             │
             ▼                       │
   Hybrid Retrieval (Dense + BM25)   │
   (chunk cache: Redis, TTL-based)   │
             ▼                       │
     Cross-Encoder Re-ranking        │
             ▼                       │
        Gap Analysis  ◄──────────────┘
             ▼
      Content Brief
             ▼
     Script Generation (on request)
```

---

## Tech Stack

**Backend**
- Python
- Streamlit
- LangGraph

**AI / LLM**
- Groq API (Llama 3.3 70B Versatile)
- Whisper (via `faster-whisper`, local CPU inference)

**Retrieval**
- ChromaDB (vector store)
- `sentence-transformers` (embeddings: `all-MiniLM-L6-v2`)
- `rank_bm25` (sparse retrieval)
- Cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`)

**Infrastructure**
- Redis (chunk caching, run via Docker)
- Docker (Redis container)
- GitHub Actions (CI — automated eval gate)

**External APIs**
- YouTube Data API v3
- Tavily Search API

---

## Project Structure

```
youtube-gap-analyzer/
├── agents/              # research, YouTube, gap analysis, script generation
├── ingestion/            # chunking, embedding + storage
├── retrieval/            # hybrid search, re-ranking, Redis cache
├── eval/                 # eval harness, baseline scores, CI gate check
├── observability/        # latency/cost tracking + dashboard data
├── scripts/               # manual diagnostic/dev scripts (not automated tests)
├── chroma_db/             # sample vector DB, used by CI for eval gate testing
├── .github/workflows/    # GitHub Actions CI workflow
├── app.py                 # main Streamlit app
├── graph_pipeline.py      # LangGraph pipeline (main orchestration)
├── main.py                 # simpler sequential pipeline (pre-LangGraph version)
├── observability_dashboard.py  # separate observability Streamlit app
├── requirements.txt
└── README.md
```

> Note: `chroma_db/` in this repo is sample data used by the CI eval gate, not a live dataset. Running the app yourself will populate/query your own topic-specific data at runtime.

---

## Pipeline

1. User enters a topic.
2. **Research Agent** searches the live web (Tavily) and summarizes findings with source attribution.
3. **YouTube Agent** pulls the top-ranked videos for the topic.
4. For each video, get content in this priority order:
   - Official YouTube captions (with translation fallback for non-English captions)
   - Local Whisper transcription (if no captions exist)
   - Title + description (last-resort fallback)
5. Chunk all research and video content (fixed-size chunking).
6. Generate embeddings and store in ChromaDB (chunk data cached in Redis).
7. Retrieve the most relevant chunks via hybrid search (dense + BM25 + RRF), then re-rank with a cross-encoder.
8. **Gap Analysis Agent** compares retrieved research against retrieved YouTube coverage to identify what's missing.
9. Generate a structured Content Brief.
10. On request, generate a Script grounded in the brief.

---

## Evaluation

Retrieval quality was measured on a 16-query hand-labeled eval set across a ~280-chunk multi-topic corpus (machine learning, deep learning, Python, data science):

| Method | Recall@5 | MRR |
|---|---|---|
| Dense only | 87.5% | 0.6271 |
| BM25 only | 75.0% | 0.4135 |
| Hybrid (RRF) | 87.5% | 0.6146 |
| Hybrid + Re-ranked | 87.5% | **0.6615** |

Dense embeddings outperformed BM25 alone, since eval queries were often paraphrased relative to source transcripts. Cross-encoder re-ranking gave the best ranking quality (MRR) even without improving raw recall — it consistently pushed correct chunks higher when found.

An automated eval gate (`eval/gate_check.py`) re-runs this evaluation and fails (exit code 1) if recall or MRR regresses by more than 5 percentage points versus a saved baseline. This runs automatically in CI on every push via GitHub Actions.

Additional evaluation utilities:
- **Faithfulness scoring** — custom LLM-as-judge groundedness check (does the generated brief only state things present in retrieved context?). Note: this measures groundedness, not factual truth — a source-level error would still score as "faithful."
- **Latency tracking** — p50/p95/p99 per-query timing, viewable in the observability dashboard
- **Cost tracking** — token counts + USD cost per LLM call, logged per query

---

## Known Limitations

- Eval set is small (16 queries) — directionally meaningful, not statistically strong. A larger eval set (50+) would give more confidence in method comparisons.
- Faithfulness scoring checks groundedness against retrieved context, not real-world factual accuracy.
- Redis cache is keyed by exact topic string (no normalization) — "AI" and "ai" would be treated as different cache entries.
- Redis cache doesn't auto-invalidate on new writes mid-session; acceptable for this scale, would need proper invalidation in a production setting.
- Whisper fallback trades speed for coverage — CPU transcription is noticeably slower than reading existing captions, and very long videos are skipped via a duration cap to avoid stalling the pipeline.
- `main.py` (simple sequential pipeline) and `graph_pipeline.py` (LangGraph version) currently coexist; the app primarily uses the LangGraph version.

---

## Future Roadmap

**Version 1.1**
- Larger eval set
- Query rewriting
- Additional data sources (Reddit, research papers)

**Version 1.2**
- Multi-video comparison
- Thumbnail/title suggestions

**Version 2.0**
- FastAPI backend + React frontend
- Full application containerization (Docker Compose: app + Redis together)
- Authentication and cloud deployment for multi-user use

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/devharshhhh/youtube-gap-analyzer.git
cd youtube-gap-analyzer
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Install ffmpeg** (required for Whisper audio extraction)
```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

**5. Start Redis** (required — via Docker)
```bash
docker run -d -p 6379:6379 --name redis-cache redis:latest
```

**6. Create a `.env` file** in the project root with your own API keys (see below).

**7. Run the app**
```bash
streamlit run app.py
```

**8. (Optional) Run the observability dashboard separately**
```bash
streamlit run observability_dashboard.py --server.port 8502
```

---

## Environment Variables

Create a `.env` file with:
```
GROQ_API_KEY=
YOUTUBE_API_KEY=
TAVILY_API_KEY=
```

- **Groq** — free tier available at console.groq.com
- **YouTube Data API v3** — free quota via console.cloud.google.com (Google Cloud Console)
- **Tavily** — free tier available at tavily.com

---

## License

MIT License

---

## Author

**Harsh Kumar**

Building AI systems focused on Retrieval-Augmented Generation, agent orchestration, and LLM evaluation.
