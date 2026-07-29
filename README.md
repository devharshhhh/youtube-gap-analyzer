# TubeScope - AI-Powered YouTube Gap Analyzer

> Analyze YouTube content like a researcher, not a viewer.

TubeScope is an AI-powered research assistant that helps YouTube creators discover content opportunities by analyzing existing videos, extracting transcripts, identifying knowledge gaps, and generating evidence-backed video ideas and scripts.

Instead of relying on intuition, TubeScope uses Retrieval-Augmented Generation (RAG), hybrid search, and LLM reasoning to answer one question:

> **"What valuable information is missing from the top YouTube videos on this topic?"**

---

## Demo

*(GIF coming soon)*

---

# Why?

Most creators spend hours:

- Watching competitor videos
- Taking notes
- Comparing information
- Reading articles
- Looking for unique angles

TubeScope automates this workflow.

It searches YouTube, extracts or transcribes video content, performs semantic retrieval, analyzes what has already been covered, identifies missing information, and generates research-backed content suggestions.

---

# Features

- Search YouTube videos by topic
- Fetch official YouTube transcripts
- Automatically transcribe videos using Whisper when transcripts are unavailable
- Hybrid Retrieval (Semantic Search + Keyword Search)
- Evidence-backed Gap Analysis
- AI-generated Video Briefs
- AI-generated Script Drafts
- Citation-aware Responses
- Cost Tracking
- Latency Benchmarking
- Observability Dashboard

---

# Architecture

(Add architecture diagram here)

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
      ▼              ▼               ▼
 YouTube Agent   Research Agent   Gap Analysis
      │              │               │
      ▼              ▼               ▼
Transcript      Web Research     Evidence Merge
      │
      ▼
Whisper (Fallback)
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
Vector Database (ChromaDB)
      │
      ▼
Hybrid Retrieval
      │
      ▼
Script Generation
```

---

# Tech Stack

## Backend

- Python
- Streamlit
- LangGraph
- LangChain

## AI

- OpenAI
- Gemini
- Groq
- Whisper

## Retrieval

- ChromaDB
- Vector Embeddings
- Semantic Search
- Hybrid Retrieval

## External APIs

- YouTube Data API
- Tavily Search API

---

# Project Structure

```
youtube-gap-analyzer/

├── agents/
├── ingestion/
├── retrieval/
├── eval/
├── observability/
├── tests/
├── docs/
├── app.py
├── graph_pipeline.py
├── requirements.txt
└── README.md
```

---

# Pipeline

1. User enters a YouTube topic.

2. Search the top-ranked YouTube videos.

3. Retrieve transcripts.

4. If transcript is unavailable:
   - Download audio
   - Transcribe using Whisper

5. Chunk transcripts.

6. Generate embeddings.

7. Store embeddings inside ChromaDB.

8. Retrieve relevant information using Hybrid Search.

9. Perform AI-based Gap Analysis.

10. Generate:

- Content Brief
- Missing Topics
- Video Structure
- Script Draft

---

# Evaluation

The system includes evaluation utilities for measuring:

- Faithfulness
- Source Coverage
- Retrieval Quality
- Latency
- Cost Tracking

---

# Future Roadmap

## Version 1.1

- Better reranking
- Query rewriting
- Improved prompt engineering

## Version 1.2

- Multi-video comparison
- Thumbnail suggestions
- Title generation

## Version 2.0

- FastAPI backend
- React frontend
- Authentication
- Cloud deployment

---

# Installation

Clone the repository

```bash
git clone https://github.com/devharshhh/youtube-gap-analyzer.git
```

Go inside the project

```bash
cd youtube-gap-analyzer
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create an environment file

```bash
cp .env.example .env
```

Add your API keys.

Run

```bash
streamlit run app.py
```

---

# Environment Variables

```
OPENAI_API_KEY=
GROQ_API_KEY=
GEMINI_API_KEY=
YOUTUBE_API_KEY=
TAVILY_API_KEY=
```

---

# License

MIT License

---

# Author

Harsh Kumar

Building production-grade AI systems focused on Retrieval-Augmented Generation, Agentic AI, and LLM Evaluation.
