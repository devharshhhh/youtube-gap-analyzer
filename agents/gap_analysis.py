import os
from groq import Groq
from dotenv import load_dotenv
from observability.cost_tracker import estimate_cost

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_gaps(topic: str, retrieved_chunks: list) -> dict:
    """
    retrieved_chunks: list of dicts from hybrid_search()/rerank(), each with
    'text' and 'metadata' (metadata includes source_type: 'youtube' or 'research')
    """
    research_chunks = [c for c in retrieved_chunks if c["metadata"]["source_type"] == "research"]
    youtube_chunks = [c for c in retrieved_chunks if c["metadata"]["source_type"] == "youtube"]

    research_text = "\n".join(f"- {c['text']}" for c in research_chunks)
    youtube_text = "\n".join(
        f"- [{c['metadata']['source_name']}] {c['text']}" for c in youtube_chunks
    )

    prompt = f"""
    Topic: {topic}

    RESEARCHED FACTS (most relevant, retrieved from live web search):
    {research_text}

    WHAT EXISTING YOUTUBE VIDEOS ALREADY COVER (most relevant retrieved segments):
    {youtube_text}

    Produce a content brief with these sections:
    1. Topic overview (2-3 sentences, grounded in the researched facts)
    2. What's already well-covered on YouTube (bullet list)
    3. Content gaps — facts or angles from the research that existing videos do NOT mention
    4. One suggested unique angle for a new video, with reasoning
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    brief_text = response.choices[0].message.content
    cost_info = estimate_cost(input_text=prompt, output_text=brief_text)

    return {"brief": brief_text, "cost": cost_info}
