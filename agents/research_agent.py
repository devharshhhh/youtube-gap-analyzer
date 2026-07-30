import os
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from observability.cost_tracker import estimate_cost

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def research_topic(topic: str) -> dict:
    search_results = tavily_client.search(
        query=topic,
        max_results=8,
        include_answer=False,
    )

    sources = [{"title": r["title"], "url": r["url"]} for r in search_results["results"]]

    combined_content = "\n\n".join(
        f"Source: {r['title']} ({r['url']})\n{r['content']}"
        for r in search_results["results"]
    )

    prompt = f"""
    Based only on the sources below, provide 5-8 key facts about: "{topic}"

    For each fact, note which source it came from. Prioritize recent,
    authoritative information. Flag anything that only appears in one source.

    SOURCES:
    {combined_content}
    """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    research_text = response.choices[0].message.content
    cost_info = estimate_cost(input_text=prompt, output_text=research_text)

    return {"research_text": research_text, "sources": sources, "cost": cost_info}