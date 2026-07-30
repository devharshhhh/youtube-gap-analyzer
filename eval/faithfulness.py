import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_claims(brief_text: str) -> list:
    """Ask the LLM to break the brief into individual, checkable factual claims."""
    prompt = f"""
    Break the following text into a list of individual factual claims.
    Each claim should be a single, standalone statement.
    Ignore section headers, opinions, and suggestions — only extract
    statements presented as facts.

    Return ONLY a numbered list, one claim per line, nothing else.

    TEXT:
    {brief_text}
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    lines = response.choices[0].message.content.strip().split("\n")
    claims = [line.split(".", 1)[-1].strip() for line in lines if line.strip()]
    return claims


def check_claim_support(claim: str, context: str) -> bool:
    """Ask the LLM: is this specific claim supported by the given context?"""
    prompt = f"""
    CONTEXT:
    {context}

    CLAIM:
    {claim}

    Is this claim directly supported by the context above? Answer with
    exactly one word: YES or NO. Do not explain.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")


def faithfulness_score(brief_text: str, retrieved_chunks: list) -> dict:
    """
    Returns the fraction of extracted claims that are actually supported
    by the retrieved context. This is your hallucination detector.
    """
    context = "\n".join(c["text"] for c in retrieved_chunks)
    claims = extract_claims(brief_text)

    if not claims:
        return {"score": None, "claims": [], "supported_count": 0, "total_claims": 0}

    results = []
    for claim in claims:
        supported = check_claim_support(claim, context)
        results.append({"claim": claim, "supported": supported})

    supported_count = sum(1 for r in results if r["supported"])
    score = supported_count / len(claims)

    return {
        "score": round(score, 3),
        "claims": results,
        "supported_count": supported_count,
        "total_claims": len(claims),
    }