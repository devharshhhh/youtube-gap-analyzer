import os
from dotenv import load_dotenv
from groq import Groq
from observability.cost_tracker import estimate_cost

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_script(brief: str, length_minutes: int = 5, tone: str = "conversational") -> dict:
    prompt = f"""
    Using ONLY the facts and angle in this content brief, write a
    {length_minutes}-minute video script in a {tone} tone.

    BRIEF:
    {brief}

    Requirements:
    - Hook in the first 10 seconds
    - Structure the middle section around the content gap identified in the brief
    - End with a clear call to action
    - Do not introduce facts that are not present in the brief
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    script_text = response.choices[0].message.content
    cost_info = estimate_cost(input_text=prompt, output_text=script_text)

    return {"script": script_text, "cost": cost_info}
