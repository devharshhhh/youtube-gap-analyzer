import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # Approximation for Llama tokenization

# Groq pricing for openai/gpt-oss-120b (as of mid-2026)
INPUT_COST_PER_M = 0.59
OUTPUT_COST_PER_M = 0.79


def count_tokens(text: str) -> int:
    return len(encoder.encode(text))


def estimate_cost(input_text: str, output_text: str) -> dict:
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text)

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_M

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }
