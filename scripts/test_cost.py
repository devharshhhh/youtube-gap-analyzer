from observability.cost_tracker import estimate_cost

sample_prompt = "Summarize the key facts about machine learning in 3 sentences." * 20  # simulate a realistic-size prompt
sample_response = "Machine learning is a field of AI that enables computers to learn from data without explicit programming. " * 10

cost = estimate_cost(sample_prompt, sample_response)
print(cost)
