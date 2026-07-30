from main import run_pipeline
from eval.faithfulness import faithfulness_score

topic = "adobe photoshop vs gimp which one is better"
result = run_pipeline(topic, want_script=False)

print("\n--- BRIEF ---")
print(result["brief"])

print("\n--- CHECKING FAITHFULNESS ---")
faith = faithfulness_score(result["brief"], result["retrieved_chunks"])

print(f"\nFaithfulness score: {faith['score']:.1%} ({faith['supported_count']}/{faith['total_claims']} claims supported)")
print("\nPer-claim breakdown:")
for r in faith["claims"]:
    status = "✓ SUPPORTED" if r["supported"] else "✗ NOT SUPPORTED"
    print(f"  [{status}] {r['claim'][:100]}")