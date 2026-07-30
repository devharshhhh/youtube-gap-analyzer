from ingestion.embed_and_store import get_or_create_collection

topic = "eval_corpus"
collection = get_or_create_collection(topic)
data = collection.get(include=["documents", "metadatas"])

for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
    print(f"ID: {cid}")
    print(f"Source: {meta['source_name'][:50]}")
    print(f"Text: {doc[:150]}...")
    print("-" * 80)