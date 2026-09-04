import chromadb
from sentence_transformers import SentenceTransformer

# Local, persistent vector database — saves to a folder on disk
import tempfile
import os

CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_DIR)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_or_create_collection(topic: str):
    collection_name = "topic_" + "".join(c if c.isalnum() else "_" for c in topic.lower())
    collection_name = collection_name.strip("_")  # remove leading/trailing underscores
    collection_name = collection_name[:512]  # enforce max length just in case
    return chroma_client.get_or_create_collection(name=collection_name)

def store_chunks(topic: str, chunks: list, source_type: str, source_name: str):
    if not chunks:
        return

    collection = get_or_create_collection(topic)
    embeddings = embedding_model.encode(chunks).tolist()

    # Sanitize the source name once, truncate BEFORE adding the index
    safe_source = "".join(c if c.isalnum() else "_" for c in source_name)[:60]

    ids = [f"{source_type}_{i}_{safe_source}" for i in range(len(chunks))]

    metadatas = [
        {"source_type": source_type, "source_name": source_name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

def query_collection(topic: str, query: str, n_results: int = 5):
    """
    Quick dense-search test — full hybrid search comes in Phase 3.
    """
    collection = get_or_create_collection(topic)
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    return results
