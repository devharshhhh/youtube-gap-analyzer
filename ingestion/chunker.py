def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Baseline chunking: splits text into word-based windows with overlap.
    (Word count is a rough stand-in for tokens — good enough to start.)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap
    return chunks


def semantic_chunk(text: str, similarity_threshold: float = 0.5) -> list:
    """
    Splits text where consecutive sentences drop in similarity —
    a real topic shift, not just a fixed word count.
    """
    from sentence_transformers import SentenceTransformer, util
    import re

    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    if len(sentences) < 2:
        return [text]

    embeddings = model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
        if sim < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks