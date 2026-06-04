import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================
# Load FAISS Index
# ==========================

INDEX_PATH = "vector_db/faiss_index.bin"
META_PATH = "vector_db/chunk_metadata.pkl"

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    chunks = pickle.load(f)

# ==========================
# Embedding Model
# ==========================

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ==========================
# Retriever Function
# ==========================

def retrieve(query, top_k=5):
    """
    Dense retrieval using FAISS.
    Returns top-k most relevant chunks.
    """

    # 1. Encode query
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    # 2. Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:

        # safety check
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        results.append({
            "paper_id": chunk.get("paper_id"),
            "title": chunk.get("title"),
            "page": chunk.get("page"),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text")
        })

    return results

# ==========================
# Test
# ==========================

if __name__ == "__main__":

    while True:

        query = input("\nQuestion: ")

        if query.lower() == "exit":
            break

        results = retrieve(query, top_k=5)

        for i, r in enumerate(results, 1):

            print("\n" + "=" * 80)
            print(f"Rank: {i}")
            print(f"Paper ID: {r['paper_id']}")
            print(f"Title: {r['title']}")
            print(f"Page: {r['page']}")
            print(f"Chunk ID: {r['chunk_id']}")
            print("\nText:\n")
            print(r["text"][:1000])