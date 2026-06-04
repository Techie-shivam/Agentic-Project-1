import os
import faiss
import pickle

from sentence_transformers import SentenceTransformer
from google import genai

# ==========================
# Gemini
# ==========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ==========================
# Load FAISS
# ==========================

index = faiss.read_index(
    "vector_db/faiss_index.bin"
)

with open(
    "vector_db/chunk_metadata.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)

print(
    f"Loaded {index.ntotal} vectors"
)

# ==========================
# Embedding Model
# ==========================

embed_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ==========================
# Retriever
# ==========================

def retrieve(
    query,
    top_k=5
):

    query_embedding = embed_model.encode(
        [query]
    )

    distances, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(
            chunks[idx]
        )

    return results

# ==========================
# Context Builder
# ==========================

def build_context(
    retrieved_chunks
):

    context = ""

    for chunk in retrieved_chunks:

        context += (
            f"\n\n"
            f"[Paper: {chunk['paper_id']}] "
            f"[Page: {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    return context

# ==========================
# Answer Generation
# ==========================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are a research assistant.

Answer ONLY using the provided context.

If information is missing,
say that the corpus does not contain enough evidence.

Use paper ids when citing.

Context:
{context}

Question:
{question}

Provide:
1. Direct answer
2. Supporting evidence
3. Paper citations
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# ==========================
# Main Loop
# ==========================

while True:

    question = input(
        "\nQuestion: "
    )

    if question.lower() == "exit":
        break

    retrieved_chunks = retrieve(
        question,
        top_k=5
    )

    context = build_context(
        retrieved_chunks
    )

    answer = generate_answer(
        question,
        context
    )

    print("\n")
    print("=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(answer)