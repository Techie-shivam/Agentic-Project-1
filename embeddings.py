import json
import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# ==========================
# Config
# ==========================

CHUNKS_FILE = "data/chunks/chunks.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

VECTOR_DB_DIR = "vector_db"

os.makedirs(
    VECTOR_DB_DIR,
    exist_ok=True
)

# ==========================
# Load Chunks
# ==========================

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(
    f"Loaded {len(chunks)} chunks"
)

# ==========================
# Embedding Model
# ==========================

model = SentenceTransformer(
    EMBED_MODEL
)

# ==========================
# Create Embeddings
# ==========================

texts = [
    chunk["text"]
    for chunk in chunks
]

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype(
    "float32"
)

print(
    f"Embeddings shape: {embeddings.shape}"
)

# ==========================
# Build FAISS Index
# ==========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

print(
    f"Indexed {index.ntotal} chunks"
)

# ==========================
# Save FAISS Index
# ==========================

faiss.write_index(
    index,
    os.path.join(
        VECTOR_DB_DIR,
        "faiss_index.bin"
    )
)

# ==========================
# Save Metadata
# ==========================

with open(
    os.path.join(
        VECTOR_DB_DIR,
        "chunk_metadata.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )

print(
    "\nFAISS index saved successfully."
)