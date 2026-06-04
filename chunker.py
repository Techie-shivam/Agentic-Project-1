import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = "data/processed/documents.json"
OUTPUT_FILE = "data/chunks/chunks.json"

os.makedirs("data/chunks", exist_ok=True)

# -------------------------
# Load Documents
# -------------------------

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    documents = json.load(f)

print(f"Loaded {len(documents)} pages")

# -------------------------
# Chunk Splitter
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " "
    ]
)

chunks = []

# -------------------------
# Create Chunks
# -------------------------

for doc in documents:

    split_texts = splitter.split_text(
        doc["text"]
    )

    for idx, chunk_text in enumerate(split_texts):

        chunks.append({

            "paper_id": doc["paper_id"],

            "title": doc["title"],

            "published": doc["published"],

            "page": doc["page"],

            "chunk_id": idx,

            "text": chunk_text

        })

print(
    f"Created {len(chunks)} chunks"
)

# -------------------------
# Save
# -------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    f"Saved -> {OUTPUT_FILE}"
)