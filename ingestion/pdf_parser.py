from pypdf import PdfReader
import pandas as pd
import os
import json

# ==========================
# Paths
# ==========================

PDF_DIR = "data/pdfs"
METADATA_FILE = "data/filtered_metadata.csv"
OUTPUT_FILE = "data/processed/documents.json"

os.makedirs("data/processed", exist_ok=True)

# ==========================
# Load Paper Metadata
# ==========================

metadata_df = pd.read_csv(METADATA_FILE)

metadata_lookup = {}

for _, row in metadata_df.iterrows():

    paper_id = str(row["id"])

    metadata_lookup[paper_id] = {
        "title": row["title"],
        "published": str(row["published"]),
        "pdf_url": row["pdf_url"]
    }

# ==========================
# Extract PDFs
# ==========================

documents = []

pdf_files = [
    f for f in os.listdir(PDF_DIR)
    if f.endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDFs")

for pdf_file in pdf_files:

    try:

        paper_id = pdf_file.replace(".pdf", "")

        pdf_path = os.path.join(
            PDF_DIR,
            pdf_file
        )

        reader = PdfReader(pdf_path)

        paper_meta = metadata_lookup.get(
            paper_id,
            {}
        )

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

              text = text.encode(
              "utf-8",
              errors="ignore"
              ).decode(
              "utf-8",
              errors="ignore"
                         )

            if not text:
                continue

            documents.append({
                "paper_id": paper_id,
                "title": paper_meta.get(
                    "title",
                    ""
                ),
                "published": paper_meta.get(
                    "published",
                    ""
                ),
                "pdf_url": paper_meta.get(
                    "pdf_url",
                    ""
                ),
                "page": page_num + 1,
                "text": text
            })

        print(
            f"Processed: {paper_id}"
        )

    except Exception as e:

        print(
            f"Failed: {pdf_file}"
        )

        print(e)

# ==========================
# Save Documents
# ==========================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    f"\nSaved {len(documents)} pages"
)

print(
    f"Output -> {OUTPUT_FILE}"
)