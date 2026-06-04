import arxiv
import pandas as pd

client = arxiv.Client()

query = """
all:"llm agent"
OR all:"agentic"
OR all:"tool use"
OR all:"agent memory"
OR all:"react"
OR all:"reflexion"
OR all:"self-rag"
OR all:"multi-agent"
OR all:"computer use agent"
OR all:"agent benchmark"
"""

search = arxiv.Search(
    query=query,
    max_results=2000
)

papers = []

print("Collecting metadata...")

for paper in client.results(search):

    papers.append({
        "id": paper.entry_id.split("/")[-1],
        "title": paper.title,
        "abstract": paper.summary,
        "published": paper.published,
        "pdf_url": paper.pdf_url
    })

df = pd.DataFrame(papers)

df.to_csv("metadata.csv", index=False)

print(f"Collected {len(df)} papers")





import pandas as pd

# ==========================
# Load Metadata
# ==========================

df = pd.read_csv("metadata.csv")

df["published"] = pd.to_datetime(df["published"])

# ==========================
# Date Filter
# ==========================

df = df[
    (df["published"] >= "2024-01-01")
    &
    (df["published"] <= "2026-04-30")
]

print(f"After date filter: {len(df)}")

# ==========================
# Required Terms
# ==========================

LLM_TERMS = [
    "llm",
    "large language model",
    "language model",
    "gpt",
    "chatgpt",
    "transformer"
]

AGENT_TERMS = [
    "agent",
    "agentic",
    "tool use",
    "tool calling",
    "planning",
    "memory",
    "reflection",
    "react",
    "reflexion",
    "self-rag",
    "multi-agent",
    "computer use",
    "benchmark",
    "retrieval",
    "rag"
]

# ==========================
# Excluded Domains
# ==========================

NEGATIVE_TERMS = [
    "drug",
    "molecular",
    "protein",
    "battery",
    "marine",
    "forecasting",
    "pokemon",
    "trading card",
    "finance",
    "financial",
    "chemistry",
    "biology",
    "medical",
    "healthcare",
    "pedagogical",
    "agriculture",
    "satellite",
    "wireless sensor",
    "power grid",
    "battery parameter",
    "urban perception",
    "interatomic",
    "medication recommendation"
]

# ==========================
# Filtering Function
# ==========================

def is_relevant(title, abstract):

    text = (str(title) + " " + str(abstract)).lower()

    has_llm = any(
        term in text
        for term in LLM_TERMS
    )

    agent_matches = sum(
        term in text
        for term in AGENT_TERMS
    )

    has_negative = any(
        term in text
        for term in NEGATIVE_TERMS
    )

    return (
        has_llm
        and agent_matches >= 2
        and not has_negative
    )

# ==========================
# Apply Filter
# ==========================

filtered_df = df[
    df.apply(
        lambda row: is_relevant(
            row["title"],
            row["abstract"]
        ),
        axis=1
    )
]

# ==========================
# Remove Duplicates
# ==========================

filtered_df = filtered_df.drop_duplicates(
    subset=["id"]
)

# ==========================
# Save
# ==========================

filtered_df.to_csv(
    "filtered_metadata.csv",
    index=False
)

print(
    f"\nRemaining papers: {len(filtered_df)}"
)

print("\nTop 50 papers:\n")

for title in filtered_df["title"].head(50):
    print("-", title)
    


import pandas as pd
import requests
import os
import time

os.makedirs("data/pdfs", exist_ok=True)

df = pd.read_csv(
    "filtered_metadata.csv"
)

print(f"Downloading {len(df)} PDFs...")

for _, row in df.iterrows():

    paper_id = row["id"]

    pdf_url = row["pdf_url"]

    if not str(pdf_url).endswith(".pdf"):
        pdf_url += ".pdf"

    filepath = f"data/pdfs/{paper_id}.pdf"

    if os.path.exists(filepath):
        continue

    try:

        response = requests.get(
            pdf_url,
            timeout=60
        )

        if response.status_code == 200:

            with open(filepath, "wb") as f:
                f.write(response.content)

            print("Downloaded:", paper_id)

        else:

            print(
                "Failed:",
                paper_id,
                response.status_code
            )

        time.sleep(1)

    except Exception as e:

        print(
            "Error:",
            paper_id,
            e
        )