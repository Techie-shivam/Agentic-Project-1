import json
import sys
import os

# allow imports from project root
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent.managing_agent import run_research_agent


FILES = [
    "predictions/full_agent.jsonl",
    "predictions/baseline.jsonl",
    "predictions/no_planner.jsonl",
    "predictions/no_reflector.jsonl",
    "predictions/no_verifier.jsonl"
]


def load(file):
    with open(file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rewrite_with_metrics(file):

    print(f"\nFixing: {file}")

    data = load(file)
    new_data = []

    for item in data:

        # SAFE fallback (prevents crash)
        question = item.get("question") or item.get("id")

        if not question:
            print("Skipping item (no question found)")
            continue

        # run agent again ONLY for metrics
        result = run_research_agent(question)

        item["question"] = question  # ensure consistency
        item["latency"] = result.get("latency", 0)
        item["tool_calls"] = result.get("tool_calls", 0)

        new_data.append(item)

    with open(file, "w", encoding="utf-8") as f:
        for d in new_data:
            f.write(json.dumps(d) + "\n")

    print(f"Done: {file}")


for f in FILES:
    rewrite_with_metrics(f)