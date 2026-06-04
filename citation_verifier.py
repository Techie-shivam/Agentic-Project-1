import requests
import json

# ==========================
# Ollama Config
# ==========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"   # or mistral / phi3

# ==========================
# SINGLE CLAIM CHECKER
# ==========================

def verify_claim(claim, evidence):
    """
    Checks if evidence supports claim using Ollama.
    """

    prompt = f"""
You are a citation verification system.

Your job is to check whether the evidence supports the claim.

Return ONLY valid JSON.

Claim:
{claim}

Evidence:
{evidence}

Output format:
{{
  "supported": true,
  "reason": "short explanation"
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    output = response.json().get("response", "")

    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        return json.loads(output[start:end])
    except:
        return {
            "supported": False,
            "reason": "parse error"
        }


# ==========================
# BATCH VERIFIER (USED BY MANAGER)
# ==========================

def verify_citations(claims_and_evidence):
    """
    Input:
    [
        {"claim": "...", "evidence": "..."}
    ]
    """

    results = []

    for item in claims_and_evidence:

        result = verify_claim(
            item["claim"],
            item["evidence"]
        )

        results.append({
            "claim": item["claim"],
            "supported": result.get("supported", False),
            "reason": result.get("reason", "")
        })

    return results