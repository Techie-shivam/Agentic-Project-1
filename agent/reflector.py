import requests
import json

# ==========================
# Ollama Config
# ==========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:latest"      # or mistral / phi3

# ==========================
# Reflector Function
# ==========================

def reflect(original_question, evidence):
    """
    Decides whether retrieved evidence is sufficient.
    If not, returns a new search query.
    """

    # prevent huge prompts
    evidence = evidence[:12000]

    prompt = f"""
You are a research reflection agent.

Decide whether the retrieved evidence is sufficient
to answer the question.

Question:
{original_question}

Evidence:
{evidence}

Return ONLY valid JSON.

If sufficient:
{{
  "enough": true,
  "new_query": ""
}}

If not sufficient:
{{
  "enough": false,
  "new_query": "specific search query"
}}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        output = response.json().get("response", "").strip()

        start = output.find("{")
        end = output.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON found")

        json_text = output[start:end]

        result = json.loads(json_text)

        return {
            "enough": bool(result.get("enough", True)),
            "new_query": result.get("new_query", "")
        }

    except Exception as e:

        print(f"Reflector Error: {e}")

        return {
            "enough": True,
            "new_query": ""
        }


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    q = "What is overfitting?"

    evidence = """
Overfitting occurs when a machine learning model learns training data too closely.
It performs well on training data but poorly on unseen test data.
    """

    print(reflect(q, evidence))