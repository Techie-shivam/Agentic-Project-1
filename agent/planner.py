import requests

# ==========================
# Ollama Config
# ==========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:latest"      # or mistral / phi3
 # change to mistral, phi3, etc. if needed

# ==========================
# Planner Function
# ==========================

def create_plan(question):
    """
    Breaks a research question into 3-6 sub-questions.
    """

    prompt = f"""
You are an expert research planning agent.

Break the question into 3-6 focused sub-questions.

Rules:
- Each sub-question must be independently searchable
- No overlap
- Cover all aspects
- Return ONLY numbered list
- No explanations

Question:
{question}
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

        result = response.json()

        return result.get("response", "").strip()

    except Exception as e:

        print(f"Planner Error: {e}")

        return question


# ==========================
# Test
# ==========================

if __name__ == "__main__":

    q = input("Question: ")

    print("\nPLAN\n")

    print(create_plan(q))