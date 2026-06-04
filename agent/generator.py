import requests

# ==========================
# Ollama Config
# ==========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:latest"   # change to mistral / phi3 if needed

# ==========================
# Generator Agent (SAME INTERFACE)
# ==========================

def generate_answer(question, evidence):
    """
    Generates final answer using Ollama.
    (Drop-in replacement for OpenAI/Groq version)
    """

    prompt = f"""
You are a deep research assistant.

Answer the question ONLY using the given evidence.

Rules:
- Do NOT use outside knowledge
- Do NOT hallucinate
- Always stay grounded in evidence
- Add citations using paper_id or titles
- Use bullet points if needed

Question:
{question}

Evidence:
{evidence}

Return a structured final answer.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    return result.get("response", "")


# ==========================
# Optional Test
# ==========================

if __name__ == "__main__":

    q = input("Question: ")
    e = input("\nEvidence:\n")

    print("\n======================")
    print("FINAL ANSWER (OLLAMA)")
    print("======================\n")

    print(generate_answer(q, e))