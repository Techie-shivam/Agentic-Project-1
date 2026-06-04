# Agentic-Project-1


# 🧠 Agentic Deep Research System

This project is a simple implementation of an agentic AI system designed to answer research-style questions using LLMs and retrieval over a corpus of LLM-agent papers.

Instead of giving a single-shot answer like a normal LLM, the system behaves like a research assistant that plans, searches, reflects, and then generates an answer based on evidence.

---

## 💡 What this project does

The system answers complex questions by:

- breaking the question into smaller sub-questions
- retrieving relevant information from a paper corpus
- improving search if initial results are not enough
- generating a final answer using evidence
- verifying whether the answer is supported by retrieved text

The goal is to study whether agent-like components actually improve reasoning quality.

---

## 🏗️ System Flow

```
Question
  ↓
Planner → splits question into sub-questions
  ↓
Retriever → fetches relevant paper chunks
  ↓
Reflector → decides if more retrieval is needed
  ↓
Generator → produces final answer
  ↓
Verifier → checks citation correctness
```

---

## 📁 Project Structure

```
agent/
  ├── managing_agent.py   → main pipeline
  ├── planner.py
  ├── reflector.py
  ├── dense_retriever.py
  ├── generator.py
  └── citation_verifier.py

evaluation/
  ├── questions.jsonl     → evaluation dataset (30 questions)
  └── evaluation.py      → evaluation script

predictions/
  ├── full_agent.jsonl
  ├── baseline.jsonl
  ├── no_planner.jsonl
  ├── no_reflector.jsonl
  └── no_verifier.jsonl

run_predictions.py
README.md
```

---

## 🧪 Experiments

We run ablation studies with five configurations:

- Full Agent (all components enabled)
- Baseline (simple version without agent loops)
- No Planner
- No Reflector
- No Verifier

This helps us understand the contribution of each component.

---

## 📊 Evaluation Metrics

We evaluate using:

- Latency (response time per question)
- Tool Calls (number of reasoning steps)
- Answer quality (manual + logical correctness)
- Citation faithfulness (whether claims are supported)

---

## 🔍 Key Research Question

The main question explored in this project is:

> Do agentic components actually improve reasoning quality, or do they just increase complexity and latency?

---

## ⚙️ How to Run

Generate predictions:

```bash
python run_predictions.py
```

Run evaluation:

```bash
python evaluation/evaluation.py
```

---

## 📌 Output

After running the system, you get:

- prediction files for each configuration
- evaluation results comparing all ablations

---

## 🧠 Key Insights

From experiments:

- Planner improves structure for complex questions
- Reflector improves retrieval quality but increases latency
- Verifier reduces hallucinations
- Baseline is faster but less reliable

Overall, agentic systems improve reasoning but introduce higher computational cost.

---

## 🚀 Future Improvements

- hybrid retrieval (BM25 + dense)
- better reflection strategy
- improved citation grounding
- faster local inference (Ollama optimization)

---

## 👨‍💻 Note

This project was built as part of an AI research assignment on agentic systems and evaluation of LLM-based research agents.
