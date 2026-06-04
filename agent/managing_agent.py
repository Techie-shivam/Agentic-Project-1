from agent.planner import create_plan
from agent.reflector import reflect
from agent.citation_verifier import verify_citations
from agent.dense_retriever import retrieve
from agent.generator import generate_answer
import time

# ==========================
# MAIN AGENT PIPELINE
# ==========================

def run_research_agent(
    question,
    max_iters=3,
    use_planner=True,
    use_reflector=True,
    use_verifier=True
):

    start_time = time.time()
    tool_calls = 0   # ✅ TOOL COUNTER

    print("\n==============================")
    print("🔍 INPUT QUESTION")
    print("==============================")
    print(question)

    # ==========================
    # 1. PLANNER
    # ==========================

    if use_planner:

        print("\n[1] Planning...")
        tool_calls += 1  # planner call

        plan = create_plan(question)
        print(plan)

        if isinstance(plan, str):
            queries = plan.split("\n")
        else:
            queries = [str(plan)]

    else:

        print("\n[1] Planner Disabled")

        plan = "DISABLED"
        queries = [question]

    # ==========================
    # 2. RETRIEVAL
    # ==========================

    print("\n[2] Initial Retrieval...")

    evidence_chunks = []

    for q in queries[:3]:

        tool_calls += 1  # retrieval call

        retrieved = retrieve(q)
        evidence_chunks.extend(retrieved)

    evidence = "\n\n".join(chunk["text"] for chunk in evidence_chunks)

    # ==========================
    # 3. REFLECTOR
    # ==========================

    if use_reflector:

        print("\n[3] Reflection Loop Starting...")

        for i in range(max_iters):

            tool_calls += 1  # reflector call

            reflection = reflect(question, evidence)

            print(f"\nIteration {i+1}:")
            print(reflection)

            if reflection["enough"]:
                print("\n✔ Enough evidence found.")
                break

            print("\n➕ Retrieving more evidence using:")
            print(reflection["new_query"])

            tool_calls += 1  # extra retrieval

            new_evidence_chunks = retrieve(reflection["new_query"])

            evidence += "\n\n" + "\n\n".join(
                chunk["text"] for chunk in new_evidence_chunks
            )

    else:
        print("\n[3] Reflector Disabled")

    # ==========================
    # 4. GENERATOR
    # ==========================

    print("\n[4] Generating Final Answer...")

    tool_calls += 1  # generator call

    answer = generate_answer(question, evidence)

    # ==========================
    # 5. VERIFIER
    # ==========================

    if use_verifier:

        print("\n[5] Verifying Claims...")

        tool_calls += 1  # verifier call

        claims = answer.split(". ")

        claims_and_evidence = [
            {
                "claim": c,
                "evidence": evidence
            }
            for c in claims[:5]
        ]

        verification_results = verify_citations(claims_and_evidence)

        print("\n==============================")
        print("📊 VERIFICATION RESULTS")
        print("==============================")

        for r in verification_results:
            print(r)

    else:

        print("\n[5] Verifier Disabled")
        verification_results = []

    # ==========================
    # RETURN
    # ==========================

    latency = time.time() - start_time

    return {
        "question": question,
        "plan": plan,
        "evidence": evidence,
        "answer": answer,
        "verification": verification_results,
        "latency": latency,
        "tool_calls": tool_calls   # ✅ IMPORTANT
    }


# ==========================
# RUN SYSTEM
# ==========================

if __name__ == "__main__":

    q = input("\nEnter research question: ")

    output = run_research_agent(
        q,
        use_planner=True,
        use_reflector=True,
        use_verifier=True
    )

    print("\n==============================")
    print("📌 FINAL ANSWER")
    print("==============================\n")

    print(output["answer"])