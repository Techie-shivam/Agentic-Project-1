import json
import os

from agent.managing_agent import run_research_agent

QUESTIONS_FILE = "evaluation/questions.jsonl"


def load_questions():

    questions = []

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            questions.append(
                json.loads(line)
            )

    return questions


def run_configuration(
    config_name,
    use_planner,
    use_reflector,
    use_verifier
):

    questions = load_questions()

    # Uncomment for testing only 1 question
    # questions = questions[:1]

    output_file = f"predictions/{config_name}.jsonl"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as out:

        for item in questions:

            qid = item["id"]
            question = item["question"]

            print(
                f"\nRunning {config_name} | {qid}"
            )

            result = run_research_agent(
                question,
                use_planner=use_planner,
                use_reflector=use_reflector,
                use_verifier=use_verifier
            )

            prediction = {
                "id": qid,
                "answer": result["answer"]
            }

            out.write(
                json.dumps(prediction)
                + "\n"
            )

    print(
        f"\nSaved {output_file}"
    )


if __name__ == "__main__":

    os.makedirs(
        "predictions",
        exist_ok=True
    )

    # Full Agent
    run_configuration(
        "full_agent",
        True,
        True,
        True
    )

    # Baseline
    run_configuration(
        "baseline",
        False,
        False,
        False
    )

    # No Planner
    run_configuration(
        "no_planner",
        False,
        True,
        True
    )

    # No Reflector
    run_configuration(
        "no_reflector",
        True,
        False,
        True
    )

    # No Verifier
    run_configuration(
        "no_verifier",
        True,
        True,
        False
    )