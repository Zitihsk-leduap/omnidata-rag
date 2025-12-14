# This is the test function that helps to verify the functionality of the query_rag function in query.py
from query import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}

---
(Answer with 'true' or 'false') Does the actual response match the expected response exactly?
"""


def test_monopoly_rules():
    assert query_and_validated(
        question = " How much total money does a player start with in the game of Monopoly?",
        expected_response="$1500",
    )


def test_ticket_to_ride_rules():
    assert query_and_validated(
        question = " In Ticket to Ride, how many train cards does a player draw on their turn?",
        expected_response="2",
    )



def query_and_validated(question: str, expected_response: str):
    response_test = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_test,
    )

    model = Ollama(model="llama3")
    evaluation_results = model.invoke(prompt)
    evaluation_results_cleaned = evaluation_results.strip().lower()

    print(prompt)

    if "true" in evaluation_results_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
