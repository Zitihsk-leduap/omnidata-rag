from typing import List
from rag.pipeline import query_rag
from evaluation.eval_data import EVAL_QUERIES

def faithfulness_score(llm_response: str, retrieved_docs: List[str]) -> float:
    """
    Compute a simple faithfulness score based on token overlap
    between the LLM response and retrieved document texts.
    """
    combined_text = " ".join(retrieved_docs).lower()
    answer_tokens = set(llm_response.lower().split())

    if not answer_tokens:
        return 0.0

    faithful_tokens = [token for token in answer_tokens if token in combined_text]
    score = len(faithful_tokens) / len(answer_tokens)
    return score



def main():
    for item in EVAL_QUERIES:
        query = item["query"]
        print(f"\nQuery: {query}")

        result = query_rag(query)
        llm_response = result["answer"]
        retrieved_docs = [s["text"] for s in result["sources"]]

        score = faithfulness_score(llm_response, retrieved_docs)
        print("LLM Response:", llm_response)
        print(f"Faithfulness Score: {score:.2f}")

if __name__ == "__main__":
    main()
