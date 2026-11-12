import matplotlib.pyplot as plt
import numpy as np

from evaluation.eval_data import EVAL_QUERIES
from evaluation.faithfulness import faithfulness_score
from evaluation.retrieval_metrics import keyword_match_validation
from rag.pipeline import query_rag


def main():
    faithfulness_scores = []
    keyword_scores = []

    for item in EVAL_QUERIES:
        query = item["query"]
        expected_keywords = item.get("expected_keywords", [])

        result = query_rag(query)
        retrieved_docs = [s["text"] for s in result["sources"]]

        faithfulness_scores.append(faithfulness_score(result["answer"], retrieved_docs))
        keyword_scores.append(keyword_match_validation(result["answer"], expected_keywords))

    avg_faithfulness = np.mean(faithfulness_scores)
    avg_keyword_match = np.mean(keyword_scores)

    plt.figure()
    plt.bar(["faithfulness", "keyword match"], [avg_faithfulness, avg_keyword_match])
    plt.ylim(0, 1)
    plt.title("RAG Evaluation Metrics — Company Act 2063")
    plt.ylabel("Score")
    plt.show()


if __name__ == "__main__":
    main()
