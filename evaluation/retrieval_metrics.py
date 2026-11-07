from typing import List

from evaluation.eval_data import EVAL_QUERIES
from rag.pipeline import query_rag


def keyword_match_validation(answer: str, expected_keywords: List[str]) -> float:
    """Percentage of expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0

    found = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
    return found / len(expected_keywords)


def main():
    print("\n" + "=" * 70)
    print("EVALUATION: Company Act 2063 RAG System")
    print("=" * 70)

    scores = []
    for idx, item in enumerate(EVAL_QUERIES, 1):
        query = item["query"]
        mode = item.get("mode", "explanation")
        expected_keywords = item.get("expected_keywords", [])

        print(f"\n{'─' * 70}")
        print(f"Test {idx}: {query}")
        print(f"Mode: {mode} | Keywords to verify: {expected_keywords}")

        result = query_rag(query)
        score = keyword_match_validation(result["answer"], expected_keywords)
        scores.append(score)

        print(result["answer"])
        print(f"Keyword match: {score:.0%}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print("\n" + "=" * 70)
    print(f"Average keyword match across {len(scores)} queries: {avg:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
