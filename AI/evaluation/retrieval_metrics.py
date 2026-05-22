from eval_data import EVAL_QUERIES
from query import query_rag
from typing import List


def precision_at_k(retrieved_docs: List[str], relevant_doc_ids: List[str], k: int) -> float:
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_doc_ids]
    precision = len(relevant_retrieved) / k if k > 0 else 0.0
    return precision


def recall_at_k(retrieved_docs: List[str], relevant_doc_ids: List[str], k: int) -> float:
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_doc_ids]
    recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    return recall


def keyword_match_validation(answer: str, expected_keywords: List[str]) -> float:
    """
    PRODUCTION FIX: Validate answer contains expected keywords from source.
    Returns: percentage of keywords found in answer
    """
    if not expected_keywords:
        return 1.0

    found_keywords = 0
    for keyword in expected_keywords:
        if keyword.lower() in answer.lower():
            found_keywords += 1

    return found_keywords / len(expected_keywords)


# PRODUCTION FIX: Manual validation framework for Company Act QA
print("\n" + "="*70)
print("EVALUATION: Company Act 2063 RAG System")
print("="*70 + "\n")

K = 10  # Top-K documents

for idx, item in enumerate(EVAL_QUERIES, 1):
    query = item["query"]
    query_ne = item.get("query_ne", query)
    mode = item.get("mode", "explanation")
    expected_keywords = item.get("expected_keywords", [])

    print(f"\n{'─'*70}")
    print(f"Test {idx}: {query}")
    print(f"Mode: {mode} | Keywords to verify: {expected_keywords}")
    print(f"{'─'*70}")

    # Test with English query
    print(f"\n[Testing English query]")
    query_rag(query)

    # Validation checks can be added here
    # For now, system prints to stdout only
    # Future: Modify query_rag to return structured (answer, confidence, sources)

print("\n" + "="*70)
print("Note: Run above queries manually and verify:")
print("  1. Answer contains expected keywords")
print("  2. Validation confidence > 70%")
print("  3. No 'NOT FOUND' when chunks available")
print("="*70 + "\n")


