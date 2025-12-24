from eval_data import EVAL_QUERIES
from query import query_rag  # your RAG query function
from typing import List


def precision_at_k(retrieved_docs: List[str], relevant_doc_ids: List[str], k: int) -> float:
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_doc_ids]
    precision = len(relevant_retrieved) / k
    return precision


def recall_at_k(retrieved_docs: List[str], relevant_doc_ids: List[str], k: int) -> float:
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_doc_ids]
    recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    return recall


K = 10  # Top-K documents


for item in EVAL_QUERIES:
    query = item["query"]
    relevant_doc_ids = item["relevant_doc_ids"]
    
    response_text, retrieved_docs = query_rag(query, k=K)
    
    precision = precision_at_k(retrieved_docs, relevant_doc_ids, K)
    recall = recall_at_k(retrieved_docs, relevant_doc_ids, K)
    
    print(f"Query: {query}")
    print(f"Precision@{K}: {precision:.2f}, Recall@{K}: {recall:.2f}")

