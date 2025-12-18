# evaluation/eval_data.py

EVAL_QUERIES = [
    {
        "query": "What is machine learning?",
        "relevant_doc_ids": [
            "pdf_thebook.pdf_page_10_chunk_0",
            "pdf_thebook.pdf_page_12_chunk_15",
            "pdf_thebook.pdf_page_12_chunk_5"
        ]
    },
    {
        "query": "Explain the concept of overfitting in machine learning.",
        "relevant_doc_ids": [
            "pdf_thebook.pdf_page_12_chunk_16",
            "pdf_thebook.pdf_page_222_chunk_1",
            "pdf_thebook.pdf_page_12_chunk_15",
            "pdf_thebook.pdf_page_41_chunk_0"
        ]
    },
    {
        "query": "How does a decision tree algorithm work?",
        "relevant_doc_ids": [
            "pdf_thebook.pdf_page_165_chunk_2",
            "web_page_0_chunk_102",
            "web_page_0_chunk_50"
        ]
    },
    {
        "query": "What is monopoly and how is it played?",
        "relevant_doc_ids": [
            "Data/monopoly.pdf_page_3_chunk_2",
            "pdf_monopoly.pdf_page_3_chunk_2",
            "pdf_monopoly.pdf_page_2_chunk_2"
        ]
    }
]
