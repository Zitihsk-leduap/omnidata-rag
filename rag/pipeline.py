import argparse
import os
import warnings

# Suppress deprecation warning from google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from FlagEmbedding import FlagReranker

from rag.embeddings import get_embeddings
from rag.retrieval import HybridRetriever
from rag.language import detect_language, normalize_bs_date
from rag.prompts import CLASSIFIER_PROMPT, FACT_PROMPT, EXPLANATION_PROMPT

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=genai.types.GenerationConfig(
        temperature=0.0,
        max_output_tokens=5000,
    ),
)


# PATHS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

print("CHROMA PATH:", CHROMA_PATH)


# RERANKER
print("Loading multilingual reranker...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
print("Reranker loaded.")


# DATABASE
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embeddings(),
)

print("TOTAL CHUNKS:", len(db.get()["ids"]))


# HYBRID RETRIEVER
print("Initializing hybrid retriever...")
hybrid_retriever = HybridRetriever(db)


def detect_mode(query: str) -> str:
    try:
        prompt = CLASSIFIER_PROMPT.format(question=query)
        response = gemini_model.generate_content(prompt)
        result = response.text.strip().upper()

        if "EXPLANATION" in result:
            print(f"Classifier result: {result} -> explanation")
            return "explanation"
        if "FACT" in result:
            print(f"Classifier result: {result} -> fact")
            return "fact"

        print(
            f"Warning: unexpected classifier result '{result}', defaulting to explanation"
        )
        return "explanation"
    except Exception as error:
        print(f"Warning: mode classifier failed ({error}), defaulting to explanation")
        return "explanation"


def _select_chunks(reranked, mode: str):
    """Pick the chunks to use as context based on reranker score gaps."""
    if mode == "fact":
        top_score = reranked[0][0]
        second_score = reranked[1][0] if len(reranked) > 1 else -999
        gap = top_score - second_score

        if gap > 2.0:
            print(f"High confidence — using TOP 1 chunk (gap: {gap:.2f})")
            return [reranked[0][1]]

        print(f"Moderate confidence — using TOP 2 chunks (gap: {gap:.2f})")
        return [d for _, d in reranked[:2]]

    top_score = reranked[0][0]
    relevance_threshold = top_score - 5.0
    top_docs = [d for score, d in reranked if score > relevance_threshold][:5]
    print(f"Explanation mode — using TOP {len(top_docs)} highly relevant chunks (within 5 pts of best)")
    return top_docs


# QUERY FUNCTION
def query_rag(query_text: str, k: int = 10) -> dict:
    """Run the RAG pipeline and return a structured result.

    Returns:
        {"answer": str, "mode": str, "language": str, "sources": [...]}
    """
    query_text = query_text.strip()
    mode = detect_mode(query_text)
    language = detect_language(query_text)
    print(f"Language detected: {language.upper()}")
    print(f"Mode detected: {mode.upper()}")

    # HYBRID RETRIEVAL
    print("Searching with hybrid retrieval (vector + BM25)...")
    hybrid_results = hybrid_retriever.retrieve(query_text, k=k)

    if not hybrid_results:
        print("No documents found.")
        return {"answer": "Not found in provided context", "mode": mode, "language": language, "sources": []}

    docs = [doc for _, doc in hybrid_results]
    print(f"Retrieved {len(docs)} candidate chunks.")

    # RERANKING (NO TRUNCATION)
    print("Reranking results...")
    pairs = [(query_text, d.page_content) for d in docs]
    scores = reranker.compute_score(pairs)
    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    print("Selecting chunks by relative relevance...")
    top_docs = _select_chunks(reranked, mode)

    # CONTEXT BUILD
    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{d.page_content}" for i, d in enumerate(top_docs)
    )

    prompt_template = FACT_PROMPT if mode == "fact" else EXPLANATION_PROMPT

    if language == "nepali":
        language_instruction = "Answer completely in Nepali. Quote exact text from context and explain in Nepali only."
    else:
        language_instruction = "Answer completely in English. Translate the Nepali context into clear English."

    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        question=query_text, context=context, language_instruction=language_instruction
    )

    print("Sending to LLM...")
    response = gemini_model.generate_content(prompt)
    answer = response.text

    # DATE NORMALIZATION — only for English answers
    if language == "english":
        answer = normalize_bs_date(answer)

    sources = [
        {
            "id": d.metadata.get("id"),
            "source_type": d.metadata.get("source_type", "txt"),
            "title": d.metadata.get("title", ""),
            "text": d.metadata.get("_original_content", d.page_content),
        }
        for d in top_docs
    ]

    print(f"[Mode: {mode.upper()} | Chunks used: {len(top_docs)}]")

    return {"answer": answer, "mode": mode, "language": language, "sources": sources}


# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    result = query_rag(args.query_text)
    print("\n================ ANSWER ================\n")
    print(result["answer"])
    print("\n================ SOURCES ================\n")
    for s in result["sources"]:
        print(f"{s['source_type']} | {s['id']}")


if __name__ == "__main__":
    main()
