import argparse
import os
import re

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from generate_embeddings import get_embeddings


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
print("CHROMA PATH:", CHROMA_PATH)

# -----------------------------
# STRICT PROMPT (important)
# -----------------------------
PROMPT_TEMPLATE = """
You are a constitutional legal assistant for Nepal.

RULES:
- Answer ONLY from the provided context.
- Do NOT use outside knowledge.
- Extract the answer directly from the context.
- The context may contain OCR noise or slightly broken Nepali words.
- You must understand the intended meaning of the text.
- If the answer is clearly present, answer confidently.
- Only say "Not clearly mentioned in provided text." if the answer truly does not exist in the context.
- Preserve the language of the question.

Context:
{context}

Question:
{question}

Answer:
"""

# -----------------------------
# Embeddings + DB (IMPORTANT FIX)
# -----------------------------
embedding_function = get_embeddings()

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function,
)

print("TOTAL CHUNKS IN DB:", len(db.get()["ids"]))


# -----------------------------
# Query normalization
# -----------------------------
# def normalize_query(text: str) -> str:
#     text = text.strip()

#     is_nepali = bool(re.search(r'[\u0900-\u097F]', text))

#     if is_nepali:
#         return "Nepali: " + text
#     else:
#         return "English: " + text

def normalize_query(text: str) -> str:
    return text.strip()

# -----------------------------
# Main RAG function
# -----------------------------
def query_rag(query_text: str, k: int = 10, return_docs: bool = False):

    # Normalize query (IMPORTANT FIX)
    query_text = normalize_query(query_text)

    # Retrieval (better for legal QA than MMR)
    results = db.similarity_search_with_score(
        query_text,
        k=k
    )

    # -------------------------
    # DEBUG OUTPUT
    # -------------------------
    print("\n===== RETRIEVAL DEBUG =====")
    for i, (doc, score) in enumerate(results):
        print(f"\n[{i}] SCORE: {score}")
        print("ID:", doc.metadata.get("id"))
        print(doc.page_content[:250])

    print("\nRetrieved chunks:", len(results))

    # -------------------------
    # Build context
    # -------------------------
    context_text = "\n\n---\n\n".join(
        [
            f"[{doc.metadata.get('source_type', 'unknown')}]\n{doc.page_content}"
            for doc, _ in results
        ]
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format_prompt(
        context=context_text,
        question=query_text
    )

    print("\nSending prompt to LLM...")

    # -------------------------
    # LLM (Mistral via Ollama)
    # -------------------------
    model = OllamaLLM(
        model="mistral",
        temperature=0.2,
        streaming=False,
        timeout=60
    )

    try:
        response_text = model.invoke(prompt)
    except Exception as e:
        response_text = f"LLM error: {e}"

    # -------------------------
    # Sources formatting
    # -------------------------
    sources = [
        f"{doc.metadata.get('source_type','unknown')} | {doc.metadata.get('id')}"
        for doc, _ in results
    ]

    formatted_response = (
        f"\nResponse:\n{response_text}\n\nSources:\n" + "\n".join(sources)
    )

    print(formatted_response)

    if return_docs:
        return response_text, [doc.page_content for doc, _ in results]

    return response_text


# -----------------------------
# CLI entry
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()

    query_rag(args.query_text)


if __name__ == "__main__":
    main()