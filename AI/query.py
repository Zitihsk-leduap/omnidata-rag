import argparse
import os

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from generate_embeddings import get_embeddings

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

print("CHROMA PATH:", CHROMA_PATH)

# -----------------------------
# STRICT PROMPT (FINAL FIXED VERSION)
# -----------------------------
PROMPT_TEMPLATE = """
You are a STRICT legal EXTRACTION system for Nepali law.

RULES (ABSOLUTE):
1. Use ONLY the provided context.
2. DO NOT infer, explain, or assume anything.
3. DO NOT calculate or derive values.
4. DO NOT use outside knowledge.
5. If exact answer is not in context, say:
   "Answer not found in provided context."

TASK:
- Find the exact sentence(s) that answer the question.
- Return ONLY those sentences as the answer.

Question:
{question}

Context:
{context}

FINAL ANSWER:
"""

# -----------------------------
# DATABASE
# -----------------------------
embedding_function = get_embeddings()

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function,
)

print("TOTAL CHUNKS IN DB:", len(db.get()["ids"]))


# -----------------------------
# RELEVANCE FILTER (IMPORTANT)
# -----------------------------
def filter_relevant(docs, query_text):
    query_terms = query_text.lower().split()

    scored = []
    for doc in docs:
        text = doc.page_content.lower()
        score = sum(1 for t in query_terms if t in text)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    # keep only TOP 3 most relevant chunks (VERY IMPORTANT)
    return [doc for _, doc in scored[:3]]


# -----------------------------
# QUERY FUNCTION
# -----------------------------
def query_rag(query_text: str, k: int = 10, return_docs: bool = False):

    query_text = query_text.strip()

    # -----------------------------
    # RETRIEVAL
    # -----------------------------
    results = db.similarity_search(query_text, k=k)

    # -----------------------------
    # FILTER (CRITICAL FIX)
    # -----------------------------
    results = filter_relevant(results, query_text)

    # -----------------------------
    # DEBUG OUTPUT
    # -----------------------------
    print("\n===== RETRIEVAL DEBUG =====")

    for i, doc in enumerate(results):
        print(f"\n[{i}]")
        print("ID:", doc.metadata.get("id"))
        print(doc.page_content[:250])

    print("\nRetrieved chunks:", len(results))

    # -----------------------------
    # CLEAN CONTEXT BUILD
    # -----------------------------
    context_text = "\n".join(
        f"[{doc.metadata.get('id')}] {doc.page_content}"
        for doc in results
    )

    # -----------------------------
    # PROMPT CREATION
    # -----------------------------
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    final_prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )

    print("\nSending prompt to LLM...")

    # -----------------------------
    # LLM
    # -----------------------------
    model = OllamaLLM(
        model="mistral",
        temperature=0.0,   # IMPORTANT: fully deterministic
        streaming=False,
        timeout=60
    )

    try:
        response_text = model.invoke(final_prompt)
    except Exception as e:
        response_text = f"LLM error: {e}"

    # -----------------------------
    # SOURCES
    # -----------------------------
    sources = [
        f"{doc.metadata.get('source_type','unknown')} | {doc.metadata.get('id')}"
        for doc in results
    ]

    formatted_response = (
        f"\nResponse:\n{response_text}\n\nSources:\n" +
        "\n".join(sources)
    )

    print(formatted_response)

    if return_docs:
        return response_text, [doc.page_content for doc in results]

    return response_text


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()

    query_rag(args.query_text)


if __name__ == "__main__":
    main()