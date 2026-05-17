import argparse
import os

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
# PROMPT
# -----------------------------
PROMPT_TEMPLATE = """
You are a STRICT document QA system for Nepal legal documents.

RULES:
- Answer ONLY from the given context.
- Do NOT use outside knowledge.
- Do NOT assume or guess missing information.
- If answer is not EXACTLY present, say:
  "Not clearly mentioned in the provided context."
- Keep answer short and factual.

Context:
{context}

Question:
{question}

Answer:
"""

# -----------------------------
# DB
# -----------------------------
embedding_function = get_embeddings()

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function,
)

print("TOTAL CHUNKS IN DB:", len(db.get()["ids"]))


# -----------------------------
# QUERY FUNCTION
# -----------------------------
def query_rag(query_text: str, k: int = 3, return_docs: bool = False):

    query_text = query_text.strip()

    # ✅ MMR retrieval (correct)
    results = db.max_marginal_relevance_search(
        query_text,
        k=k,
        fetch_k=20,
        lambda_mult=0.7
    )

    # -----------------------------
    # DEBUG
    # -----------------------------
    print("\n===== RETRIEVAL DEBUG =====")

    for i, doc in enumerate(results):
        print(f"\n[{i}]")
        print("ID:", doc.metadata.get("id"))
        print(doc.page_content[:250])

    print("\nRetrieved chunks:", len(results))

    # -----------------------------
    # CONTEXT BUILD (FIXED)
    # -----------------------------
    context_text = "\n\n---\n\n".join(
        [
            f"[{doc.metadata.get('id')}]\n{doc.page_content}"
            for doc in results
            if doc.page_content and len(doc.page_content.strip()) > 50
        ]
    )

    # -----------------------------
    # PROMPT
    # -----------------------------
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format_prompt(
        context=context_text,
        question=query_text
    )

    print("\nSending prompt to LLM...")

    # -----------------------------
    # LLM
    # -----------------------------
    model = OllamaLLM(
        model="mistral",
        temperature=0.1,
        streaming=False,
        timeout=60
    )

    try:
        response_text = model.invoke(prompt)
    except Exception as e:
        response_text = f"LLM error: {e}"

    # -----------------------------
    # SOURCES (FIXED)
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