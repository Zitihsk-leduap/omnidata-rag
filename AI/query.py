import argparse
import os

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from FlagEmbedding import FlagReranker

from generate_embeddings import get_embeddings


# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

print("CHROMA PATH:", CHROMA_PATH)


# -----------------------------
# MULTILINGUAL RERANKER
# IMPORTANT FIX
# -----------------------------
print("Loading multilingual reranker...")

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=False
)

print("Reranker loaded.")


# -----------------------------
# PROMPT
# -----------------------------
PROMPT_TEMPLATE = """
You are a STRICT Nepali legal extraction system.

RULES:
1. Use ONLY the provided context.
2. DO NOT use outside knowledge.
3. DO NOT infer or assume.
4. DO NOT summarize unrelated content.
5. If answer is not explicitly present, say:
   "Answer not found in provided context."

TASK:
- Extract ONLY the exact legal statement that answers the question.
- Keep the answer concise.

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

print("TOTAL CHUNKS:", len(db.get()["ids"]))


# -----------------------------
# QUERY FUNCTION
# -----------------------------
def query_rag(query_text: str, k: int = 25):

    query_text = query_text.strip()

    print("\nSearching vector DB...")

    # -----------------------------
    # STEP 1: VECTOR SEARCH
    # -----------------------------
    docs = db.similarity_search(
        query_text,
        k=k
    )

    if not docs:
        print("No documents found.")
        return

    print(f"Retrieved {len(docs)} candidate chunks.")

    # -----------------------------
    # STEP 2: MULTILINGUAL RERANK
    # -----------------------------
    print("\nReranking results...")

    pairs = [
        [query_text, d.page_content[:1000]]
        for d in docs
    ]

    scores = reranker.compute_score(pairs)

    reranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    # -----------------------------
    # TAKE TOP 5
    # -----------------------------
    top_docs = [d for _, d in reranked[:5]]

    # -----------------------------
    # DEBUG
    # -----------------------------
    print("\n===== TOP 5 RERANKED CHUNKS =====")

    for i, d in enumerate(top_docs):
        print(f"\n[{i}]")
        print("ID:", d.metadata.get("id"))
        print(d.page_content[:400])

    # -----------------------------
    # CONTEXT BUILD
    # -----------------------------
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('id')}]\n{d.page_content}"
        for d in top_docs
    )

    # -----------------------------
    # PROMPT BUILD
    # -----------------------------
    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    ).format(
        question=query_text,
        context=context
    )

    print("\nSending to LLM...")

    # -----------------------------
    # LLM
    # -----------------------------
    model = OllamaLLM(
        model="mistral",
        temperature=0.0,
        num_predict=256
    )

    try:
        answer = model.invoke(prompt)
    except Exception as e:
        answer = f"LLM ERROR: {e}"

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("\n================ ANSWER ================\n")
    print(answer)

    print("\n================ SOURCES ================\n")

    for d in top_docs:
        print(
            f"{d.metadata.get('source_type', 'unknown')} "
            f"| {d.metadata.get('id')}"
        )

    return answer


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