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
# RERANKER
# -----------------------------
print("Loading multilingual reranker...")

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=False
)

print("Reranker loaded.")


# -----------------------------
# DATE NORMALIZATION (NEW ADDITION)
# -----------------------------
BS_MONTHS = {
    "बैशाख": "01",
    "जेठ": "02",
    "असार": "03",
    "श्रावण": "04",
    "भदौ": "05",
    "आश्विन": "06",
    "कार्तिक": "07",
    "मंसिर": "08",
    "पौष": "09",
    "माघ": "10",
    "फाल्गुन": "11",
    "चैत्र": "12"
}

def normalize_bs_date(text: str) -> str:
    import re

    def repl(match):
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)

        month_num = BS_MONTHS.get(month, "00")

        return f"{year}-{month_num}-{int(day):02d}"

    pattern = r"(\d{4})\s*साल\s*([^\s]+)\s*(\d{1,2})\s*गते"
    return re.sub(pattern, repl, text)


# -----------------------------
# PROMPT
# -----------------------------
PROMPT_TEMPLATE = """
You are a STRICT LEGAL TEXT EXTRACTION SYSTEM.

ABSOLUTE RULES:
1. Use ONLY the provided context.
2. DO NOT explain.
3. DO NOT infer.
4. DO NOT translate Nepali text.
5. DO NOT convert or change meaning.
6. COPY EXACT sentence(s) from context.
7. Keep numbers and dates EXACT as in source.

QUESTION:
{question}

CONTEXT:
{context}

EXTRACTED ANSWER:
"""


# -----------------------------
# DATABASE
# -----------------------------
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embeddings(),
)

print("TOTAL CHUNKS:", len(db.get()["ids"]))


# -----------------------------
# QUERY FUNCTION
# -----------------------------
def query_rag(query_text: str, k: int = 10):

    query_text = query_text.strip()

    print("\nSearching vector DB...")

    # -----------------------------
    # RETRIEVAL
    # -----------------------------
    docs = db.similarity_search(query_text, k=k)

    if not docs:
        print("No documents found.")
        return

    print(f"Retrieved {len(docs)} candidate chunks.")

    # -----------------------------
    # RERANKING
    # -----------------------------
    print("\nReranking results...")

    pairs = [(query_text, d.page_content[:1000]) for d in docs]

    scores = reranker.compute_score(pairs)

    reranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    top_docs = [d for _, d in reranked[:5]]

    # -----------------------------
    # DEBUG OUTPUT
    # -----------------------------
    print("\n===== TOP CHUNKS =====")
    for i, d in enumerate(top_docs):
        print(f"\n[{i}] ID: {d.metadata.get('id')}")
        print(d.page_content[:300])

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
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        question=query_text,
        context=context
    )

    # -----------------------------
    # LLM
    # -----------------------------
    model = OllamaLLM(
        model="mistral",
        temperature=0.0,
        num_predict=150
    )

    print("\nSending to LLM...\n")

    answer = model.invoke(prompt)

    # -----------------------------
    # ✅ APPLY NORMALIZATION (ONLY ADDITION)
    # -----------------------------
    answer = normalize_bs_date(answer)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("\n================ ANSWER ================\n")
    print(answer)

    print("\n================ SOURCES ================\n")
    for d in top_docs:
        print(f"{d.metadata.get('source_type','txt')} | {d.metadata.get('id')}")

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