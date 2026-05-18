import argparse
import os
import re

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from FlagEmbedding import FlagReranker

from generate_embeddings import get_embeddings



# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

print("CHROMA PATH:", CHROMA_PATH)



# RERANKER
print("Loading multilingual reranker...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
print("Reranker loaded.")


# DATE NORMALIZATION
BS_MONTHS = {
    "बैशाख": "01", "जेठ": "02", "असार": "03", "श्रावण": "04",
    "भदौ": "05", "आश्विन": "06", "असोज": "06", "कार्तिक": "07",
    "मंसिर": "08", "पौष": "09", "माघ": "10", "फाल्गुन": "11",
    "चैत्र": "12"
}


def normalize_bs_date(text: str) -> str:
    """Only normalize dates that appear outside of Nepali quoted text."""
    def repl(match):
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        month_num = BS_MONTHS.get(month, "??")
        return f"BS {year}-{month_num}-{int(day):02d}"

    # Split on the em-dash that separates Nepali quote from English translation
    if "—" in text:
        nepali_part, english_part = text.split("—", 1)
        english_part = re.sub(
            r"(\d{4})\s*साल\s*([^\s]+)\s*(\d{1,2})\s*गते",
            repl,
            english_part
        )
        return nepali_part + "—" + english_part
    else:
        # No dash separator, apply normally
        return re.sub(
            r"(\d{4})\s*साल\s*([^\s]+)\s*(\d{1,2})\s*गते",
            repl,
            text
        )



# MODE DETECTION
EXPLANATION_TRIGGERS = [
    "explain", "how does", "how do", "what is the process",
    "what are the rules", "describe", "walk me through",
    "what happens when", "what are the grounds", "how is it",
    "what are the conditions", "what are the steps",
    "what are the duties", "what are the rights",
    "what are the requirements", "how can", "under what"
]

def detect_mode(query: str) -> str:
    q = query.lower()
    for trigger in EXPLANATION_TRIGGERS:
        if trigger in q:
            return "explanation"
    return "fact"




# PROMPTS
FACT_PROMPT = """Extract the exact answer from the context below.

Example:
Question: What is the name of this Act?
Context: यस ऐनको नाम "कम्पनी ऐन, २०६३" रहेको छ।
Answer: "कम्पनी ऐन, २०६३" — the name of this Act is Company Act, 2063.

Example:
Question: What is the minimum number of founders for a Public Company?
Context: पब्लिक कम्पनीको संस्थापनाको लागि कम्तीमा सातजना संस्थापक हुनु पर्नेछ।
Answer: कम्तीमा सातजना — at least 7 founders are required.

Example:
Question: What is the maximum number of shareholders in a Private Company?
Context: प्राइभेट कम्पनीको शेयरधनीहरुको सङ्ख्या एकसय एकभन्दा बढी हुन हुँदैन।
Answer: एकसय एकभन्दा बढी हुन हुँदैन — a Private Company cannot have more than 101 shareholders.

Now answer this:
Question: {question}
Context: {context}
Answer (quote the Nepali text first, then translate only the key fact):"""




EXPLANATION_PROMPT = """You are a legal assistant for Nepal Company Act 2063.
Answer the question using ONLY the text provided in the context chunks below.
Do not use any outside knowledge. Do not summarize general legal concepts.
If the answer is not in the context, say: "The provided context does not contain enough information to answer this question."

QUESTION: {question}

CONTEXT:
{context}

INSTRUCTIONS:
- Read each chunk carefully.
- Extract only what is relevant to the question.
- Use numbered points.
- After each point, cite the chunk it came from in parentheses.
- Do not add anything not present in the context.

ANSWER:"""




# DATABASE
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embeddings(),
)

print("TOTAL CHUNKS:", len(db.get()["ids"]))



# QUERY FUNCTION
def query_rag(query_text: str, k: int = 10):

    query_text = query_text.strip()
    mode = detect_mode(query_text)

    print(f"\nMode detected: {mode.upper()}")

    
    # RETRIEVAL
    print("Searching vector DB...")
    docs = db.similarity_search(query_text, k=k)

    if not docs:
        print("No documents found.")
        return

    print(f"Retrieved {len(docs)} candidate chunks.")

    
    # RERANKING
    print("Reranking results...")
    pairs = [(query_text, d.page_content[:1000]) for d in docs]
    scores = reranker.compute_score(pairs)
    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    
    # CHUNK SELECTION
    if mode == "fact":
        top_score = reranked[0][0]
        second_score = reranked[1][0] if len(reranked) > 1 else -999
        gap = top_score - second_score

        if gap > 2.0:
            top_docs = [reranked[0][1]]
            print(f"High confidence — using TOP 1 chunk (reranker gap: {gap:.2f})")
        else:
            top_docs = [d for _, d in reranked[:2]]
            print(f"Low confidence — using TOP 2 chunks (reranker gap: {gap:.2f})")
    else:
    # Explanation mode — use top 4 but drop chunks scoring below -1.0
    # They are noise and confuse the model
        top_docs = [
        d for score, d in reranked[:4]
        if score > -1.0
    ]
        print(f"Explanation mode — using {len(top_docs)} chunks (filtered noise below -1.0)")

    

    # DEBUG OUTPUT
    print(f"\n===== TOP CHUNKS (mode={mode.upper()}) =====")
    for i, (score, d) in enumerate(reranked[:len(top_docs)]):
        print(f"\n[{i}] Score: {score:.4f} | ID: {d.metadata.get('id')}")
        print(d.page_content[:300])
        print("...")


    
    # CONTEXT BUILD
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}"
        for i, d in enumerate(top_docs)
    )


    
    # PROMPT + LLM CONFIG
    if mode == "fact":
        prompt_template = FACT_PROMPT
        num_predict = 120
    else:
        prompt_template = EXPLANATION_PROMPT
        num_predict = 280

    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        question=query_text,
        context=context
    )

    model = OllamaLLM(
        model="mistral",
        temperature=0.0,
        num_predict=num_predict,
    )

    print("\nSending to LLM...\n")
    answer = model.invoke(prompt)


    
    # DATE NORMALIZATION
    answer = normalize_bs_date(answer)

    
    # OUTPUT
    print("\n================ ANSWER ================\n")
    print(answer)
    print(f"\n[Mode: {mode.upper()} | Chunks used: {len(top_docs)}]")

    print("\n================ SOURCES ================\n")
    for d in top_docs:
        print(f"{d.metadata.get('source_type', 'txt')} | {d.metadata.get('id')}")

    return answer



# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    query_rag(args.query_text)


if __name__ == "__main__":
    main()