import argparse
import os
import re

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from FlagEmbedding import FlagReranker

from generate_embeddings import get_embeddings
from hybrid_retrieval import HybridRetriever
from query_rewriting import QueryRewriter



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
CLASSIFIER_PROMPT = """Classify this question into one of two categories:

FACT — if the question asks for:
- A single specific value (number, date, name, yes/no)
- A threshold or limit (minimum, maximum, how many, how much)
- A direct definition ("what is X defined as")

EXPLANATION — if the question asks for:
- A process or procedure (how does X work, what are the steps)
- Multiple conditions or qualifications (who is eligible, what are the requirements)
- Reasons or grounds (what are the grounds for X)
- Duties, rights, or responsibilities
- Anything that requires combining information from multiple places

Examples:
Q: When did the Company Act come into force? → FACT
Q: How many founders are required? → FACT
Q: What is the definition of a Listed Company? → FACT
Q: Can a director be a Company Secretary? → FACT
Q: What are the grounds for refusing company registration? → EXPLANATION
Q: What is the process for establishing a company? → EXPLANATION
Q: What are the qualifications for a Company Secretary? → EXPLANATION
Q: What is the minimum experience required for a Company Secretary? → EXPLANATION
Q: Explain how dividend distribution works. → EXPLANATION
Q: कम्पनी दर्ता गर्न इन्कार गर्न के के आधारहरू छन्? → EXPLANATION
Q: कम्पनी ऐन २०६३ कहिले लागू भयो? → FACT
Q: कम्पनी सचिवको योग्यता के हो? → EXPLANATION

Question: {question}
Category (reply with FACT or EXPLANATION only):"""

classifier_llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.0,
    num_predict=5,
)


def detect_mode(query: str) -> str:
    try:
        response = classifier_llm.invoke(CLASSIFIER_PROMPT.format(question=query))
        result = str(response).strip().upper()

        if "EXPLANATION" in result:
            print(f"Classifier result: {result} -> explanation")
            return "explanation"
        if "FACT" in result:
            print(f"Classifier result: {result} -> fact")
            return "fact"

        print(f"Warning: unexpected classifier result '{result}', defaulting to explanation")
        return "explanation"
    except Exception as error:
        print(f"Warning: mode classifier failed ({error}), defaulting to explanation")
        return "explanation"




# PROMPTS
FACT_PROMPT = """ROLE: Legal fact extractor for Nepal Company Act 2063.

MANDATORY CONSTRAINTS:
1. Extract ONLY facts explicitly stated in the provided context
2. If the answer is NOT found in context, respond: "Not found in provided context"
3. Do NOT infer, interpret, or apply external legal knowledge
4. Do NOT paraphrase—use exact quotes from context
5. Do NOT expand or generalize beyond what is stated
6. Every factual claim must be traceable to the context

POSITIVE EXAMPLES (Correct Grounding):
Question: What is the name of this Act?
Context: यस ऐनको नाम "कम्पनी ऐन, २०६३" रहेको छ।
Answer: [Direct quote: "यस ऐनको नाम 'कम्पनी ऐन, २०६३' रहेको छ।"] The Act is named Company Act, 2063.

Question: What is the minimum number of founders for a Public Company?
Context: पब्लिक कम्पनीको संस्थापनाको लागि कम्तीमा सातजना संस्थापक हुनु पर्नेछ।
Answer: [Direct quote: "कम्तीमा सातजना संस्थापक हुनु पर्नेछ।"] At least 7 founders are required.

NEGATIVE EXAMPLES (Do NOT Do These):
Question: What qualifications must a founder have?
Context: पब्लिक कम्पनीको संस्थापनाको लागि कम्तीमा सातजना संस्थापक हुनु पर्नेछ।
WRONG: "Founders should be experienced professionals." (Not in context)
CORRECT: "Not found in provided context" (qualifications not mentioned)

Question: How does Nepal Company Act compare to Indian Companies Act?
Context: कम्पनी ऐन, २०६३...
WRONG: "It is similar to Indian law..." (external knowledge)
CORRECT: "Not found in provided context" (comparison not in context)

GROUNDING PROCESS:
1. Read the question carefully
2. Search context for exact answer
3. If found: Quote the relevant text [in brackets], then provide the answer
4. If not found: Respond with "Not found in provided context"
5. Verify: Did I only use context? Did I avoid interpretation?

Question: {question}

Context: {context}

Answer (show source text in [brackets], then the fact):"""




EXPLANATION_PROMPT = """ROLE: Legal explanation assistant for Nepal Company Act 2063.
Extract ONLY from provided context. Zero external knowledge. Zero interpretation.

MANDATORY CONSTRAINTS:
1. Every claim MUST be explicitly stated in the provided context
2. Do NOT infer relationships between sections
3. Do NOT apply general legal knowledge
4. Do NOT paraphrase—quote and cite exact text
5. Do NOT interpret vague language
6. If insufficient context, respond: "Not found in provided context"

POSITIVE EXAMPLES (Correct Grounding):
Question: What are the duties of directors?
Context: [Chunk 1] संचालकहरूको कर्तव्य... क. कंपनीको लाभको लागि काम गर्नु पर्नेछ। ख. शेयरधनीको हितको रक्षा गर्नु पर्नेछ।
CORRECT ANSWER:
1. [From context: "कंपनीको लाभको लागि काम गर्नु पर्नेछ।"] Directors must work for company profit.
2. [From context: "शेयरधनीको हितको रक्षा गर्नु पर्नेछ।"] Directors must protect shareholder interests.

Question: How is a company dissolved?
Context: [Chunk 1] कंपनीको विघटन... कंपनीको बोर्डले विघटनको निर्णय गर्नुपर्छ।
CORRECT ANSWER:
1. [From context: "कंपनीको बोर्डले विघटनको निर्णय गर्नुपर्छ।"] The company board must decide on dissolution.

NEGATIVE EXAMPLES (Do NOT Do These):
Question: What happens to employee benefits after dissolution?
Context: कंपनीको विघटन प्रक्रिया...
WRONG: "Employees typically receive severance packages." (Not in context, external knowledge)
CORRECT: "Not found in provided context" (employee benefits not mentioned)

Question: What is the relationship between the board and shareholders?
Context: [Only says: "Board meets quarterly. Shareholders meet annually."]
WRONG: "The board reports to shareholders." (Inference, not stated)
CORRECT: "Board meets quarterly. Shareholders meet annually." (Only state what's explicit)

GROUNDING INSTRUCTIONS:
For each point in your answer:
a. Show the exact quote from context in [brackets]
b. Extract the fact directly—do not interpret
c. Do not combine quotes to infer new meaning
d. Stop if you cannot ground the answer

QUESTION: {question}

CONTEXT:
{context}

INSTRUCTIONS:
- Use numbered points
- Cite source chunk for each claim [in brackets]
- Quote exact text from context
- Do NOT add interpretation or connections
- If information is missing, stop and respond: "Not found in provided context"

ANSWER:"""




# DATABASE
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embeddings(),
)

print("TOTAL CHUNKS:", len(db.get()["ids"]))


# HYBRID RETRIEVER & QUERY REWRITER
print("Initializing hybrid retriever...")
hybrid_retriever = HybridRetriever(db)

print("Initializing query rewriter...")
query_rewriter = QueryRewriter()



# QUERY FUNCTION
def query_rag(query_text: str, k: int = 10):

    query_text = query_text.strip()
    mode = detect_mode(query_text)

    print(f"\nMode detected: {mode.upper()}")


    # QUERY REWRITING (temporarily disabled)
    print("Query rewriter disabled — using original query")
    rewritten_query = query_text
    rewrite_note = "rewriter disabled"
    print(f"Original: {query_text}")
    print(f"Rewritten: {rewritten_query}")
    print(f"Note: {rewrite_note}")


    # HYBRID RETRIEVAL
    print("Searching with hybrid retrieval (vector + BM25)...")
    hybrid_results = hybrid_retriever.retrieve(rewritten_query, k=k)

    if not hybrid_results:
        print("No documents found.")
        return

    docs = [doc for _, doc in hybrid_results]
    print(f"Retrieved {len(docs)} candidate chunks.")


    # RERANKING (NO TRUNCATION)
    print("Reranking results...")
    pairs = [(rewritten_query, d.page_content) for d in docs]
    scores = reranker.compute_score(pairs)
    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

# CONFIDENCE-BASED FILTERING
    print("Applying confidence filtering...")

    if mode == "fact":
        confidence_threshold = -0.5
        max_chunks = 2
    else:
        confidence_threshold = -3.0
        max_chunks = 5

    filtered = [(score, doc) for score, doc in reranked if score > confidence_threshold]

    if not filtered:
        print(f"No chunks above threshold {confidence_threshold}, using best available")
        filtered = reranked[:1]
    else:
        num_filtered_out = len(reranked) - len(filtered)
        if num_filtered_out > 0:
            print(f"Filtered out {num_filtered_out} low-confidence chunks (score ≤ {confidence_threshold})")

    # CHUNK SELECTION
    if mode == "fact":
        top_score = filtered[0][0]
        second_score = filtered[1][0] if len(filtered) > 1 else -999
        gap = top_score - second_score

        if gap > 2.0:
            top_docs = [filtered[0][1]]
            print(f"High confidence — using TOP 1 chunk (reranker gap: {gap:.2f})")
        else:
            top_docs = [d for _, d in filtered[:max_chunks]]
            print(f"Low confidence — using TOP {len(top_docs)} chunks (reranker gap: {gap:.2f})")
    else:
        top_docs = [d for _, d in filtered[:max_chunks]]
        print(f"Explanation mode — using {len(top_docs)} chunks (threshold: {confidence_threshold})")



    # DEBUG OUTPUT
    print(f"\n===== TOP CHUNKS (mode={mode.upper()}) =====")
    for i, (score, d) in enumerate(filtered[:len(top_docs)]):
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
    else:
        prompt_template = EXPLANATION_PROMPT

    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        question=query_text,
        context=context
    )

    model = OllamaLLM(
        model="qwen2.5:7b",
        temperature=0.0,
        num_ctx=4096,
        repeat_penalty=1.1,
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