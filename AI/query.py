import argparse
import os
import re
import warnings

# Suppress deprecation warning from google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from FlagEmbedding import FlagReranker

from AI.generate_embeddings import get_embeddings
from AI.hybrid_retrieval import HybridRetriever
from AI.query_rewriting import QueryRewriter

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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

print("CHROMA PATH:", CHROMA_PATH)


# RERANKER
print("Loading multilingual reranker...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
print("Reranker loaded.")


# DATE NORMALIZATION
BS_MONTHS = {
    "बैशाख": "01",
    "जेठ": "02",
    "असार": "03",
    "श्रावण": "04",
    "भदौ": "05",
    "आश्विन": "06",
    "असोज": "06",
    "कार्तिक": "07",
    "मंसिर": "08",
    "पौष": "09",
    "माघ": "10",
    "फाल्गुन": "11",
    "चैत्र": "12",
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
            r"(\d{4})\s*साल\s*([^\s]+)\s*(\d{1,2})\s*गते", repl, english_part
        )
        return nepali_part + "—" + english_part
    else:
        # No dash separator, apply normally
        return re.sub(r"(\d{4})\s*साल\s*([^\s]+)\s*(\d{1,2})\s*गते", repl, text)


# LANGUAGE DETECTION
def detect_language(text: str) -> str:
    """Detect if query is Nepali or English."""
    nepali_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "english"
    nepali_ratio = nepali_chars / total_chars
    if nepali_ratio > 0.3:
        return "nepali"
    return "english"


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

{language_instruction}

Answer:"""


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

{language_instruction}

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
    language = detect_language(query_text)
    print(f"Language detected: {language.upper()}")

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

    # CHUNK SELECTION (relative relevance - production approach)
    print("Selecting chunks by relative relevance...")

    if mode == "fact":
        # For facts: top 1-2 based on gap
        top_score = reranked[0][0]
        second_score = reranked[1][0] if len(reranked) > 1 else -999
        gap = top_score - second_score

        if gap > 2.0:
            top_docs = [reranked[0][1]]
            print(f"High confidence — using TOP 1 chunk (gap: {gap:.2f})")
        else:
            top_docs = [d for _, d in reranked[:2]]
            print(f"Moderate confidence — using TOP 2 chunks (gap: {gap:.2f})")
    else:
        # For explanations: include chunks within 5 points of best score
        top_score = reranked[0][0]
        relevance_threshold = top_score - 5.0

        top_docs = [d for score, d in reranked if score > relevance_threshold]
        top_docs = top_docs[:5]  # Cap at 5 to avoid over-context

        print(
            f"Explanation mode — using TOP {len(top_docs)} highly relevant chunks (within 5 pts of best)"
        )

    # DEBUG OUTPUT
    print(f"\n===== TOP CHUNKS (mode={mode.upper()}) =====")
    for i, (score, d) in enumerate(reranked[: len(top_docs)]):
        print(f"\n[{i}] Score: {score:.4f} | ID: {d.metadata.get('id')}")
        print(d.page_content[:300])
        print("...")

    # CONTEXT BUILD
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(top_docs)
    )

    # PROMPT + LLM CONFIG
    if mode == "fact":
        prompt_template = FACT_PROMPT
    else:
        prompt_template = EXPLANATION_PROMPT

    if language == "nepali":
        language_instruction = "Answer completely in Nepali. Quote exact text from context and explain in Nepali only."
    else:
        language_instruction = "Answer completely in English. Translate the Nepali context into clear English."

    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        question=query_text, context=context, language_instruction=language_instruction
    )

    print("\nSending to LLM...\n")
    response = gemini_model.generate_content(prompt)
    answer = response.text

    # DATE NORMALIZATION — only for English answers
    if language == "english":
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
