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
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
print("Reranker loaded.")


# -----------------------------
# DATE NORMALIZATION
# -----------------------------
BS_MONTHS = {
    "बैशाख": "01", "जेठ": "02", "असार": "03", "श्रावण": "04",
    "भदौ": "05", "आश्विन": "06", "असोज": "06", "कार्तिक": "07",
    "मंसिर": "08", "पौष": "09", "माघ": "10", "फाल्गुन": "11",
    "चैत्र": "12"
}


def normalize_bs_date(text: str) -> str:
    def repl(match):
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        month_num = BS_MONTHS.get(month, "??")
        return f"BS {year}-{month_num}-{int(day):02d}"

    if "—" in text:
        nepali_part, english_part = text.split("—", 1)
        english_part = re.sub(
            r"(\d{4})\s*साल\s*([^\s]+)\s*(\d{1,2})\s*गते",
            repl, english_part
        )
        return nepali_part + "—" + english_part
    return re.sub(
        r"(\d{4})\s*साल\s*([^\s]+)\s*(\d{1,2})\s*गते",
        repl, text
    )


# -----------------------------
# CLASSIFIER
# -----------------------------
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
- What happens when a condition occurs
- Anything requiring combining information from multiple places

Examples:
Q: When did the Company Act come into force? → FACT
Q: How many founders are required? → FACT
Q: What is the definition of a Listed Company? → FACT
Q: Can a director be a Company Secretary? → FACT
Q: What are the grounds for refusing company registration? → EXPLANATION
Q: What is the process for establishing a company? → EXPLANATION
Q: What are the qualifications for a Company Secretary? → EXPLANATION
Q: What is the minimum experience required for a Company Secretary? → EXPLANATION
Q: What happens to application money if share allotment fails? → EXPLANATION
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
        response = classifier_llm.invoke(
            CLASSIFIER_PROMPT.format(question=query)
        )
        result = str(response).strip().upper()

        if "EXPLANATION" in result:
            print(f"Classifier result: {result} -> explanation")
            return "explanation"
        if "FACT" in result:
            print(f"Classifier result: {result} -> fact")
            return "fact"

        print(f"Classifier returned unexpected '{result}' — defaulting to explanation")
        return "explanation"

    except Exception as e:
        print(f"Classifier failed ({e}) — defaulting to explanation")
        return "explanation"


# -----------------------------
# QUERY REWRITER
# -----------------------------
REWRITER_PROMPT = """You are a query translator for Nepal Company Act 2063.

Translate the following question into Nepali legal language as it would appear in the Act.
Output BOTH the original query AND the Nepali translation together on one line.
Do not explain. Do not add section numbers. Just translate naturally.

KEY TRANSLATION TIPS:
- "accumulated losses" = "सञ्चित नोक्सानी" 
- "dividend/profit distribution" = "लाभांश वितरण"
- "writing off" = "हटाइनु / लेखेर हटाउनु"
- "before" = "अगाडी / अघि / पहिले"

Example:
Input: If share allotment fails after extension period what happens to application money?
Output: If share allotment fails after extension period what happens to application money? यदि म्याद थपेपछि पनि शेयर बाँडफाँड हुन नसकेमा दरखास्त रकमको के हुन्छ?

Example:
Input: What is the minimum paid up capital for a public company?
Output: What is the minimum paid up capital for a public company? पब्लिक कम्पनीको न्यूनतम चुक्ता पूँजी कति हुनु पर्छ?

Example:
Input: Can a director also be company secretary?
Output: Can a director also be company secretary? के सञ्चालक कम्पनी सचिव पनि हुन सक्छ?

Example:
Input: Can a company pay dividends before writing off accumulated losses from previous years?
Output: Can a company pay dividends before writing off accumulated losses from previous years? एउटा कम्पनीले अघिल्लो वर्षका सञ्चित नोक्सानी लेखेर हटाउनु अगाडी दिवेंडस् दिन सक्छ?

Input: {question}
Output:"""


def rewrite_query(query: str) -> str:
    try:
        rewriter = OllamaLLM(
            model="qwen2.5:7b",
            temperature=0.0,
            num_predict=80,
        )
        result = rewriter.invoke(
            REWRITER_PROMPT.format(question=query)
        ).strip()

        # Safety check — result must contain original query start
        if query[:20].lower() not in result.lower():
            print("Rewriter went off track — using original query")
            return query

        print(f"Enhanced query: {result}")
        return result

    except Exception as e:
        print(f"Rewriter failed ({e}) — using original query")
        return query


# -----------------------------
# PROMPTS
# -----------------------------
FACT_PROMPT = """Extract the exact answer from the context below.
Answer ONLY from the provided context. Do NOT use any outside knowledge.
Do NOT convert BS dates to AD.

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

Example:
Question: When did the Company Act 2063 come into force?
Context: यो ऐन सम्वत्‌ २०६३ साल असोज २० गतेदेखि प्रारम्भ भएको मानिनेछ ।
Answer: २०६३ साल असोज २० गतेदेखि — The Act came into force from Asoj 20, 2063 BS.

Now answer this:
Question: {question}
Context: {context}
Answer (quote the exact Nepali text first, then after — translate only the key fact):"""


EXPLANATION_PROMPT = """You are a legal assistant for Nepal Company Act 2063.
Answer ONLY from the context chunks provided below.
Do NOT use training knowledge. Do NOT invent laws, acts, or section numbers.
The context chunks ARE the only authoritative source.

QUESTION: {question}

CONTEXT:
{context}

STRICT INSTRUCTIONS:
1. READ FIRST: Examine each chunk carefully. Does it directly answer the question?
2. IF YES — Quote the Nepali text exactly, translate it, then state your conclusion clearly.
3. IF NO — Tell the user the provided chunks don't contain the answer. DO NOT make up an answer or invent information.
4. CRITICAL RULE: You are explicitly forbidden from inferring, guessing, or using general knowledge. If the answer is not clearly in the chunks, say "NOT FOUND IN PROVIDED CHUNKS" rather than inventing facts.
5. IF YES/NO QUESTION: Say YES or NO first (based ONLY on explicit text), then explain with quotes.
6. IF MULTIPLE CONDITIONS: List each one with its source chunk number.
7. DO NOT mention concepts that aren't explicitly in the chunks.

Key Nepali legal words:
    छैन / हुँदैन = cannot / is not allowed
    पर्छ / पर्नेछ = must / is required
    सक्नेछ = can / is allowed
    बाहेक = except
    तर = however / but
    कम्तीमा = at least
    बढीमा = at most
    देहायका = the following
    फिर्ता = refund / return
    म्याद = deadline / period
    दरखास्त = application

EXAMPLE - Answer when NOT found:
Question: What is the maximum age of directors?
Context: [Some chunks about qualifications but no age limit mentioned]
Answer: "NOT FOUND IN PROVIDED CHUNKS. The provided context does not specify a maximum age limit for directors."

ANSWER:"""


# -----------------------------
# DATABASE
# -----------------------------
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embeddings(),
)

print("TOTAL CHUNKS:", len(db.get()["ids"]))


# -----------------------------
# HYBRID RETRIEVER
# -----------------------------
print("Initializing hybrid retriever...")
hybrid_retriever = HybridRetriever(db)

print("Initializing query rewriter...")
query_rewriter = QueryRewriter()


# -----------------------------
# QUERY FUNCTION
# -----------------------------
def query_rag(query_text: str, k: int = 10):

    query_text = query_text.strip()

    # STEP 1 — Classify
    mode = detect_mode(query_text)
    print(f"\nMode detected: {mode.upper()}")

    # STEP 2 — Enhance query for retrieval
    print("Enhancing query with Nepali translation...")
    retrieval_query = rewrite_query(query_text)

    # STEP 3 — Hybrid retrieval using enhanced query
    print("Searching with hybrid retrieval (vector + BM25)...")
    hybrid_results = hybrid_retriever.retrieve(retrieval_query, k=k)

    if not hybrid_results:
        print("No documents found.")
        return

    docs = [doc for _, doc in hybrid_results]
    print(f"Retrieved {len(docs)} candidate chunks.")

    # STEP 4 — Reranking using enhanced query
    print("Reranking results...")
    pairs = [(retrieval_query, d.page_content) for d in docs]
    scores = reranker.compute_score(pairs)
    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    # STEP 5 — Chunk selection with confidence guard
    best_score = reranked[0][0] if reranked else -999

    if mode == "fact":
        # Fact mode — strict, top 1 or 2 based on gap
        if best_score < -1.0:
            print(f"Low confidence for fact query (best: {best_score:.2f}) — refusing to answer")
            print("\n================ ANSWER ================\n")
            print("I could not find sufficiently relevant information to answer this question confidently. Please rephrase or ask about a specific section.")
            print(f"\n[Mode: FACT | Chunks used: 0 — insufficient confidence]")
            return

        second_score = reranked[1][0] if len(reranked) > 1 else -999
        gap = best_score - second_score

        if gap > 2.0:
            top_docs = [reranked[0][1]]
            print(f"High confidence — using TOP 1 chunk (gap: {gap:.2f})")
        else:
            top_docs = [d for _, d in reranked[:2]]
            print(f"Moderate confidence — using TOP 2 chunks (gap: {gap:.2f})")

    else:
        # Explanation mode — adaptive gap-based chunk selection
        # Keeps chunks while they're close in relevance, stops at relevance cliff
        if best_score < -2.0:
            print(f"All chunks below confidence (best: {best_score:.2f}) — refusing to answer")
            print("\n================ ANSWER ================\n")
            print("I could not find sufficiently relevant information to answer this question confidently. Please rephrase or ask about a specific section.")
            print(f"\n[Mode: EXPLANATION | Chunks used: 0 — insufficient confidence]")
            return

        # Adaptive selection: keep chunks until relevance gap > 1.5 (relevance cliff)
        # This lets complex questions keep 3-4 related chunks, but stops at noise
        top_docs = []
        gap_threshold = 1.5
        
        for i, (score, doc) in enumerate(reranked[:10]):
            if i == 0:
                top_docs.append(doc)
            else:
                prev_score = reranked[i-1][0]
                gap = prev_score - score
                
                if gap > gap_threshold:
                    # Large gap detected — relevance cliff, stop here
                    print(f"Relevance cliff at chunk {i} (gap: {gap:.2f} > {gap_threshold}) — stopping")
                    break
                
                top_docs.append(doc)
        
        top_docs = top_docs[:5]  # Cap at 5 to prevent over-context
        print(f"Explanation mode — using {len(top_docs)} chunks (adaptive gap-based selection)")

    # STEP 6 — Debug output
    print(f"\n===== TOP CHUNKS (mode={mode.upper()}) =====")
    for i, (score, d) in enumerate(reranked[:len(top_docs) + 1]):
        kept = "✓" if d in top_docs else "✗ filtered"
        print(f"\n[{i}] Score: {score:.4f} | {kept} | ID: {d.metadata.get('id')}")
        print(d.page_content[:300])
        print("...")

    # STEP 7 — Context build
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}"
        for i, d in enumerate(top_docs)
    )

    # STEP 8 — Prompt selection
    prompt_template = FACT_PROMPT if mode == "fact" else EXPLANATION_PROMPT

    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        question=query_text,  # always use original query for LLM
        context=context
    )

    # STEP 9 — LLM
    model = OllamaLLM(
        model="qwen2.5:7b",
        temperature=0.0,
        num_ctx=4096,
        repeat_penalty=1.1,
    )

    print("\nSending to LLM...\n")
    answer = model.invoke(prompt)

    # STEP 10 — Post-process
    answer = normalize_bs_date(answer)

    # STEP 11 — Output
    print("\n================ ANSWER ================\n")
    print(answer)
    print(f"\n[Mode: {mode.upper()} | Chunks used: {len(top_docs)}]")

    print("\n================ SOURCES ================\n")
    for d in top_docs:
        print(f"{d.metadata.get('source_type', 'txt')} | {d.metadata.get('id')}")

    return answer


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, nargs="+")
    args = parser.parse_args()
    query_text = " ".join(args.query_text)
    query_rag(query_text)


if __name__ == "__main__":
    main()