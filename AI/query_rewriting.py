import warnings
import os

# Suppress deprecation warning from google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyBUn1H_aEDD8KIZ8M9ngV9E0j2pbAsK0pw"
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)


FACT_REWRITE_PROMPT = """You are a legal query optimization assistant for Nepal Company Act documents.
Your task is to rewrite user queries to improve keyword-based and semantic search retrieval.

Original Query: {query}

Rewrite the query to:
1. Expand acronyms and legal terms
2. Add synonyms (e.g., "founder" -> "promoter", "shareholder" -> "member")
3. Include related legal concepts that might be in the document
4. Keep it concise but comprehensive

Output ONLY the rewritten query, nothing else:"""


EXPLANATION_REWRITE_PROMPT = """You are a legal query optimization assistant for Nepal Company Act documents.
Your task is to rewrite user queries to improve keyword-based and semantic search retrieval.

Original Query: {query}

This is an explanation/how-to query. Rewrite it to:
1. Include broader related terms and concepts
2. Add synonyms for procedures and processes
3. Include potential related sections or clauses
4. Make it more specific to legal/procedural context
5. Keep it concise but comprehensive

Output ONLY the rewritten query, nothing else:"""


class QueryRewriter:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    def rewrite_query(self, query: str, mode: str = "fact") -> str:
        prompt_template = (
            FACT_REWRITE_PROMPT if mode == "fact" else EXPLANATION_REWRITE_PROMPT
        )
        prompt = prompt_template.format(query=query)

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )
            response = model.generate_content(prompt)
            rewritten = response.text.strip()
            return rewritten
        except Exception as e:
            print(f"Error in query rewriting: {e}")
            return query

    def rewrite_with_explanation(self, query: str, mode: str = "fact") -> tuple:
        rewritten = self.rewrite_query(query, mode)
        if rewritten != query:
            return rewritten, f"Query rewritten from: '{query}'"
        return query, "Query rewritten (no changes needed)"
