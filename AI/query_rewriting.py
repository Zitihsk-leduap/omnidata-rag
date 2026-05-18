from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


FACT_REWRITE_PROMPT = """You are a legal query optimization assistant for Nepal Company Act documents.
Your task is to rewrite user queries to improve keyword-based and semantic search retrieval.

Original Query: {query}

Rewrite the query to:
1. Expand acronyms and legal terms
2. Add synonyms (e.g., "founder" → "promoter", "shareholder" → "member")
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
    def __init__(self, model_name: str = "mistral"):
        self.model = OllamaLLM(
            model=model_name,
            temperature=0.3,
            num_predict=100,
        )

    def rewrite_query(self, query: str, mode: str = "fact") -> str:
        prompt_template = FACT_REWRITE_PROMPT if mode == "fact" else EXPLANATION_REWRITE_PROMPT

        prompt = ChatPromptTemplate.from_template(prompt_template).format(query=query)

        try:
            rewritten = self.model.invoke(prompt).strip()
            return rewritten
        except Exception as e:
            print(f"Error in query rewriting: {e}")
            return query

    def rewrite_with_explanation(self, query: str, mode: str = "fact") -> tuple:
        rewritten = self.rewrite_query(query, mode)
        if rewritten != query:
            return rewritten, f"Query rewritten from: '{query}'"
        return query, "Query rewritten (no changes needed)"
