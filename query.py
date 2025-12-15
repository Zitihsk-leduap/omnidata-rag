import argparse
from langchain_chroma import Chroma  # updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from generate_embeddings import get_embeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the context below:
{context}

---

Answer the question based on the above context: {question}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text to search for.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the vector database
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the database
    results = db.similarity_search_with_score(query_text, k=3)  # smaller k to prevent hangs

    # Filter API docs to appear first if query mentions 'latest' or 'recent'
    if "latest" in query_text.lower() or "recent" in query_text.lower():
        api_docs = [(doc, store) for doc, store in results if doc.metadata.get("source_type") == "api"]
        other_docs = [(doc, store) for doc, store in results if doc.metadata.get("source_type") != "api"]
        results = api_docs + other_docs

    # Prepare context
    context_text = "\n\n---\n\n".join(
        [f"[{doc.metadata.get('source_type','unknown')}]\n{doc.page_content}" for doc, _ in results]
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format_prompt(context=context_text, question=query_text)

    print("🔹 Retrieved chunks from Chroma:", len(results))
    print("🔹 Sending prompt to LLM...")

    # Initialize LLM
    model = OllamaLLM(model="smollm2", streaming=False, timeout=30)

    try:
        response_text = model.invoke(prompt)
        print("🔹 LLM response received")
    except Exception as e:
        response_text = f"⚠️ LLM error: {e}"

    # Prepare sources for reference
    sources = [f"{doc.metadata.get('source_type','unknown')} | {doc.metadata.get('id')}" for doc, _ in results]
    formatted_response = f"Response:\n{response_text}\n\nSources:\n" + "\n".join(sources)

    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
