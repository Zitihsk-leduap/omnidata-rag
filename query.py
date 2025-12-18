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


def query_rag(query_text: str, k: int = 5):
    """
    RAG query function.
    
    Args:
        query_text (str): The user query.
        k (int): Number of top documents to retrieve from Chroma.
        
    Returns:
        Tuple[str, List[str]]: (LLM response, list of retrieved doc IDs)
    """
    #  Prepare the vector database
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve top-K similar documents
    results = db.similarity_search_with_score(query_text)

    # Optional: prioritize API docs if query mentions "latest" or "recent"
    if "latest" in query_text.lower() or "recent" in query_text.lower():
        api_docs = [(doc, score) for doc, score in results if doc.metadata.get("source_type") == "api"]
        other_docs = [(doc, score) for doc, score in results if doc.metadata.get("source_type") != "api"]
        results = api_docs + other_docs

    #  Prepare context for LLM
    context_text = "\n\n---\n\n".join(
        [f"[{doc.metadata.get('source_type', 'unknown')}]\n{doc.page_content}" for doc, _ in results]
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format_prompt(context=context_text, question=query_text)

    print("🔹 Retrieved chunks from Chroma:", len(results))
    print("🔹 Sending prompt to LLM...")

    #  Initialize LLM
    model = OllamaLLM(model="smollm2", streaming=False, timeout=30)
    try:
        response_text = model.invoke(prompt)
        print("🔹 LLM response received")
    except Exception as e:
        response_text = f"⚠️ LLM error: {e}"

    # Prepare list of retrieved doc IDs
    retrieved_doc_ids = [doc.metadata.get("id") for doc, _ in results]

    # Print formatted response and sources
    sources = [f"{doc.metadata.get('source_type','unknown')} | {doc.metadata.get('id')}" for doc, _ in results]
    formatted_response = f"Response:\n{response_text}\n\nSources:\n" + "\n".join(sources)
    print(formatted_response)

    #  Return both LLM response and retrieved doc IDs
    return response_text, retrieved_doc_ids

    


if __name__ == "__main__":
    main()
