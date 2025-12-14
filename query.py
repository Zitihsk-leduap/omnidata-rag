import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.community.llms import Ollama


from generate_embeddings import get_embeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the context below:
{context}

---

Answer the question based on the above context: {question}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text",type=str,help="The query text to search for.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)



def query_rag(query_text: str):

    #prepare the db
    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    #search the db
    results = db.similarity_search_with_score(query_text, k=3)


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format_prompt(context=context_text, question=query_text)


    #print output
    model = Ollama(model="nomic-embed-text")
    response_text = model.invoke(prompt)


    sources = [doc.metadata.get("id",None) for doc, _score in results]
    formatted_response = f"Response:\n{response_text}\n\nSources:\n" + "\n".join(sources)
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()