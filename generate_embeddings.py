from langchain_community.embeddings import OllaEmbeddings


# Instantiation of the Ollama Embeddings model
def get_embeddings():
    embeddings = OllaEmbeddings(
    model="llama3",
    )
    return embeddings