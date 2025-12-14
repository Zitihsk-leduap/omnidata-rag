from langchain_community.embeddings import OllamaEmbeddings


# Instantiation of the Ollama Embeddings model
def get_embeddings():
    embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    )
    return embeddings