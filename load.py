import argparse
import os
import shutil
from typing import List
import requests

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

API_KEY = "pub_f1220880c32346cd8f11360ae3eb6ae5"
url = f"https://newsdata.io/api/1/news?apikey={API_KEY}&country=np&language=en"

from generate_embeddings import get_embeddings


DATA_PATH = "Data"
CHROMA_PATH = "chroma"


def load_pdfs() -> List[Document]:
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"]="pdf"

    return documents



def load_api_data() -> List[Document]:
    documents = []

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching API data: {e}")
        return documents
    
    # iterate over articles
    for item in data.get("results", []):  # results is a list
        content = f"""
            Title: {item.get('title', 'N/A')}
            Description: {item.get('description','N/A')}
            Date: {item.get('pubDate','N/A')}
            Link: {item.get('link','N/A')}
"""
        documents.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "source_type": "api",
                    "api_name": "newsdata",
                    "timestamp": item.get("pubDate","N/A"),
                    "link": item.get("link","N/A")
                },
            )
        )

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    chunks = []

    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    api_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )

    for doc in documents:
        if doc.metadata.get("source_type")=="api":
            chunks.extend(api_splitter.split_documents([doc]))

        else:
            chunks.extend(pdf_splitter.split_documents([doc]))
    
    return chunks



def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    chunk_index = 0

    for chunk in chunks:
        source_type = chunk.metadata.get("source_type", "unknown")
        source_file = chunk.metadata.get("api_name", "unknown_api") if chunk.metadata.get("source_type")=="api" else os.path.basename(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page", 0)
        page_id = f"{source_type}_{source_file}_page_{page}"

        if page_id == last_page_id:
            chunk_index += 1
        else:
            chunk_index = 0

        chunk.metadata["id"] = f"{page_id}_chunk_{chunk_index}"
        last_page_id = page_id

    return chunks




def add_to_vectorstore(chunks: List[Document]) -> None:
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
    )

    chunks = calculate_chunk_ids(chunks)

    existing_items = db.get()
    existing_ids = set(existing_items["ids"])

    new_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks to Chroma")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print(" No new chunks to add — database is up to date")


def clear_database() -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(" Chroma database cleared")


def load_documents() -> List[Document]:
    documents = []
    documents.extend(load_pdfs())
    documents.extend(load_api_data())
    return documents


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma DB")
    parser.add_argument(
        "--clear_db",
        action="store_true",
        help="Clear the existing Chroma database before ingesting",
    )
    args = parser.parse_args()

    if args.clear_db:
        clear_database()

    print("Loading documents...")
    documents = load_documents()
    print(f" Loaded {len(documents)} documents")

    print(" Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f" Created {len(chunks)} chunks")

    print(" Storing chunks in vector database...")
    add_to_vectorstore(chunks)

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
