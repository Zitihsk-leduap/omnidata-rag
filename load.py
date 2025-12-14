import argparse
import os
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from generate_embeddings import get_embeddings


DATA_PATH = "Data"
CHROMA_PATH = "chroma"


# ----------------------------
# Load PDF documents
# ----------------------------
def load_documents() -> List[Document]:
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


# ----------------------------
# Split documents into chunks
# ----------------------------
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(documents)


# ----------------------------
# Generate deterministic chunk IDs
# ----------------------------
def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        page_id = f"{source}_page_{page}"

        if page_id == last_page_id:
            chunk_index += 1
        else:
            chunk_index = 0

        chunk.metadata["id"] = f"{page_id}_chunk_{chunk_index}"
        last_page_id = page_id

    return chunks


# ----------------------------
# Add documents to Chroma DB
# ----------------------------
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
        print(f"➕ Adding {len(new_chunks)} new chunks to Chroma")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new chunks to add — database is up to date")


# ----------------------------
# Clear Chroma database
# ----------------------------
def clear_database() -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 Chroma database cleared")


# ----------------------------
# Main CLI entry point
# ----------------------------
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

    print("📄 Loading documents...")
    documents = load_documents()
    print(f"📚 Loaded {len(documents)} documents")

    print("✂️ Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"🧩 Created {len(chunks)} chunks")

    print("📦 Storing chunks in vector database...")
    add_to_vectorstore(chunks)

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    main()
