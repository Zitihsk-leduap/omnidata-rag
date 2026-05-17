import argparse
import os
import shutil
from typing import List
import re
import unicodedata

from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from generate_embeddings import get_embeddings


# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "Data")
DATA_PATH = os.path.abspath(DATA_PATH)

CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

print("CHROMA PATH:", CHROMA_PATH)
print("DATA PATH:", DATA_PATH)


# ---------------- CLEAN TEXT ----------------
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------- LOAD PDF ----------------
def load_pdfs() -> List[Document]:
    print("DATA PATH CHECK:", DATA_PATH)
    print("FILES:", os.listdir(DATA_PATH))

    loader = DirectoryLoader(DATA_PATH,
    glob="**/*.pdf",
    show_progress=True,
    loader_cls=PyMuPDFLoader)

    documents = loader.load()

    print("RAW DOCUMENTS LOADED:", len(documents))

    for doc in documents:
        doc.metadata["source_type"] = "pdf"

    return documents


# ---------------- SPLIT ----------------
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = []

    for doc in documents:
        split_chunks = splitter.split_documents([doc])

        for chunk in split_chunks:
            # CLEAN TEXT BEFORE EMBEDDING
            chunk.page_content = clean_text(chunk.page_content)
            chunks.append(chunk)

    return chunks


# ---------------- VECTOR STORE ----------------
def add_to_vectorstore(chunks: List[Document]) -> None:
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )

    print("TOTAL BEFORE:", len(db.get()["ids"]))

    # assign IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"chunk_{i}"

    existing_ids = set(db.get().get("ids", []))

    new_chunks = [
        c for c in chunks if c.metadata["id"] not in existing_ids
    ]

    print("NEW CHUNKS:", len(new_chunks))

    if new_chunks:
        db.add_documents(
            new_chunks,
            ids=[c.metadata["id"] for c in new_chunks]
        )

    print("TOTAL AFTER:", len(db.get()["ids"]))


# ---------------- CLEAR DB ----------------
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("DB CLEARED")


# ---------------- LOAD DOCS ----------------
def load_documents():
    return load_pdfs()


# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_db", action="store_true")

    args = parser.parse_args()

    if args.clear_db:
        clear_database()

    print("Loading PDFs...")
    docs = load_documents()
    print("Docs:", len(docs))

    print("Splitting...")
    chunks = split_documents(docs)
    print("Chunks:", len(chunks))

    print("Adding to vector DB...")
    add_to_vectorstore(chunks)

    print("DONE")


if __name__ == "__main__":
    main()