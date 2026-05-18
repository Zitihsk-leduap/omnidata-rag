import argparse
import os
import shutil
import re
import unicodedata
import hashlib
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from generate_embeddings import get_embeddings


#PATHS 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "Data"))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

print("CHROMA PATH:", CHROMA_PATH)
print("DATA PATH:", DATA_PATH)


#CLEAN 
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# LOAD TXT FILES 
def load_txt_files() -> List[Document]:
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )

    docs = loader.load()

    for d in docs:
        d.metadata["source_type"] = "txt"

    return docs


# STRUCTURE SPLIT (IMPORTANT FOR NEPALI LAW) 
def preprocess_documents(docs: List[Document]) -> List[Document]:
    processed = []

    for doc in docs:
        text = clean_text(doc.page_content)

        # split by legal markers (NEPALI + ENGLISH)
        sections = re.split(r'(?=\d+\.)|(?=धारा)|(?=परिच्छेद)|(?=Section)', text)

        for sec in sections:
            sec = sec.strip()

            if len(sec) < 80:
                continue

            processed.append(
                Document(
                    page_content=sec,
                    metadata=doc.metadata.copy()
                )
            )

    print("After preprocessing:", len(processed))
    return processed


#CHUNKING
def split_documents(docs: List[Document]) -> List[Document]:
    chunks = []

    # clause markers (legal structure)
    clause_pattern = r'(\([क-ह]+\))'

    for d in docs:
        text = d.page_content

        # STEP 1: split by clauses first (MOST IMPORTANT)
        parts = re.split(clause_pattern, text)

        buffer = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # if it's a clause marker, attach it
            if re.match(clause_pattern, part):
                buffer += " " + part
                continue

            buffer += " " + part

            # STEP 2: finalize chunk when it becomes meaningful size
            if len(buffer) > 400:
                cleaned = clean_text(buffer)

                if len(cleaned) > 60:
                    chunks.append(
                        Document(
                            page_content=cleaned,
                            metadata=d.metadata.copy()
                        )
                    )

                buffer = ""

        # leftover buffer
        if buffer.strip():
            cleaned = clean_text(buffer)

            if len(cleaned) > 60:
                chunks.append(
                    Document(
                        page_content=cleaned,
                        metadata=d.metadata.copy()
                    )
                )

    print("Final chunks:", len(chunks))
    return chunks
    

# VECTOR STORE 
def add_to_vectorstore(chunks: List[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )

    print("BEFORE:", len(db.get().get("ids", [])))

    for c in chunks:
        c.metadata["id"] = hashlib.md5(
            c.page_content.encode()
        ).hexdigest()

    # avoid duplicates in same run
    seen = set()
    unique = []

    for c in chunks:
        if c.metadata["id"] not in seen:
            unique.append(c)
            seen.add(c.metadata["id"])

    db.add_documents(
        unique,
        ids=[c.metadata["id"] for c in unique]
    )

    print("AFTER:", len(db.get().get("ids", [])))


# ---------------- CLEAR DB ----------------
def clear_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("DB CLEARED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_db", action="store_true")
    args = parser.parse_args()

    if args.clear_db:
        clear_db()

    print("Loading TXT files...")
    docs = load_txt_files()

    print("Preprocessing...")
    docs = preprocess_documents(docs)

    print("Chunking...")
    chunks = split_documents(docs)

    print("Saving to DB...")
    add_to_vectorstore(chunks)

    print("DONE")


if __name__ == "__main__":
    main()