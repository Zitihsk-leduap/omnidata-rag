import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from generate_embeddings import get_embeddings
from langchain.vectorstores.chroma import Chroma

Data_Path = "Data"


# this function is repsonsible for loading the documents from the Data directory 
def load_documents():
    document_loader = PyPDFDirectoryLoader(Data_Path)
    return document_loader.load()



# this function is responsible for splitting the documents into smaller chunks, since the data is too large to be proecessed at once
def split_documents(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,    
    )
    return text_splitter.split_documents(documents)


# Storing and Updating the Data
def add_to_vectorstore(chunks:list[Document]):

    #Load the existing database
    db = Chroma(
        persist_directory="chroma", embedding_function=get_embeddings()
    )


    #calculate page ids
    chunks_with_ids =calculate_chunk_ids(chunks)

    # Add or update the documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    print(f"Existing IDs in the database: {existing_ids}")


    new_chunks =[]

    for chunk in chunks_with_ids:
        if chunks.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new chunks to the database.")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add.documents(new_chunks, ids=new_chunks_ids)
        db.persist()
    else:
        print("No new chunks to add. Database is up to date.")





def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}_page_{page}"


        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            
        chunk_id = f"{current_page_id}_chunk_{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks




def clear_database():
    if os.path.exists("chroma"):
        shutil.rmtree("chroma")





def main():
    parser = argparse.ArgumentParser(description="Load and process PDF documents.")
    parser.add_argument(
        "--clear_db",
        action="store_true",
        help="Clear the existing vector store database before loading new documents.",
    )
    args = parser.parse_args()

    if args.clear_db:
        clear_database()
        print("Cleared the existing vector store database.")

    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Adding chunks to vector store...")
    add_to_vectorstore(chunks)
    print("Process completed.")


if __name__ == "__main__":
    main()