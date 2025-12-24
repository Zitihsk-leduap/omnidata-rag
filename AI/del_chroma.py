from chromadb import Client

client = Client()


collections = client.list_collections()
for collection in collections:
    print("deleting collection:", collection.name)
    client.delete_collection(name=collection.name)