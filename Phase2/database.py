# database.py
import pinecone
from pinecone import Pinecone, ServerlessSpec

def setup_database(api_key, index_name, dimension):
    pc = Pinecone(api_key=api_key)
    existing_indices = pc.list_indexes().names()
    if index_name not in existing_indices:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        index = pc.Index(index_name)
    else:
        index = pc.Index(index_name)
    return index

def batch_data(data, batch_size=100):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]
