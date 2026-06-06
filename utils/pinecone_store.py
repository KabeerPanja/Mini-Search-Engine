import os
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

index_name = st.secrets["PINECONE_INDEX"]

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


def upsert_vectors(vectors):
    index.upsert(vectors=vectors)


def search_vector(vector, top_k=5):
    return index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )
