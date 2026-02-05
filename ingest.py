
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from data_loader.loader import (
    load_pdf_file,
    text_splitter,
    minimal_docs
)
from utils.embeddings import get_embeddings

load_dotenv()

# ---- Pinecone ----
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# ---- Load & process docs ----
docs = load_pdf_file("documents/")
chunks = text_splitter(docs)
clean_chunks = minimal_docs(chunks)

# ---- Upload to Pinecone ----
PineconeVectorStore.from_documents(
    documents=clean_chunks,
    embedding=get_embeddings(),
    index_name=index_name
)

print("Documents successfully ingested into Pinecone")

