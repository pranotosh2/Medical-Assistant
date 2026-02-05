from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", "")
    return text.strip()

def text_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

def minimal_docs(split_docs):
    cleaned_docs = []
    for doc in split_docs:
        cleaned_docs.append(
            Document(
                page_content=clean_text(doc.page_content),
                metadata=doc.metadata
            )
        )
    return cleaned_docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
