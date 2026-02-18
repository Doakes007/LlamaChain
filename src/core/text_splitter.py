# src/core/text_splitter.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


def split_documents(documents):
    """
    Splits documents into chunks for embedding (RAG only).
    IMPORTANT: Returns LangChain Document objects (DO NOT convert to dict).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # ✅ Keep Documents as Documents
    return splitter.split_documents(documents)
