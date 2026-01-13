from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
import os


def load_documents(file_paths):
    """
    Loads PDF and PPTX files into LangChain Document objects
    with proper metadata for filtered retrieval
    """
    documents = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)

        if ext == ".pdf":
            loader = PyPDFLoader(path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = filename
                d.metadata["doc_type"] = "job_description"   # 🔥 KEY FIX

            documents.extend(docs)

        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = filename
                d.metadata["doc_type"] = "lecture_notes"

            documents.extend(docs)

    return documents
