# src/core/embed_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 🔒 Create ONCE (important)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="LlamaChainDocs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


def embed_and_store(chunks):
    """
    Stores LangChain Document chunks directly into Chroma.
    """
    # ✅ Correct: Chroma natively supports Document objects
    vectorstore.add_documents(chunks)
    return vectorstore
