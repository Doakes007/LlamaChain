from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.rag.rag_query import build_retrieval_chain, ask_question

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load existing vector store
vectorstore = Chroma(
    collection_name="LlamaChainDocs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Build retrieval chain
chain = build_retrieval_chain(vectorstore)

# Ask a test question
print(ask_question(chain, "What is overfitting?"))
