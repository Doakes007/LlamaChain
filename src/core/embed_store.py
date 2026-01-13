from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        collection_name="LlamaChainDocs",
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    return vectorstore
