# src/core/embed_store.py

def embed_and_store(chunks, vectorstore):
    vectorstore.add_documents(chunks)
    return vectorstore