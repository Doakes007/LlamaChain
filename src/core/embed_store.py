# =====================================================
# embed_store.py
# =====================================================
from langchain_community.vectorstores.utils import filter_complex_metadata


def embed_and_store(chunks, vectorstore):
    """
    Embeds and stores document chunks in ChromaDB.
    Filters metadata to prevent crashes on complex types.
    """
    if not chunks:
        print("embed_and_store: no chunks to store")
        return vectorstore

    try:
        clean_chunks = filter_complex_metadata(chunks)
        vectorstore.add_documents(clean_chunks)
        print(f"embed_and_store: stored {len(clean_chunks)} chunks")
    except Exception as e:
        print(f"embed_and_store failed: {e}")

    return vectorstore