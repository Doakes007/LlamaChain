from langchain_community.vectorstores.utils import filter_complex_metadata


def embed_and_store(chunks, vectorstore):
    """
    Embeds and stores documents safely in the vector database.

    Fixes:
    - Removes complex metadata (lists, dicts, embeddings)
    - Prevents ChromaDB crashes
    """

    try:
        # -------------------------------------------------
        # 🔥 Clean metadata (VERY IMPORTANT)
        # -------------------------------------------------
        clean_chunks = filter_complex_metadata(chunks)

        # -------------------------------------------------
        # Store in vector DB
        # -------------------------------------------------
        vectorstore.add_documents(clean_chunks)

    except Exception as e:
        print("Embedding/Storage failed:", e)

    return vectorstore