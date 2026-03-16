def embed_and_store(chunks, vectorstore):

    vectorstore.add_documents(chunks)

    return vectorstore