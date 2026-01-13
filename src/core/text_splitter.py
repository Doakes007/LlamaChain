from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Splits documents into chunks for embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    formatted_chunks = []
    for doc in chunks:
        formatted_chunks.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    return formatted_chunks
