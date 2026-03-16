from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )

    chunks = []

    for doc in documents:

        metadata = doc.metadata

        # DO NOT split images
        if metadata.get("chunk_type") == "image":
            chunks.append(doc)
            continue

        splits = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(splits):

            meta = metadata.copy()

            meta["chunk_id"] = i
            meta["preview"] = chunk[:200]

            chunks.append(
                Document(
                    page_content=chunk,
                    metadata=meta
                )
            )

    return chunks