from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def chunk_text(docs):
    """
    Converts extracted docs into clean, smaller RAG-friendly chunks while keeping metadata.
    """
    chunks = []
    for doc in docs:
        split_texts = text_splitter.split_text(doc["content"])
        for part in split_texts:
            chunks.append({
                "content": part,
                "metadata": doc["metadata"]
            })
    return chunks
