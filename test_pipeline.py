from src.core.doc_loader import load_documents
from src.core.chunker import chunk_text
from src.core.embed_store import embed_and_store

# Files you are using for loader test (kept in root)
files = ["1. Machine Learning Basics.pptx", "2nd phase project.pdf"]

# Step 0 → Load docs
docs = load_documents(files)

# Step 1 → Chunk text
chunks = chunk_text(docs)

# Step 2 → Store embeddings in ChromaDB
vectorstore = embed_and_store(chunks)

# Print status
print("Total refined chunks:", len(chunks))
print("Vectors stored in ChromaDB:", len(chunks))
