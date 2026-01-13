from src.core.doc_loader import load_documents

files = ["1. Machine Learning Basics.pptx", "2nd phase project.pdf"]
chunks = load_documents(files)

print("Total Chunks:", len(chunks))
print(chunks[:2])
