
import os
from typing import List, Optional
from langchain_chroma import Chroma 
from langchain_core.documents import Document


from langchain_community.vectorstores.utils import filter_complex_metadata 


import chromadb 

from src.core.embeddings import get_embeddings
from src.config import CHROMA_DIR



COLLECTION_NAME = "llamachain_docs"

os.makedirs(CHROMA_DIR, exist_ok=True) 


_vectorstore = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


def add_documents(docs: List[Document]) -> int:
    
    vs = get_vectorstore()
    
    cleaned_docs = filter_complex_metadata(docs)
    
    if not cleaned_docs:
        print("[VECTORSTORE] No documents left after filtering complex metadata.")
        return 0
        
    
    ids = vs.add_documents(cleaned_docs)
    
    
    print(f"[INFO] Successfully added {len(ids)} documents to Chroma store at {CHROMA_DIR}")
    
    return len(ids)


def delete_collection():
    """
    Deletes the specified Chroma collection from the persistent directory.
    This effectively clears all embedded documents and metadata for that collection.
    """
    global _vectorstore
    
    print(f"[INFO] Attempting to delete Chroma collection: {COLLECTION_NAME}")
    try:
        
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
       
        client.delete_collection(name=COLLECTION_NAME)
        
        
        _vectorstore = None
        
        print(f"[SUCCESS] Collection '{COLLECTION_NAME}' deleted successfully from {CHROMA_DIR}.")
        
    except ValueError as e:
        if "does not exist" in str(e):
            print(f"[WARNING] Collection '{COLLECTION_NAME}' does not exist. No action needed.")
        else:
            print(f"[ERROR] An unexpected error occurred while deleting the collection: {e}")
    except Exception as e:
        print(f"[ERROR] Could not delete Chroma collection: {e}")