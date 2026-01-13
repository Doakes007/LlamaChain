
from typing import Union

from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings 

from src.config import USE_GEMINI, API_KEY, EMBEDDING_MODEL, OLLAMA_MODEL 


def get_embeddings() -> Embeddings:
    
    
    is_google_model = any(keyword in EMBEDDING_MODEL.lower() for keyword in ["gemini-embedding", "embedding-001"])

    if USE_GEMINI and is_google_model:
        
        if not API_KEY:
            raise ValueError(
                "GEMINI API Key not set for embedding model. Cannot initialize Google Generative AI Embeddings."
            )
        
        embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, 
            google_api_key=API_KEY,
        )
        print(f"[INFO] Using Google Generative AI Embedding model: {EMBEDDING_MODEL}")
        
    elif "ollama" in EMBEDDING_MODEL.lower():
        embedding_model = OllamaEmbeddings(
            model=EMBEDDING_MODEL
        )
        print(f"[INFO] Using Ollama Embedding model: {EMBEDDING_MODEL}")
        
    else:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
            print(f"[INFO] Using HuggingFace Embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"[ERROR] HuggingFace embeddings failed. Using a simpler Ollama model as fallback. Error: {e}")
            embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)
            
    return embedding_model