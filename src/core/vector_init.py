import chromadb
from chromadb.config import Settings

def get_chroma_client():
    return chromadb.Client(Settings(anonymized_telemetry=False))
