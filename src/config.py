import os
from dotenv import load_dotenv




load_dotenv()


CHROMA_DIR = "./chroma_store"


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Flag to control whether to use the Gemini models
USE_GEMINI = True 


API_KEY = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))

# Text generation model (2.5 Flash is the fast, general-purpose model)
GEMINI_MODEL = "gemini-2.5-flash" 

# Vision model for multimodal tasks (2.5 Flash handles multimodal input)
VISION_MODEL = "gemini-2.5-flash" 

# Ollama model (used when USE_GEMINI is False or for comparison)
OLLAMA_MODEL = "llama3:8b"


if USE_GEMINI and not API_KEY:
    print("\n" + "="*60)
    print("⚠️  WARNING: GEMINI API Key not found!")
    print("="*60)
    print("\nPlease set your API key in the .env file:")

    print("  1. Create a .env file in the project root")
    print("  2. Add: GOOGLE_API_KEY=your-api-key-here")
    print("  3. Get key from: https://ai.google.dev/gemini-api/docs/api-key")
    print("\n" + "="*60 + "\n")