
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.chat_models import ChatOllama
from src.config import USE_GEMINI, API_KEY, GEMINI_MODEL, OLLAMA_MODEL, VISION_MODEL 

_llm = None
_vision_llm = None 


def get_llm():
    global _llm
    if _llm is None:
        if USE_GEMINI:
            if not API_KEY:
                raise ValueError(
                    "GEMINI API Key not set. Get your free API key from: "
                    "https://ai.google.dev/gemini-api/docs/api-key"
                )
            _llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=API_KEY, 
                temperature=0.2,
            )
            print(f"[INFO] Using Gemini model: {GEMINI_MODEL}")
        else:
            _llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.2,
            )
            print(f"[INFO] Using Ollama model: {OLLAMA_MODEL}")
    return _llm


def get_vision_llm():
    global _vision_llm
    if _vision_llm is None:
        if USE_GEMINI:
            if not API_KEY:
                raise ValueError(
                    "GEMINI API Key not set for vision model. Get your free API key from: "
                    "https://ai.google.dev/gemini-api/docs/api-key"
                )
            
           
            _vision_llm = ChatGoogleGenerativeAI( 
                model=VISION_MODEL, # Uses gemini-2.5-flash from your config
                google_api_key=API_KEY, 
                temperature=0.0, 
            )
            print(f"[INFO] Using Gemini Vision model: {VISION_MODEL}")
        else:
            _vision_llm = ChatOllama( 
                model=OLLAMA_MODEL, 
                temperature=0.2,
            )
            print(f"[INFO] Using Ollama model for vision fallback: {OLLAMA_MODEL}")
            
    return _vision_llm