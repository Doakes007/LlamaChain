from langchain_ollama import OllamaLLM

_llm_cache: dict = {}


def get_llm(mode: str = "rag"):
    global _llm_cache

    if mode in _llm_cache:
        return _llm_cache[mode]

    # 🔥 BEST MODEL CHOICE
    model_name = "mistral:7b-instruct"

    if mode == "summary":
        llm = OllamaLLM(
            model=model_name,
            num_gpu=0,
            num_thread=8,
            num_ctx=1536,   # 🔥 reduced (faster)
            num_predict=300,
            temperature=0.2,
        )
    else:
        llm = OllamaLLM(
            model=model_name,
            num_gpu=0,
            num_thread=8,
            num_ctx=1536,
            num_predict=400,
            temperature=0.1,
        )

    _llm_cache[mode] = llm
    return llm


def invalidate_llm_cache():
    global _llm_cache
    _llm_cache.clear()
    print("LLM cache cleared")