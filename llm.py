from langchain_ollama import OllamaLLM

_llm_cache: dict = {}


def get_llm(mode: str = "rag"):
    global _llm_cache

    if mode in _llm_cache:
        return _llm_cache[mode]

    if mode == "summary":
        llm = OllamaLLM(
            model="phi3:mini",
            num_gpu=0,
            num_thread=8,
            num_ctx=2048,
            num_predict=400,
            temperature=0.2,
        )
    else:
        llm = OllamaLLM(
            model="phi3:mini",
            num_gpu=0,
            num_thread=8,
            num_ctx=2048,
            num_predict=500,
            temperature=0.1,
        )

    _llm_cache[mode] = llm
    return llm


def invalidate_llm_cache():
    global _llm_cache
    _llm_cache.clear()
    print("LLM cache cleared")