from langchain_community.llms import Ollama
from functools import lru_cache


@lru_cache(maxsize=2)
def get_llm(mode="rag"):

    # LIGHT MODEL
    if mode == "summary":
        return Ollama(
            model="phi3:mini",
            num_gpu=0,
            num_thread=6,
            num_ctx=1024,
            num_predict=200,
            temperature=0.2
        )

    # HEAVY MODEL (USED RARELY)
    return Ollama(
        model="mistral:instruct",   # 🔥 NOT 7B (important)
        num_gpu=0,
        num_thread=6,
        num_ctx=1536,
        num_predict=300,
        temperature=0.1
    )