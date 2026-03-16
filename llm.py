from langchain_community.llms import Ollama
from functools import lru_cache


@lru_cache(maxsize=2)
def get_llm(mode="rag"):

    # Light model for summarization
    if mode == "summary":
        return Ollama(
            model="phi3:mini",
            temperature=0.2,
            num_ctx=2048,
            num_predict=256
        )

    # Strong model for answering
    return Ollama(
        model="mistral:7b-instruct",
        temperature=0.1,
        num_ctx=2048,
        num_predict=400
    )