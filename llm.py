from langchain_community.llms import Ollama


def get_llm(mode="rag"):
    return Ollama(
        model="llama3:8b-instruct-q4_K_M",
        temperature=0.1,
        num_ctx=2048,
        num_predict=256,
    )
