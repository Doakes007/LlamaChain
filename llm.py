from langchain_community.llms import Ollama

def get_llm():
    return Ollama(
        model="llama3:8b-instruct-q4_K_M",
        temperature=0,
    )
