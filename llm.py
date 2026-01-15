from langchain_community.llms import Ollama

def get_llm(mode="rag"):
    if mode == "summary":
        # 🔒 CPU-only (stable)
        return Ollama(
            model="llama3:8b-instruct-q4_K_M",
            temperature=0,
            num_ctx=2048,
            num_predict=512,
            num_gpu=0     # ✅ force CPU
        )

    # 🔥 GPU for RAG
    return Ollama(
        model="llama3:8b-instruct-q4_K_M",
        temperature=0,
        num_ctx=2048,
        num_predict=512,
    )
