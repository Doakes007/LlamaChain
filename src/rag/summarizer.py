from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm import get_llm   # ✅ uses Ollama + GPU automatically


# ----------------------------
# BULLET SUMMARY PROMPT
# ----------------------------
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
You are an AI assistant.

Summarize the following content into 5 to 7 concise bullet points.
Use clear, simple language.
Do NOT add information that is not present.

Content:
{text}

Bullet Point Summary:
"""
)


# ----------------------------
# SUMMARIZE DOCUMENTS
# ----------------------------
def summarize_documents(documents):
    """
    documents: List[LangChain Document]
    returns: bullet-point summary (string)
    """

    if not documents:
        return "No documents available to summarize."

    # ✅ Initialize LLM ONCE (GPU via Ollama)
    llm = get_llm()

    # ----------------------------
    # SAFE CHUNKING (RTX 3050)
    # ----------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    # 🔴 HARD LIMIT (prevents long hangs)
    docs = docs[:10]

    # Optional progress logging
    for i, _ in enumerate(docs):
        print(f"Summarizing chunk {i + 1}/{len(docs)}")

    # ----------------------------
    # FAST SUMMARIZATION CHAIN
    # ----------------------------
    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",          # ⚡ much faster than map_reduce
        prompt=SUMMARY_PROMPT
    )

    summary = chain.run(docs)
    return summary.strip()
