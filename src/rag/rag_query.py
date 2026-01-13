from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from collections import defaultdict
from llm import get_llm   # ✅ Ollama + GPU handled centrally


# ----------------------------
# RAG PROMPT (STRICT)
# ----------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a retrieval-based assistant.

RULES (STRICT):
- You MUST answer using ONLY the provided context.
- You MUST NOT use any external knowledge.
- If the context does NOT contain the answer, respond EXACTLY with:
  "Not specified in the Job Description."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ----------------------------
# BUILD RAG CHAIN (🔒 FILTERED)
# ----------------------------
def build_retrieval_chain(vectorstore):
    """
    vectorstore: Chroma / FAISS / other LangChain vectorstore
    returns: RetrievalQA chain
    """

    # ✅ Initialize LLM ONCE (GPU via Ollama)
    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": {
                "doc_type": "job_description"
            }
        }
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                  # ⚡ fast + GPU friendly
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    return chain


# ----------------------------
# ASK QUESTION + SOURCES
# ----------------------------
def ask_question(chain, query):
    """
    chain: RetrievalQA chain
    query: user question (string)
    returns: formatted answer with sources (string)
    """

    res = chain.invoke({"query": query})

    answer = res.get("result", "").strip()
    source_docs = res.get("source_documents", [])

    # 🔒 RULE 1: Strict fallback → no sources
    if answer.startswith("Not specified in the Job Description"):
        return "Not specified in the Job Description."

    # 🔒 RULE 2: No source docs → return answer only
    if not source_docs:
        return answer

    # ----------------------------
    # SOURCE FORMATTING (PAGE RANGES)
    # ----------------------------
    page_map = defaultdict(set)

    for doc in source_docs:
        metadata = doc.metadata
        filename = metadata.get("source", "unknown")
        page = metadata.get("page")

        if isinstance(page, int):
            page_map[filename].add(page)

    formatted_sources = []

    for filename, pages in page_map.items():
        if pages:
            min_page = min(pages)
            max_page = max(pages)

            if min_page == max_page:
                formatted_sources.append(
                    f"- **{filename}** (Page: {min_page})"
                )
            else:
                formatted_sources.append(
                    f"- **{filename}** (Pages: {min_page}–{max_page})"
                )
        else:
            formatted_sources.append(f"- **{filename}**")

    sources_text = "\n".join(formatted_sources)

    return f"""{answer}

---
### 📌 Sources
{sources_text}
"""
