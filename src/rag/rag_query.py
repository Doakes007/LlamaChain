# src/rag/rag_query.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from collections import defaultdict
from llm import get_llm


# =============================
# STRICT RAG PROMPT
# =============================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a retrieval-based assistant.

RULES (STRICT):
- Answer using ONLY the provided context.
- Do NOT use external knowledge.
- If the answer is not in the context, respond EXACTLY:
  "Not specified in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)


# =============================
# BUILD RAG CHAIN (SAFE)
# =============================
def build_retrieval_chain(vectorstore):
    """
    Builds a RetrievalQA chain without over-filtering.
    """
    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 8
        }
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    return chain


# =============================
# ASK QUESTION
# =============================
def ask_question(chain, query):
    """
    Executes RAG query and formats sources.
    """
    res = chain.invoke({"query": query})

    answer = res.get("result", "").strip()
    source_docs = res.get("source_documents", [])

    if answer.startswith("Not specified"):
        return "Not specified in the provided documents."

    if not source_docs:
        return answer

    page_map = defaultdict(set)

    for doc in source_docs:
        filename = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        if isinstance(page, int):
            page_map[filename].add(page)

    formatted_sources = []
    for filename, pages in page_map.items():
        if pages:
            min_p, max_p = min(pages), max(pages)
            if min_p == max_p:
                formatted_sources.append(f"- **{filename}** (Page {min_p})")
            else:
                formatted_sources.append(f"- **{filename}** (Pages {min_p}-{max_p})")
        else:
            formatted_sources.append(f"- **{filename}**")

    return f"""{answer}

---
### 📌 Sources
{chr(10).join(formatted_sources)}
"""
