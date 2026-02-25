from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from collections import defaultdict
from llm import get_llm


# =============================
# IMPROVED RAG PROMPT (SAFE)
# =============================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a retrieval-based assistant.

RULES:
- Answer using ONLY the provided context.
- Do NOT use outside knowledge.
- Provide a CLEAR, DETAILED explanation.
- Explain concepts step-by-step when possible.
- If multiple points exist, use bullet points.
- Include important definitions, examples, or explanations found in the context.
- If the answer cannot be reasonably inferred from the context, respond EXACTLY:
  "Not specified in the provided documents."

Context:
{context}

Question:
{question}

Detailed Answer:
"""
)


# =============================
# BUILD RAG CHAIN
# =============================
def build_retrieval_chain(vectorstore):
    """
    Builds RetrievalQA chain with stable retrieval.
    """
    llm = get_llm()

    # MMR retrieval → better diversity, same GPU usage
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
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

    # safer fallback handling
    if not answer or answer.lower().startswith("not specified"):
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
                formatted_sources.append(
                    f"- **{filename}** (Page {min_p})"
                )
            else:
                formatted_sources.append(
                    f"- **{filename}** (Pages {min_p}-{max_p})"
                )
        else:
            formatted_sources.append(f"- **{filename}**")

    return f"""{answer}

---
### 📌 Sources
{chr(10).join(formatted_sources)}
"""