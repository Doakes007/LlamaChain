from collections import defaultdict

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from llm import get_llm   # Centralized Ollama config


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


# ====================================================
# 🔹 COMBINED SUMMARY (EXISTING FUNCTION — UNCHANGED)
# ====================================================
def summarize_documents(documents):
    """
    documents: List[LangChain Document]
    returns: bullet-point combined summary (string)
    """

    if not documents:
        return "No documents available to summarize."

    # 🔒 CPU-safe summarization (stable)
    llm = get_llm(mode="summary")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    # 🔴 Safety limit for 6 GB GPU / CPU stability
    docs = docs[:6]

    # Optional progress logging
    for i, _ in enumerate(docs):
        print(f"Summarizing combined chunk {i + 1}/{len(docs)}")

    chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=SUMMARY_PROMPT
    )

    summary = chain.run(docs)
    return summary.strip()


# ====================================================
# 🔹 PER-DOCUMENT SUMMARY (NEW FUNCTION)
# ====================================================
def summarize_per_document(documents):
    """
    documents: List[LangChain Document]
    returns: dict { filename: summary }
    """

    if not documents:
        return {}

    # 🔒 CPU-safe summarization
    llm = get_llm(mode="summary")

    # ----------------------------
    # GROUP DOCUMENTS BY FILE
    # ----------------------------
    file_groups = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        file_groups[source].append(doc)

    summaries = {}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    # ----------------------------
    # SUMMARIZE EACH FILE SEPARATELY
    # ----------------------------
    for filename, docs in file_groups.items():
        chunks = text_splitter.split_documents(docs)

        # 🔴 Safety limit per document
        chunks = chunks[:6]

        print(f"Summarizing document: {filename}")

        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=SUMMARY_PROMPT
        )

        summaries[filename] = chain.run(chunks).strip()

    return summaries

def summarize_by_topic(documents):
    """
    documents: List[LangChain Document]
    returns: dict { topic: summary }
    """

    if not documents:
        return {}

    llm = get_llm(mode="summary")

    # 🔹 Define topics with keywords
    TOPICS = {
        "Objectives": ["objective", "aim", "goal", "purpose"],
        "Methodology": ["method", "approach", "architecture", "framework", "pipeline"],
        "Literature Review": ["literature", "related work", "existing system"],
        "Limitations": ["limitation", "challenge", "drawback", "constraint"],
        "Conclusion": ["conclusion", "future work", "summary"]
    }

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    topic_summaries = {}

    for topic, keywords in TOPICS.items():
        # 🔍 Select relevant chunks
        matched_chunks = [
            doc for doc in chunks
            if any(keyword.lower() in doc.page_content.lower() for keyword in keywords)
        ]

        if not matched_chunks:
            continue

        matched_chunks = matched_chunks[:6]  # safety limit

        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=SUMMARY_PROMPT
        )

        topic_summaries[topic] = chain.run(matched_chunks).strip()

    return topic_summaries
