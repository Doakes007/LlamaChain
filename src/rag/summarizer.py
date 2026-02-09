from collections import defaultdict
from typing import List, Dict, Iterable
import re

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from llm import get_llm


# ====================================================
# 🔹 PROMPTS
# ====================================================

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
You are a professional summarization assistant.

Summarize the following content accurately and completely.
Preserve all key ideas, concepts, arguments, and explanations.
Do NOT introduce new information.
Do NOT aggressively compress.

CONTENT:
{text}

SUMMARY:
"""
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Combine the following summaries into one coherent summary.

Rules:
- Retain ALL important points
- Merge related ideas naturally
- Do NOT discard minority or less frequent topics

SUMMARIES:
{text}

COMBINED SUMMARY:
"""
)

FINAL_BULLET_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Convert the following summary into bullet points ONLY.

Rules:
- No headings, intros, or explanations
- No nested bullets
- One sentence per bullet
- One idea per bullet

SUMMARY:
{text}

BULLETS:
"""
)


# ====================================================
# 🔹 INTERNAL HELPERS
# ====================================================

def _get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200
    )


def _build_map_reduce_chain(llm):
    return load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=REDUCE_PROMPT
    )


def _batch_documents(
    docs: List[Document],
    batch_size: int = 6
) -> Iterable[Document]:
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        yield Document(
            page_content="\n\n".join(d.page_content for d in batch),
            metadata={"batch": i // batch_size}
        )


def _format_bullets(text: str) -> str:
    text = text.strip()

    junk_prefixes = [
        "here are the bullet points",
        "here is the bullet summary",
        "bullet points",
        "summary"
    ]

    lower = text.lower()
    for junk in junk_prefixes:
        if lower.startswith(junk):
            text = text[len(junk):].strip(": \n")
            break

    bullets = []

    if "•" in text:
        parts = text.split("•")
        for p in parts:
            p = p.strip()
            if len(p) > 10:
                bullets.append(f"- {p}")
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                bullets.append(f"- {s}")

    return "\n".join(bullets)


def _final_bulletize(text: str, llm) -> str:
    raw = llm.invoke(
        FINAL_BULLET_PROMPT.format(text=text)
    ).strip()

    return _format_bullets(raw)


# ====================================================
# 🔥 SINGLE-PASS BASE SUMMARY (CORE FIX)
# ====================================================

def generate_base_summary(documents: List[Document]) -> str:
    """
    EXPENSIVE OPERATION — RUN ONCE.
    Produces a detailed base summary for reuse.
    """

    if not documents:
        return ""

    llm = get_llm(mode="summary")
    splitter = _get_splitter()

    chunks = splitter.split_documents(documents)
    batched = list(_batch_documents(chunks))

    chain = _build_map_reduce_chain(llm)
    result = chain.invoke(batched)

    return result["output_text"].strip()


# ====================================================
# 🔹 COMBINED SUMMARY (CHEAP)
# ====================================================

def summarize_documents(documents: List[Document]) -> str:
    base = generate_base_summary(documents)

    if not base:
        return "No documents available to summarize."

    llm = get_llm(mode="summary")
    return _final_bulletize(base, llm)


# ====================================================
# 🔹 PER-DOCUMENT SUMMARY (LIGHTWEIGHT)
# ====================================================

def summarize_per_document(documents: List[Document]) -> Dict[str, str]:
    if not documents:
        return {}

    base = generate_base_summary(documents)
    llm = get_llm(mode="summary")

    file_groups = defaultdict(list)
    for doc in documents:
        file_groups[doc.metadata.get("source", "unknown")].append(doc)

    summaries = {}

    for filename, docs in file_groups.items():
        text = "\n".join(d.page_content for d in docs)
        extracted = base if len(text) < 5000 else text
        summaries[filename] = _final_bulletize(extracted, llm)

    return summaries


# ====================================================
# 🔹 TOPIC-WISE SUMMARY (FAST)
# ====================================================

def summarize_by_cluster(
    documents: List[Document],
    keyword_clusters: Dict[str, List[str]]
) -> Dict[str, str]:

    if not documents:
        return {}

    base = generate_base_summary(documents)
    llm = get_llm(mode="summary")

    cluster_summaries = {}

    for topic, keywords in keyword_clusters.items():
        if any(k.lower() in base.lower() for k in keywords):
            cluster_summaries[topic] = _final_bulletize(base, llm)

    return cluster_summaries
