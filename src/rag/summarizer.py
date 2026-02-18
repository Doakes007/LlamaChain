from functools import lru_cache
from typing import Dict, List
import hashlib
import re

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from llm import get_llm
from src.core.loader import load_documents


# ====================================================
# PROMPTS
# ====================================================

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following text accurately.
Preserve technical and conceptual details.
Do NOT add new information.

{text}
"""
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Combine the summaries into one coherent summary.
Remove redundancy.
Preserve all important ideas.

{text}
"""
)

FINAL_BULLET_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Convert the text into clean bullet points.

Rules:
- No introductions
- No headings
- One sentence per bullet
- One idea per bullet

{text}
"""
)

TOPIC_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Identify DISTINCT topics present in the document.

For EACH topic:
- Give a short topic title
- List 3–6 bullet points
- Use ONLY the provided text
- Do NOT invent topics

Format:
Topic: <title>
- bullet
- bullet

TEXT:
{text}
"""
)


# ====================================================
# HELPERS
# ====================================================

def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )


def _format_bullets(text: str) -> str:
    bullets = []
    for line in re.split(r"[•\n]", text):
        line = line.strip()
        if len(line) > 25:
            bullets.append(f"- {line}")
    return "\n".join(bullets)


def _hash_paths(paths: tuple) -> str:
    return hashlib.sha256("||".join(paths).encode()).hexdigest()


# ====================================================
# BASE SUMMARIES (FAST MAP–REDUCE)
# ====================================================

@lru_cache(maxsize=8)
def _cached_base_summaries(doc_hash: str, doc_paths: tuple) -> Dict[str, str]:
    docs = load_documents(list(doc_paths))
    llm = get_llm(mode="summary")
    splitter = _splitter()

    grouped: Dict[str, List[Document]] = {}
    for d in docs:
        grouped.setdefault(d.metadata["source"], []).append(d)

    base_summaries = {}

    for source, documents in grouped.items():
        chunks = splitter.split_documents(documents)
        chunks = chunks[:15]

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=MAP_PROMPT,
            combine_prompt=REDUCE_PROMPT,
            verbose=False,
        )

        result = chain.invoke(chunks)
        base_summaries[source] = result["output_text"].strip()

    return base_summaries


def get_base_summaries(doc_paths: tuple) -> Dict[str, str]:
    return _cached_base_summaries(_hash_paths(doc_paths), doc_paths)


# ====================================================
# DERIVED SUMMARIES
# ====================================================

def combined_from_base(base_summaries: Dict[str, str]) -> str:
    llm = get_llm(mode="summary")
    combined = "\n\n".join(base_summaries.values())
    raw = llm.invoke(FINAL_BULLET_PROMPT.format(text=combined))
    return _format_bullets(raw)


def per_doc_from_base(base_summaries: Dict[str, str]) -> Dict[str, str]:
    llm = get_llm(mode="summary")
    out = {}

    for src, text in base_summaries.items():
        raw = llm.invoke(FINAL_BULLET_PROMPT.format(text=text))
        out[src] = _format_bullets(raw)

    return out


def topic_from_base(base_summaries: Dict[str, str]) -> Dict[str, str]:
    llm = get_llm(mode="summary")
    out = {}

    for src, text in base_summaries.items():
        raw = llm.invoke(TOPIC_EXTRACTION_PROMPT.format(text=text))
        out[src] = raw.strip()

    return out
