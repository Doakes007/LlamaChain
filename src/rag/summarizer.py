from functools import lru_cache
from typing import Dict, List
import hashlib
import re
import numpy as np

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm import get_llm
from src.core.loader import load_documents
import torch


# ====================================================
# DEVICE SETUP (NEW)
# ====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


# ====================================================
# EMBEDDING MODEL (GPU ENABLED)
# ====================================================
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)


# ====================================================
# PROMPTS
# ====================================================
FINAL_BULLET_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Convert this text into clean, informative bullet points.
Rules:
- Each bullet = one complete idea
- No introductions or headings
- Use plain language
- Minimum 10 words per bullet

TEXT:
{text}

BULLET POINTS:"""
)

TOPIC_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Identify the main distinct topics in this document.

For EACH topic write:
Topic: <short title>
- <relevant point from text>
- <relevant point from text>
- <relevant point from text>

Use ONLY information from the text. Do NOT use phrases like "Key point from the text:". Just provide the direct information.

TEXT:
{text}

TOPICS:"""
)


# ====================================================
# HELPERS
# ====================================================
def _splitter():
    return RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)


def _hash_paths(paths: tuple) -> str:
    return hashlib.sha256("||".join(sorted(paths)).encode()).hexdigest()


def _format_bullets(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r'^[•*\-–]\s*', '', line)
        if len(line) > 20:
            lines.append(f"- {line}")
    return "\n".join(lines) if lines else text


# ====================================================
# SAFE LLM CALL
# ====================================================
def safe_llm_call(llm, prompt: str, fallback: str = "Summary unavailable.") -> str:
    try:
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content
        text = str(result).strip()
        if not text or len(text) < 10:
            return fallback
        return text
    except Exception as e:
        print(f"LLM call failed: {e}")
        return fallback


# ====================================================
# EMBEDDING CACHE
# ====================================================
@lru_cache(maxsize=512)
def cached_embedding(sentence: str):
    return embedder.encode(sentence)


# ====================================================
# KEY SENTENCE EXTRACTION (UNCHANGED - GOOD LOGIC)
# ====================================================
def extract_key_sentences(text: str, top_k: int = 5) -> List[str]:
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.?!])\s+', text)
        if len(s.strip()) > 40
    ]

    if len(sentences) <= top_k:
        return sentences

    embeddings = np.array([cached_embedding(s) for s in sentences])
    centroid = np.mean(embeddings, axis=0)

    scores = cosine_similarity([centroid], embeddings)[0]

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    return [s for s, _ in ranked[:top_k]]


# ====================================================
# 🔥 NEW: FAST CHUNK COMPRESSION (NO LLM)
# ====================================================
def compress_chunks(chunks: List[Document]) -> str:
    compressed = []

    for c in chunks[:15]:  # limit chunks for speed
        key_sentences = extract_key_sentences(c.page_content, top_k=3)
        compressed.append(" ".join(key_sentences))

    return "\n\n".join(compressed)


# ====================================================
# 🔥 BASE SUMMARIES (OPTIMIZED)
# ====================================================
@lru_cache(maxsize=8)
def _cached_base_summaries(doc_hash: str, doc_paths: tuple) -> Dict[str, str]:
    docs = load_documents(list(doc_paths))

    # Skip images
    docs = [d for d in docs if d.metadata.get("chunk_type") != "image"]

    llm = get_llm(mode="summary")
    splitter = _splitter()

    grouped: Dict[str, List[Document]] = {}
    for d in docs:
        grouped.setdefault(d.metadata.get("source", "unknown"), []).append(d)

    base_summaries = {}

    for source, documents in grouped.items():
        chunks = splitter.split_documents(documents)

        # 🔥 NEW: NO MAP-REDUCE
        compressed_text = compress_chunks(chunks)

        prompt = f"""
Summarize the following technical content clearly and concisely.
Preserve important concepts and terminology.

{compressed_text[:3000]}

SUMMARY:
"""

        final_summary = safe_llm_call(llm, prompt)

        base_summaries[source] = final_summary.strip()

    return base_summaries


def get_base_summaries(doc_paths: tuple) -> Dict[str, str]:
    return _cached_base_summaries(_hash_paths(doc_paths), doc_paths)


# ====================================================
# DERIVED SUMMARIES (UNCHANGED INTERFACE)
# ====================================================
def combined_from_base(base_summaries: Dict[str, str]) -> str:
    llm = get_llm(mode="summary")

    combined = "\n\n".join(base_summaries.values())[:3000]

    prompt = FINAL_BULLET_PROMPT.format(text=combined)

    raw = safe_llm_call(llm, prompt)

    return _format_bullets(raw)


def per_doc_from_base(base_summaries: Dict[str, str]) -> Dict[str, str]:
    llm = get_llm(mode="summary")

    out = {}

    for src, text in base_summaries.items():
        prompt = FINAL_BULLET_PROMPT.format(text=text[:2000])
        raw = safe_llm_call(llm, prompt)
        out[src] = _format_bullets(raw)

    return out


def topic_from_base(base_summaries: Dict[str, str]) -> Dict[str, str]:
    llm = get_llm(mode="summary")

    out = {}

    for src, text in base_summaries.items():
        prompt = TOPIC_EXTRACTION_PROMPT.format(text=text[:2000])
        raw = safe_llm_call(llm, prompt)
        out[src] = raw.strip()

    return out