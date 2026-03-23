from functools import lru_cache
from typing import Dict, List
import hashlib
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm import get_llm
from src.core.loader import load_documents


# ====================================================
# EMBEDDING MODEL
# ====================================================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


# ====================================================
# PROMPTS
# ====================================================
CHUNK_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Summarize the following technical content clearly and concisely.
Preserve important technical terms and concepts.
Do NOT add information that is not in the text.
Write in complete sentences.

TEXT:
{text}

SUMMARY:"""
)

MERGE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Combine the following summaries into one coherent, non-redundant summary.
Preserve all key ideas. Write in flowing paragraphs.

SUMMARIES:
{text}

COMBINED SUMMARY:"""
)

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
- Key point from the text
- Key point from the text
- Key point from the text

Use ONLY information from the text. Do NOT invent topics.

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
    """Convert LLM bullet output to clean markdown bullets."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Normalize bullet markers
        line = re.sub(r'^[•*\-–]\s*', '', line)
        if len(line) > 20:
            lines.append(f"- {line}")
    return "\n".join(lines) if lines else text


def remove_duplicate_sentences(text: str) -> str:
    seen = set()
    output = []
    for sentence in re.split(r'(?<=[.?!])\s+', text):
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        key = re.sub(r'\s+', ' ', sentence.lower())
        if key not in seen:
            seen.add(key)
            output.append(sentence)
    return " ".join(output)


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
# KEY SENTENCE EXTRACTION
# ====================================================
def extract_key_sentences(text: str, top_k: int = 5) -> List[str]:
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 40]
    if len(sentences) <= top_k:
        return sentences
    embeddings = np.array([cached_embedding(s) for s in sentences])
    centroid = np.mean(embeddings, axis=0)
    scores = cosine_similarity([centroid], embeddings)[0]
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:top_k]]


# ====================================================
# CHUNK SUMMARIZATION
# ====================================================
def summarize_chunk(llm, text: str) -> str:
    key_sentences = extract_key_sentences(text, top_k=5)
    compressed_text = " ".join(key_sentences)
    prompt = CHUNK_SUMMARY_PROMPT.format(text=compressed_text)
    return safe_llm_call(llm, prompt, fallback="[chunk summary unavailable]")


# ====================================================
# PARALLEL CHUNK SUMMARIZATION  (FIX: captures exceptions per chunk)
# ====================================================
def summarize_chunks_parallel(llm, chunks: List[Document]) -> List[str]:
    summaries = [""] * len(chunks)

    def worker(idx, chunk):
        try:
            return idx, summarize_chunk(llm, chunk.page_content)
        except Exception as e:
            print(f"Chunk {idx} summarization failed: {e}")
            return idx, "[chunk summary failed]"

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(worker, i, c): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            try:
                idx, summary = future.result()
                summaries[idx] = summary
            except Exception as e:
                print(f"Future failed: {e}")

    return [s for s in summaries if s]


# ====================================================
# MERGE SUMMARIES
# ====================================================
def merge_summaries(llm, summaries: List[str]) -> str:
    combined = "\n\n".join(summaries)
    combined = remove_duplicate_sentences(combined)
    prompt = MERGE_SUMMARY_PROMPT.format(text=combined[:3000])
    return safe_llm_call(llm, prompt)


# ====================================================
# BASE SUMMARIES  (FIX: hash uses sorted paths, higher chunk limit)
# ====================================================
@lru_cache(maxsize=8)
def _cached_base_summaries(doc_hash: str, doc_paths: tuple) -> Dict[str, str]:
    docs = load_documents(list(doc_paths))

    # Skip image chunks for summarization
    docs = [d for d in docs if d.metadata.get("chunk_type") != "image"]

    llm = get_llm(mode="summary")
    splitter = _splitter()

    grouped: Dict[str, List[Document]] = {}
    for d in docs:
        grouped.setdefault(d.metadata.get("source", "unknown"), []).append(d)

    base_summaries = {}
    for source, documents in grouped.items():
        chunks = splitter.split_documents(documents)
        chunks = chunks[:20]  # FIX: increased from 10 to cover more content

        chunk_summaries = summarize_chunks_parallel(llm, chunks)
        if not chunk_summaries:
            base_summaries[source] = "Summary unavailable."
            continue

        final_summary = merge_summaries(llm, chunk_summaries)
        base_summaries[source] = final_summary.strip()

    return base_summaries


def get_base_summaries(doc_paths: tuple) -> Dict[str, str]:
    return _cached_base_summaries(_hash_paths(doc_paths), doc_paths)


# ====================================================
# DERIVED SUMMARIES
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