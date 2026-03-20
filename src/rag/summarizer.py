from functools import lru_cache
from typing import Dict, List
import hashlib
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ====================================================
# PROMPTS
# ====================================================

CHUNK_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following technical content clearly.

Preserve important technical concepts.
Do NOT add information that is not present.

{text}

Summary:
"""
)

MERGE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Combine the following summaries into a single coherent summary.

Remove redundancy but preserve key ideas.

{text}

Final Summary:
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
        chunk_size=900,
        chunk_overlap=150
    )


def _hash_paths(paths: tuple):
    return hashlib.sha256("||".join(paths).encode()).hexdigest()


def _format_bullets(text: str):

    bullets = []

    for line in re.split(r"[•\n\-]", text):

        line = line.strip()

        if len(line) > 30:

            if not line.startswith("-"):
                line = "- " + line

            bullets.append(line)

    return "\n".join(bullets)


def remove_duplicate_sentences(text):

    seen = set()
    output = []

    sentences = re.split(r"[.?!]", text)

    for s in sentences:

        s = s.strip()

        if len(s) < 20:
            continue

        key = s.lower()

        if key not in seen:
            seen.add(key)
            output.append(s)

    return ". ".join(output)


# ====================================================
# SAFE LLM CALL
# ====================================================

def safe_llm_call(llm, prompt, fallback="Summary generation failed."):

    try:

        result = llm.invoke(prompt)

        if isinstance(result, str):
            return result

        return str(result)

    except Exception as e:

        print("LLM ERROR:", e)

        return fallback


# ====================================================
# EMBEDDING CACHE
# ====================================================

@lru_cache(maxsize=512)
def cached_embedding(sentence: str):

    return embedder.encode(sentence)


# ====================================================
# EXTRACTIVE SENTENCE SELECTION
# ====================================================

def extract_key_sentences(text, top_k=3):

    sentences = [s.strip() for s in re.split(r"[.?!]", text) if len(s) > 40]

    if len(sentences) <= top_k:
        return sentences

    embeddings = np.array([cached_embedding(s) for s in sentences])

    centroid = np.mean(embeddings, axis=0)

    scores = cosine_similarity([centroid], embeddings)[0]

    ranked = sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [s for s, _ in ranked[:top_k]]


# ====================================================
# CHUNK SUMMARIZATION
# ====================================================

def summarize_chunk(llm, text):

    key_sentences = extract_key_sentences(text)

    compressed_text = ". ".join(key_sentences)

    prompt = CHUNK_SUMMARY_PROMPT.format(text=compressed_text)

    return safe_llm_call(llm, prompt)


# ====================================================
# PARALLEL CHUNK SUMMARIZATION
# ====================================================

def summarize_chunks_parallel(llm, chunks):

    with ThreadPoolExecutor(max_workers=4) as executor:

        summaries = list(
            executor.map(
                lambda c: summarize_chunk(llm, c.page_content),
                chunks
            )
        )

    return summaries


# ====================================================
# MERGE SUMMARIES
# ====================================================

def merge_summaries(llm, summaries):

    combined = "\n".join(summaries)

    combined = remove_duplicate_sentences(combined)

    prompt = MERGE_SUMMARY_PROMPT.format(text=combined)

    return safe_llm_call(llm, prompt)


# ====================================================
# BASE SUMMARIES (HIERARCHICAL)
# ====================================================

@lru_cache(maxsize=8)
def _cached_base_summaries(doc_hash: str, doc_paths: tuple):

    docs = load_documents(list(doc_paths))

    docs = [
        d for d in docs
        if d.metadata.get("chunk_type") != "image"
    ]

    llm = get_llm(mode="summary")

    splitter = _splitter()

    grouped: Dict[str, List[Document]] = {}

    for d in docs:
        grouped.setdefault(d.metadata["source"], []).append(d)

    base_summaries = {}

    for source, documents in grouped.items():

        chunks = splitter.split_documents(documents)

        # protect context size
        MAX_CHUNKS = 10
        chunks = chunks[:MAX_CHUNKS]

        chunk_summaries = summarize_chunks_parallel(llm, chunks)

        final_summary = merge_summaries(llm, chunk_summaries)

        base_summaries[source] = final_summary.strip()

    return base_summaries


def get_base_summaries(doc_paths: tuple):

    return _cached_base_summaries(
        _hash_paths(doc_paths),
        doc_paths
    )


# ====================================================
# DERIVED SUMMARIES
# ====================================================

def combined_from_base(base_summaries):

    llm = get_llm(mode="summary")

    combined = "\n\n".join(base_summaries.values())

    combined = combined[:2000]

    prompt = FINAL_BULLET_PROMPT.format(text=combined)

    raw = safe_llm_call(llm, prompt)

    return _format_bullets(raw)


def per_doc_from_base(base_summaries):

    llm = get_llm(mode="summary")

    out = {}

    for src, text in base_summaries.items():

        text = text[:1500]

        prompt = FINAL_BULLET_PROMPT.format(text=text)

        raw = safe_llm_call(llm, prompt)

        out[src] = _format_bullets(raw)

    return out


def topic_from_base(base_summaries):

    llm = get_llm(mode="summary")

    out = {}

    for src, text in base_summaries.items():

        text = text[:1500]

        prompt = TOPIC_EXTRACTION_PROMPT.format(text=text)

        raw = safe_llm_call(llm, prompt)

        out[src] = raw.strip()

    return out