import os
from collections import defaultdict
from functools import lru_cache

from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from llm import get_llm

# HYBRID RETRIEVER
from src.rag.hybrid_retriever import HybridRetriever

# CROSS ENCODER
from sentence_transformers import CrossEncoder


# =========================================================
# LOAD MODELS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)


# =========================================================
# CACHED CROSS-ENCODER RERANKER
# =========================================================
@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


cross_encoder = get_reranker()


# =========================================================
# QUERY EXPANSION
# =========================================================
def expand_query(query):

    expansions = {
        "dla": "document layout analysis layout detection",
        "ocr": "optical character recognition text extraction",
        "pipeline": "workflow architecture process pipeline",
        "diagram": "figure architecture diagram workflow",
        "architecture": "system architecture diagram workflow pipeline"
    }

    q = query.lower()

    for key, expansion in expansions.items():
        if key in q:
            query = query + " " + expansion

    return query


# =========================================================
# DETECT FIGURE QUERIES
# =========================================================
def is_figure_query(query):

    keywords = [
        "figure",
        "diagram",
        "workflow",
        "pipeline",
        "architecture",
        "flowchart",
        "visual",
        "illustration"
    ]

    q = query.lower()

    return any(k in q for k in keywords)


# =========================================================
# FAST CONTEXT COMPRESSION
# =========================================================
def compress_context(query, docs, max_sentences=3):

    query_embedding = embeddings.embed_query(query)

    compressed_chunks = []

    for doc in docs:

        text = doc.page_content

        sentences = [
            s.strip() for s in text.split(". ")
            if len(s.strip()) > 20
        ]

        if len(sentences) <= max_sentences:
            compressed_chunks.append(text[:400])
            continue

        try:

            # Batch embedding (FAST)
            sentence_embeddings = embeddings.embed_documents(sentences)

            scored = []

            for s, emb in zip(sentences, sentence_embeddings):

                score = cosine_similarity(
                    [query_embedding],
                    [emb]
                )[0][0]

                scored.append((score, s))

            scored.sort(reverse=True)

            best_sentences = [s for _, s in scored[:max_sentences]]

            compressed_chunks.append(". ".join(best_sentences))

        except Exception:
            compressed_chunks.append(text[:400])

    return "\n\n".join(compressed_chunks)


# =========================================================
# QA PROMPT
# =========================================================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a document QA assistant.

Answer the question using ONLY the provided context.

Rules:

1. Use only the information present in the context.
2. Combine information from multiple context snippets when needed.
3. Do NOT use outside knowledge.
4. Do NOT invent information not supported by the context.
5. If the context does not contain enough information, respond exactly:

Not specified in the provided documents.

6. If figures or diagrams appear in the context, explain them using:
   - figure description
   - OCR extracted text
   - surrounding document text.

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================================================
# BUILD RETRIEVAL CHAIN
# =========================================================
def build_retrieval_chain(vectorstore):

    llm = get_llm()
    retriever = HybridRetriever(vectorstore)

    return {
        "llm": llm,
        "retriever": retriever
    }


# =========================================================
# CROSS-ENCODER RERANKING
# =========================================================
def rerank_documents(query, docs, top_k=5):

    if not docs:
        return []

    pairs = [(query, d.page_content[:1000]) for d in docs]

    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(scores, docs))

    scored_docs.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored_docs[:top_k]]


# =========================================================
# MAIN QA FUNCTION
# =========================================================
def ask_question(chain, query):

    llm = chain["llm"]
    retriever = chain["retriever"]

    expanded_query = expand_query(query)

    figure_query = is_figure_query(query)

    # Retrieve documents
    docs = retriever.get_relevant_documents(expanded_query)

    # Rerank documents
    reranked_docs = rerank_documents(query, docs, top_k=5)

    # Image boost for figure queries
    if figure_query:

        image_docs = [
            d for d in docs
            if d.metadata.get("chunk_type") == "image"
        ]

        image_docs = rerank_documents(query, image_docs, top_k=2)

        for img in image_docs:
            if img not in reranked_docs:
                reranked_docs.append(img)

    # Context compression
    context = compress_context(query, reranked_docs)

    context = context[:1800]

    prompt = QA_PROMPT.format(
        context=context,
        question=query
    )

    answer = llm.invoke(prompt).strip()

    if not answer:
        answer = "Not specified in the provided documents."

    source_docs = reranked_docs

    if not source_docs:
        return answer

    # =====================================================
    # SOURCE FORMATTER
    # =====================================================
    page_map = defaultdict(set)

    for doc in source_docs:

        filename = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")

        if isinstance(page, int):
            page_map[filename].add(page)

    formatted_sources = []

    for filename, pages in page_map.items():

        if pages:

            min_p = min(pages)
            max_p = max(pages)

            if min_p == max_p:
                formatted_sources.append(
                    f"- **{filename}** (Page {min_p})"
                )
            else:
                formatted_sources.append(
                    f"- **{filename}** (Pages {min_p}-{max_p})"
                )
        else:
            formatted_sources.append(
                f"- **{filename}**"
            )

    # =====================================================
    # CONTEXT PREVIEW
    # =====================================================
    context_preview = []
    seen = set()

    for doc in source_docs:

        content = doc.page_content.strip()

        if content in seen:
            continue

        seen.add(content)

        filename = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunk_type = doc.metadata.get("chunk_type", "text")
        image_path = doc.metadata.get("image_path")

        preview = content.replace("\n", " ")
        preview = preview[:300] + "..." if len(preview) > 300 else preview

        if chunk_type == "image":

            context_preview.append(
                f"**{filename} (Page {page}) [image]**"
            )

            if image_path and os.path.exists(image_path):
                context_preview.append(
                    f"![image]({image_path})"
                )

            context_preview.append(f"> {preview}")

        else:

            context_preview.append(
                f"**{filename} (Page {page}) [{chunk_type}]**\n\n> {preview}"
            )

    return f"""{answer}

---
### 📌 Sources
{chr(10).join(formatted_sources)}

---
### 🔎 Retrieved Context
{chr(10).join(context_preview)}
"""