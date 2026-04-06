import os
import numpy as np
from functools import lru_cache
from collections import defaultdict
from langchain.prompts import PromptTemplate

from .rerank import multimodal_rerank, get_ce_score, clear_ce_cache, encode_query_clip
from .grounding import is_answer_grounded_semantic, is_answer_uncertain
from src.rag.hybrid_retriever import HybridRetriever

# =====================================================
# CLIP LOADER (CACHED)
# =====================================================
@lru_cache(maxsize=1)
def get_clip():
    import torch
    import open_clip

    # FIX: Corrected torch.callback to torch.cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    return model.to(device).eval(), preprocess, device


# =====================================================
# QUERY TYPE DETECTION
# =====================================================
def detect_query_type(query: str):
    q = query.lower()

    if any(w in q for w in ["compare", "difference", "vs", "similarities"]):
        return "comparison"
    elif any(w in q for w in ["why", "how", "reason", "analyze"]):
        return "analytical"
    elif any(w in q for w in ["diagram", "architecture", "flow", "structure"]):
        return "visual"
    else:
        return "factual"


# =====================================================
# QUERY EXPANSION
# =====================================================
def expand_query(query):
    query_lower = query.lower()

    expansions = {
        "diagram": "figure chart graph illustration visualization",
        "architecture": "system design structure framework components",
        "pipeline": "workflow process methodology steps preprocessing",
        "table": "data values rows columns dataset",
    }

    expanded = query
    for key, val in expansions.items():
        if key in query_lower:
            expanded += " " + val

    return expanded


# =====================================================
# IMAGE RETRIEVAL (CLIP)
# =====================================================
def retrieve_images_by_clip(vectorstore, query, top_k=3):
    try:
        from PIL import Image
        from sklearn.metrics.pairwise import cosine_similarity
        import torch

        model, preprocess, device = get_clip()
        res = vectorstore.get(include=["documents", "metadatas"])

        image_candidates = []

        for content, metadata in zip(
            res.get("documents", []), res.get("metadatas", [])
        ):
            if metadata.get("chunk_type") != "image":
                continue

            if "render" in metadata.get("image_path", "").lower():
                continue

            image_candidates.append({"content": content, "metadata": metadata})

        if not image_candidates:
            return []

        q_emb = encode_query_clip(query)
        q_emb /= np.linalg.norm(q_emb)

        scored = []

        for img in image_candidates:
            path = os.path.abspath(img["metadata"].get("image_path", ""))

            if not os.path.exists(path):
                continue

            try:
                pixel = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model.encode_image(pixel).cpu().numpy()[0]

                emb /= np.linalg.norm(emb)

                score = cosine_similarity([q_emb], [emb])[0][0]
                scored.append((score, img))

            except:
                continue

        if not scored:
            return image_candidates[:top_k]

        scored.sort(key=lambda x: x[0], reverse=True)

        return [d for _, d in scored[:top_k]]

    except Exception:
        return []


# =====================================================
# CONTEXT BUILDER
# =====================================================
def build_interleaved_context(query, docs):
    image_paths = []
    context_parts = []

    # Sort primarily by source to keep document information grouped for the LLM
    docs = sorted(
        docs,
        key=lambda d: (d.metadata.get("source", ""), d.metadata.get("page", 0)),
    )

    for d in docs:
        content = d.page_content.strip()
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "N/A")

        if d.metadata.get("chunk_type") == "image":
            context_parts.append(f"[DOCUMENT: {source} | Page: {page} | IMAGE]\n{content}")

            # Isolation Fix: Only text hints from the SAME doc
            text_chunks = [
                t.page_content for t in docs 
                if t.metadata.get("chunk_type") != "image" 
                and t.metadata.get("source") == source
            ]

            if text_chunks:
                best = max(text_chunks, key=lambda x: get_ce_score(query, x))[:250]
                context_parts.append(f"[TEXT HINT FROM SAME DOC: {source}]\n{best}")

            path = d.metadata.get("image_path", "")
            if os.path.exists(path):
                image_paths.append(os.path.abspath(path).replace("\\", "/"))

        else:
            context_parts.append(f"[DOCUMENT: {source} | Page: {page} | TEXT]\n{content}")

    return "\n\n".join(context_parts), image_paths


# =====================================================
# PROMPTS
# =====================================================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question using the provided context. 
Be concise, accurate, and refer to specific documents where possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""",
)

DIAGRAM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Explain the diagram or architecture clearly. 
Use both image descriptions and text hints from the specific document provided.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""",
)

# ✅ HIGH-IMPACT TECHNICAL COMPARISON PROMPT
COMPARISON_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are comparing multiple technical documents.

STRICT RULES:
- Do NOT give generic answers.
- Use ONLY information present in the context.
- Extract specific technical mechanisms, not high-level summaries.
- If details are missing, explicitly say "not specified in document".

OUTPUT FORMAT:

Document: <name>
- Architecture Details:
- Data Processing Pipeline (step-by-step):
- Model Mechanism (how it actually works internally):
- Performance Metrics (exact values):

Document: <name>
- Architecture Details:
- Data Processing Pipeline (step-by-step):
- Model Mechanism (how it actually works internally):
- Performance Metrics (exact values):

Deep Technical Comparison:
- Layer-level Differences (e.g., convolution vs attention):
- Information Flow Differences:
- Representation Learning Differences:
- Computational Trade-offs:

Final Insight:
- When to use which approach (based on evidence only)

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)


# =====================================================
# MAIN PIPELINE
# =====================================================
def ask_question(chain, query):
    from src.rag.summarizer import get_llm
    from langchain.schema import Document

    retriever = chain["retriever"]
    vectorstore = chain.get("vectorstore")

    clear_ce_cache()

    # -------- RETRIEVE --------
    docs = retriever.invoke(expand_query(query))
    images = retrieve_images_by_clip(vectorstore, query)

    image_docs = [
        Document(page_content=i["content"], metadata={**i["metadata"], "chunk_type": "image"})
        for i in images
    ]

    all_docs = docs + image_docs

    # -------- RERANK --------
    reranked = multimodal_rerank(query, all_docs, top_k=10)

    # -------- QUERY TYPE --------
    qtype = detect_query_type(query)

    imgs = [d for d in reranked if d.metadata.get("chunk_type") == "image"]
    txts = [d for d in reranked if d.metadata.get("chunk_type") != "image"]

    # -------- BALANCED DOCUMENT SAMPLING (RANK PRESERVING) --------
    if qtype == "comparison":
        grouped = defaultdict(list)
        # Preserve rerank order by iterating through the ranked list
        for doc in reranked:
            grouped[doc.metadata.get("source", "")].append(doc)
        
        final_docs = []
        # Re-interleave while taking top 3 from each group
        for docs_per_source in grouped.values():
            final_docs.extend(docs_per_source[:3])
            
        final_docs = final_docs[:8]
    elif qtype == "analytical":
        final_docs = reranked[:5]
    elif qtype == "visual" and imgs:  
        final_docs = imgs[:3] + txts[:2]
    else:
        final_docs = txts[:3] + imgs[:1]

    # -------- CONTEXT --------
    context, image_paths = build_interleaved_context(query, final_docs)

    # -------- PROMPT SELECTION --------
    if qtype == "comparison":
        prompt = COMPARISON_PROMPT
    elif qtype == "visual" and image_paths:
        prompt = DIAGRAM_PROMPT
    else:
        prompt = QA_PROMPT

    # -------- LLM EXECUTION --------
    llm = get_llm("rag")
    res = llm.invoke(prompt.format(context=context, question=query))
    answer = str(res.content if hasattr(res, "content") else res).strip()

    # -------- GROUNDING & CONFIDENCE --------
    if not is_answer_grounded_semantic(answer, context):
        answer = "Insufficient information in the provided documents."
        confidence = "Low"
    elif "not mentioned" in answer.lower() or "not provided" in answer.lower():
        confidence = "High (explicitly not present in docs)"
    elif len(final_docs) < 3:
        confidence = "Medium"
    elif is_answer_uncertain(answer):
        confidence = "Medium"
    else:
        confidence = "High"

    sources = sorted({
        f"{d.metadata.get('source')} (P{d.metadata.get('page')})"
        for d in final_docs if d.metadata.get("source")
    })

    if sources:
        answer += "\n\n**Sources:** " + ", ".join(sources)

    answer += f"\n\n*Confidence: {confidence}*"

    return answer, image_paths


# =====================================================
# HELPERS
# =====================================================
def build_retrieval_chain(vectorstore):
    return {
        "retriever": HybridRetriever(vectorstore=vectorstore),
        "vectorstore": vectorstore,
    }

def get_indexed_documents(vectorstore):
    try:
        res = vectorstore.get(include=["metadatas"])
        return sorted({
            m.get("source")
            for m in res.get("metadatas", [])
            if m.get("source")
        })
    except:
        return []