import os
import torch
import open_clip
from PIL import Image
from functools import lru_cache

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.hybrid_retriever import HybridRetriever
from sentence_transformers import CrossEncoder


# =========================================================
# CLIP MODEL (CPU)
# =========================================================
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to("cpu")
clip_model.eval()


# =========================================================
# RERANKER (CPU)
# =========================================================
@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

cross_encoder = get_reranker()


# =========================================================
# QUERY EXPANSION
# =========================================================
def expand_query(query):
    query_lower = query.lower()
    expansions = {
        "diagram":      "figure chart graph illustration visualization",
        "architecture": "system design structure framework components",
        "pipeline":     "workflow process methodology steps preprocessing",
        "table":        "data values rows columns dataset",
        "image":        "figure diagram visual illustration graphic",
        "workflow":     "pipeline process steps architecture",
        "ui":           "user interface frontend layout",
        "interface":    "user interface ui layout frontend",
        "figure":       "image diagram chart illustration",
        "preprocess":   "ocr layout text extraction cleaning pipeline",
        "retrieval":    "embedding search similarity query dpr clip",
        "compare":      "difference between both papers versus",
        "emotion":      "sentiment detection distilbert feeling mood",
        "voice":        "speech audio multilingual communication",
        "methodology":  "flowchart proposed system steps approach",
        "proposed":     "methodology flowchart system architecture figure",
        "clip":         "contrastive language image pretraining visual",
        "dpr":          "dense passage retrieval encoder embedding",
        "distilbert":   "bert knowledge distillation student teacher model",
        "mudoc":        "multimodal document conversational gpt4",
        "chatbot":      "nlp conversational agent reinforcement learning",
    }
    expanded = query
    for key, expansion in expansions.items():
        if key in query_lower:
            expanded += " " + expansion
    return expanded


# =========================================================
# FIGURE QUERY DETECTION
# =========================================================
def is_figure_query(query):
    keywords = [
        "figure", "diagram", "chart", "graph", "visual",
        "illustration", "architecture", "workflow", "pipeline",
        "structure", "layout", "framework", "interface", "ui",
        "image", "flowchart", "show", "depict", "demo", "look like"
    ]
    return any(k in query.lower() for k in keywords)


# =========================================================
# CLIP TEXT EMBEDDING
# =========================================================
def encode_query_clip(query):
    tokens = open_clip.tokenize([query])
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    return text_features.cpu().numpy()[0]


# =========================================================
# MULTIMODAL RERANK
# =========================================================
def multimodal_rerank(query, docs, top_k=6):
    if not docs:
        return []

    pairs = [(query, d.page_content[:800]) for d in docs]
    ce_scores = cross_encoder.predict(pairs)
    query_clip_embedding = encode_query_clip(query)

    final_scores = []
    for doc, ce_score in zip(docs, ce_scores):
        final_score = float(ce_score)

        if doc.metadata.get("chunk_type") == "image":
            try:
                image_path = doc.metadata.get("image_path", "")
                abs_path = os.path.abspath(image_path) if image_path else ""
                if abs_path and os.path.exists(abs_path):
                    image = Image.open(abs_path).convert("RGB")
                    img_tensor = clip_preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        img_features = clip_model.encode_image(img_tensor)
                    img_embedding = img_features.cpu().numpy()[0]
                    clip_score = float(cosine_similarity(
                        [query_clip_embedding], [img_embedding]
                    )[0][0])
                    final_score = (0.4 * float(ce_score)) + (0.6 * clip_score)
            except Exception as e:
                print(f"CLIP scoring failed: {e}")

            # Boost image chunks for figure queries
            if is_figure_query(query):
                final_score += 0.25
            else:
                final_score += 0.10

        final_scores.append((final_score, doc))

    final_scores.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    unique_docs = []
    for _, d in final_scores:
        key = d.page_content[:200]
        if key not in seen:
            unique_docs.append(d)
            seen.add(key)

    return unique_docs[:top_k]


# =========================================================
# STRUCTURED CONTEXT BUILDER
# =========================================================
def build_structured_context(docs):
    text_parts, image_parts, table_parts = [], [], []
    image_paths = []

    for d in docs:
        chunk_type = d.metadata.get("chunk_type")
        content = d.page_content.strip()

        if len(content) < 50:
            continue

        # Only skip "no readable text" for non-image chunks
        # Image chunks always have this in OCR section but still have valid captions
        if "no readable text" in content.lower() and chunk_type != "image":
            continue

        if chunk_type == "image":
            image_parts.append(content)
            path = d.metadata.get("image_path", "")
            if path:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    image_paths.append(abs_path)
        elif chunk_type == "table":
            table_parts.append(content)
        else:
            text_parts.append(content)

    context = ""
    if text_parts:
        context += "=== TEXT CONTEXT ===\n"
        context += "\n\n".join(text_parts[:8]) + "\n\n"
    if image_parts:
        context += "=== IMAGE CONTEXT ===\n"
        context += "\n\n".join(image_parts[:4]) + "\n\n"
    if table_parts:
        context += "=== TABLE CONTEXT ===\n"
        context += "\n\n".join(table_parts[:2])

    return context, image_paths


# =========================================================
# GROUNDING CHECK
# =========================================================
def is_answer_grounded(answer, context):
    stop_words = {
        "the","is","are","was","were","a","an","and","or","in","on","of","to",
        "for","with","as","by","this","that","it","its","be","at","from","have",
        "not","no","i","we","they","which","what","how","also","can","used","use"
    }
    answer_words = {w for w in answer.lower().split() if w not in stop_words}
    context_words = set(context.lower().split())
    if not answer_words:
        return False
    overlap = answer_words & context_words
    ratio = len(overlap) / len(answer_words)
    return ratio >= 0.20


# =========================================================
# SOURCE ANNOTATION
# =========================================================
def attach_sources(answer, docs):
    sources = set()
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        if src and page:
            sources.add(f"{src} (Page {page})")
    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sorted(sources))
    return answer


def compute_confidence(docs):
    if len(docs) >= 4:
        return "High"
    elif len(docs) >= 2:
        return "Medium"
    return "Low"


# =========================================================
# PROMPTS
# =========================================================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a strict document-grounded AI assistant.

RULES:
- Answer using ONLY information from the provided context below.
- Do NOT use any external knowledge or assumptions.
- When comparing two documents, explicitly state which document (R1.pdf or R2.pdf) each point comes from.
- If the context directly states something, quote or closely paraphrase it.
- If the context has no relevant information, say: "Not specified in the provided documents."
- Be concise and specific. Do not repeat yourself.

FORMAT:
Key Points:
- [bullet from context]

Explanation:
[2-3 sentences from context only]

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)

DIAGRAM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are analyzing a document that contains diagrams and figures.

RULES:
- Use ONLY the provided context.
- Describe components, flow, and relationships mentioned in the context.
- If a diagram is referenced, describe exactly what the context says about it.
- Do NOT invent components or steps not present in the context.

FORMAT:
Components:
- [component from context]

Flow / Steps:
1. [step from context]

Notes:
[any additional context-grounded explanation]

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)


# =========================================================
# BUILD CHAIN
# =========================================================
def build_retrieval_chain(vectorstore):
    return {"retriever": HybridRetriever(vectorstore=vectorstore)}


# =========================================================
# MAIN QA FUNCTION
# =========================================================
def ask_question(chain, query):
    from llm import get_llm

    retriever = chain["retriever"]
    expanded_query = expand_query(query)
    figure_query = is_figure_query(query)

    docs = retriever.invoke(expanded_query)
    reranked_docs = multimodal_rerank(query, docs, top_k=6)

    print("\n--- RETRIEVED DOCS ---")
    for d in reranked_docs:
        print(f"  {d.metadata.get('source')} p{d.metadata.get('page')} [{d.metadata.get('chunk_type')}]")

    context, image_paths = build_structured_context(reranked_docs)

    if not context or len(context.strip()) < 100:
        answer = "Not specified in the provided documents."
        answer = attach_sources(answer, reranked_docs)
        answer += f"\n\n*Confidence: {compute_confidence(reranked_docs)}*"
        if image_paths:
            for path in image_paths[:2]:
                clean_path = path.replace(os.sep, "/")
                answer += f"\n\n![image]({clean_path})"
        return answer

    # Truncate at sentence boundary
    if len(context) > 6000:
        truncated = context[:6000]
        last_period = truncated.rfind(".")
        context = truncated[:last_period + 1] if last_period > 4000 else truncated

    prompt_template = DIAGRAM_PROMPT if figure_query else QA_PROMPT
    prompt = prompt_template.format(context=context, question=query)

    answer = ""
    try:
        llm = get_llm(mode="summary")
        answer = llm.invoke(prompt)
        if hasattr(answer, "content"):
            answer = answer.content
        answer = str(answer).strip()
    except Exception as e:
        print(f"LLM (summary) failed: {e}")
        try:
            from llm import invalidate_llm_cache
            invalidate_llm_cache()
        except Exception:
            pass

    if not answer or len(answer) < 40:
        try:
            llm = get_llm(mode="rag")
            answer = llm.invoke(prompt)
            if hasattr(answer, "content"):
                answer = answer.content
            answer = str(answer).strip()
        except Exception as e:
            print(f"LLM (rag) fallback failed: {e}")
            try:
                from llm import invalidate_llm_cache
                invalidate_llm_cache()
            except Exception:
                pass

    if not answer:
        answer = "Not specified in the provided documents."

    if not is_answer_grounded(answer, context):
        answer = "Not specified in the provided documents."

    answer = attach_sources(answer, reranked_docs)
    answer += f"\n\n*Confidence: {compute_confidence(reranked_docs)}*"

    # Always append images after grounding check
    if image_paths:
        for path in image_paths[:2]:
            clean_path = path.replace(os.sep, "/")
            answer += f"\n\n![image]({clean_path})"

    return answer