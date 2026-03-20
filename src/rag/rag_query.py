import os
import re
import torch
import open_clip
from PIL import Image
from functools import lru_cache

from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# HYBRID RETRIEVER
from src.rag.hybrid_retriever import HybridRetriever

# CROSS ENCODER
from sentence_transformers import CrossEncoder


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)


# =========================================================
# CLIP MODEL
# =========================================================
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

clip_model = clip_model.to("cpu")
clip_model.eval()


# =========================================================
# RERANKER
# =========================================================
@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )


cross_encoder = get_reranker()


# =========================================================
# QUERY EXPANSION
# =========================================================
def expand_query(query):

    query_lower = query.lower()

    expansions = {
        "diagram": "figure chart graph illustration visualization",
        "architecture": "system design architecture structure framework",
        "pipeline": "workflow process pipeline methodology steps",
        "table": "data table dataset values rows columns",
        "image": "figure diagram visual illustration graphic",
        "workflow": "pipeline process workflow steps architecture",
        "ui": "user interface interface diagram frontend layout",
        "interface": "user interface ui layout frontend"
    }

    expanded_query = query

    for key, expansion in expansions.items():
        if key in query_lower:
            expanded_query += " " + expansion

    return expanded_query


# =========================================================
# FIGURE DETECTION
# =========================================================
def is_figure_query(query):

    keywords = [
        "figure", "diagram", "chart", "graph", "visual",
        "illustration", "architecture", "workflow", "pipeline",
        "structure", "layout", "framework", "interface", "ui"
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
# MULTIMODAL RERANK (🔥 STEP 7 CORE)
# =========================================================
def multimodal_rerank(query, docs, top_k=5):

    if not docs:
        return []

    pairs = [(query, d.page_content[:800]) for d in docs]
    ce_scores = cross_encoder.predict(pairs)

    final_scores = []

    for doc, ce_score in zip(docs, ce_scores):

        final_score = ce_score

        # IMAGE BOOST
        if doc.metadata.get("chunk_type") == "image":

            try:
                image_path = doc.metadata.get("image_path")

                if image_path and os.path.exists(image_path):

                    image = Image.open(image_path).convert("RGB")
                    image = clip_preprocess(image).unsqueeze(0)

                    with torch.no_grad():
                        img_features = clip_model.encode_image(image)

                    img_embedding = img_features.cpu().numpy()[0]
                    query_embedding = encode_query_clip(query)

                    clip_score = cosine_similarity(
                        [query_embedding],
                        [img_embedding]
                    )[0][0]

                    final_score = (0.6 * ce_score) + (0.4 * clip_score)

            except Exception:
                pass

        final_scores.append((final_score, doc))

    final_scores.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in final_scores[:top_k]]


# =========================================================
# STRUCTURED CONTEXT
# =========================================================
def build_structured_context(docs):

    text_parts = []
    image_parts = []
    table_parts = []

    for d in docs:

        chunk_type = d.metadata.get("chunk_type")

        if chunk_type == "image":
            image_parts.append(d.page_content)

        elif chunk_type == "table":
            table_parts.append(d.page_content)

        else:
            text_parts.append(d.page_content)

    context = ""

    if image_parts:
        context += "=== IMAGE CONTEXT ===\n"
        context += "\n\n".join(image_parts[:3]) + "\n\n"

    if table_parts:
        context += "=== TABLE CONTEXT ===\n"
        context += "\n\n".join(table_parts[:2]) + "\n\n"

    if text_parts:
        context += "=== TEXT CONTEXT ===\n"
        context += "\n\n".join(text_parts[:5])

    return context


# =========================================================
# PROMPTS
# =========================================================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a STRICT document-grounded AI system.

Use ONLY the provided context.
Do NOT guess.
If missing → say: Not specified in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""
)

DIAGRAM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are analyzing a diagram.

Extract:

Components:
- ...

Flow:
1.
2.
3.

Explanation:
...

Use ONLY context. Do NOT assume.

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================================================
# BUILD CHAIN
# =========================================================
def build_retrieval_chain(vectorstore):
    return {"retriever": HybridRetriever(vectorstore)}


# =========================================================
# MAIN QA
# =========================================================
def ask_question(chain, query):

    from llm import get_llm

    retriever = chain["retriever"]

    expanded_query = expand_query(query)
    figure_query = is_figure_query(query)

    docs = retriever.get_relevant_documents(expanded_query)

    # 🔥 STEP 7 ACTIVE HERE
    reranked_docs = multimodal_rerank(query, docs, top_k=5)

    # -----------------------------------------------------
    # CONTEXT
    # -----------------------------------------------------
    context = build_structured_context(reranked_docs)

    if len(context) > 3000:
        context = context[:3000]

    # -----------------------------------------------------
    # PROMPT SWITCH
    # -----------------------------------------------------
    if figure_query:
        prompt = DIAGRAM_PROMPT.format(
            context=context,
            question=query + "\nExtract step-by-step flow."
        )
    else:
        prompt = QA_PROMPT.format(context=context, question=query)

    # -----------------------------------------------------
    # LLM EXECUTION
    # -----------------------------------------------------
    answer = ""

    try:
        llm = get_llm(mode="summary")
        answer = llm.invoke(prompt).strip()
    except Exception:
        pass

    if not answer or len(answer) < 40:
        try:
            llm = get_llm(mode="rag")
            answer = llm.invoke(prompt).strip()
        except Exception:
            pass

    if not answer:
        answer = "Not specified in the provided documents."

    return answer