# ✅ FIX: Add missing import
import torch

import numpy as np
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from sentence_transformers import CrossEncoder
import os
from PIL import Image

from .intent import classify_query_intent, detect_query_domain

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(device)
clip_model.eval()

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

_ce_score_cache = {}

def get_ce_score(query, content):
    """Cache CE scores"""
    cache_key = f"{query}||{content[:100]}"
    if cache_key not in _ce_score_cache:
        pairs = [(query, content[:300])]
        _ce_score_cache[cache_key] = float(get_reranker().predict(pairs)[0])
    return _ce_score_cache[cache_key]

def clear_ce_cache():
    global _ce_score_cache
    _ce_score_cache.clear()


def encode_query_clip(query):
    """Encode query with CLIP and normalize"""
    tokens = open_clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    embedding = text_features.cpu().numpy()[0]
    
    # ✅ GAP 1: Normalize embeddings
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def is_image_relevant(query, doc):
    return True  # Let CLIP handle everything


# ✅ GAP 3: Add visual query detection
def is_visual_query(query):
    """Detect if query is about visual/structural content"""
    visual_words = [
        "architecture", "diagram", "flow", "structure", "model", 
        "pipeline", "system", "design", "layout", "framework",
        "visualization", "working", "process", "how"
    ]
    return any(w in query.lower() for w in visual_words)


def multimodal_rerank(query, docs, top_k=5):
    """Intent-aware reranking with CLIP support"""
    if not docs:
        return []

    docs = docs[:15] # Early limit to reduce CE calls

    pairs = [(query, d.page_content[:800]) for d in docs]
    ce_scores = get_reranker().predict(pairs)

    domain = detect_query_domain(query)
    intent = classify_query_intent(query)
    is_fig_query = "figure" in query.lower() or "diagram" in query.lower()
    is_vis_query = is_visual_query(query)

    # Always compute CLIP
    query_clip_embedding = encode_query_clip(query)

    final_scores = []

    for doc, ce_score in zip(docs, ce_scores):
        final_score = float(ce_score)
        content = doc.page_content.lower()

        # Intent-aware boosting
        if intent == "performance":
            if any(k in content for k in ["underperform", "worst", "lowest", "best", "highest"]):
                final_score += 0.5
            if any(k in content for k in ["confusion", "misclassified"]):
                final_score -= 0.2

        elif intent == "factual":
            if any(k in content for k in ["total", "number", "count"]):
                final_score += 0.4

        elif intent == "explanation":
            if len(content) > 200:
                final_score += 0.3

        elif intent == "comparison":
            if any(k in content for k in ["compare", "versus", "differ"]):
                final_score += 0.3

        # General boosting
        if any(k in content for k in ["accuracy", "result", "metric"]):
            final_score += 0.1

        # Source-aware boosting
        source = doc.metadata.get("source", "").lower()
        if domain == "nlp" and "nlp" in source:
            final_score += 0.2
        elif domain == "vision" and ("cnn" in source or "image" in source):
            final_score += 0.2


        # IMAGE HANDLING (FULLY UPGRADED)
        if doc.metadata.get("chunk_type") == "image":
            try:
                image_path = doc.metadata.get("image_path", "")
                abs_path = os.path.abspath(image_path)

                if os.path.exists(abs_path):
                    if not is_image_relevant(query, doc):
                        # ✅ GAP 4: Weak image gets slight bias, not killed
                        final_score = float(ce_score) + 0.1
                    else:
                        # Use CLIP for multimodal scoring
                        try:
                            image = Image.open(abs_path).convert("RGB")
                            img_tensor = clip_preprocess(image).unsqueeze(0).to(device)

                            with torch.no_grad():
                                img_features = clip_model.encode_image(img_tensor)

                            img_embedding = img_features.cpu().numpy()[0]
                            
                            # ✅ GAP 1: Normalize image embedding
                            img_embedding = img_embedding / np.linalg.norm(img_embedding)
                            
                            clip_score = cosine_similarity(
                                [query_clip_embedding], [img_embedding]
                            )[0][0]
                            
                            # SIMPLE + STABLE (NO HYBRID)
                            final_score = (clip_score * 0.8) + (float(ce_score) * 0.2)

                        except Exception as e:
                            print("CLIP error:", e)
                            # ✅ GAP 4: Fallback keeps image viable
                            final_score = float(ce_score) + 0.1

            except Exception as e:
                print("Image processing error:", e)

        final_scores.append((final_score, doc))

    final_scores.sort(key=lambda x: x[0], reverse=True)

    # ✅ GAP 5: Better duplicate filtering
    seen = set()
    unique_docs = []

    for _, d in final_scores:
        # Use image path as key if available, otherwise content
        key = d.metadata.get("image_path", "") or d.page_content[:200]
        
        if key not in seen:
            unique_docs.append(d)
            seen.add(key)

    return unique_docs[:top_k]