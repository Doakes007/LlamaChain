import torch
import numpy as np
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from sentence_transformers import CrossEncoder
import os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def get_clip_model():
    """Load and cache CLIP model/preprocess"""
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return model.to(device).eval(), preprocess

@lru_cache(maxsize=1)
def get_reranker():
    """Load and cache Cross-Encoder"""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

_ce_score_cache = {}

def get_ce_score(query, content):
    """Cache Cross-Encoder scores to prevent redundant compute"""
    cache_key = f"{query}||{content[:100]}"
    if cache_key not in _ce_score_cache:
        pairs = [(query, content[:300])]
        _ce_score_cache[cache_key] = float(get_reranker().predict(pairs)[0])
    return _ce_score_cache[cache_key]

def clear_ce_cache():
    global _ce_score_cache
    _ce_score_cache = {}

def encode_query_clip(query):
    """Generate CLIP text embedding for the query"""
    model, _ = get_clip_model()
    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        text_features = model.encode_text(tokenizer([query]).to(device))
    return text_features.cpu().numpy()[0]

def multimodal_rerank(query, docs, top_k=5):
    """
    Reranks documents with explicit weighting for image chunks 
    to ensure visual evidence is prioritized.
    """
    final_scores = []
    query_clip_embedding = encode_query_clip(query)
    query_clip_embedding /= np.linalg.norm(query_clip_embedding)
    
    model, preprocess = get_clip_model()

    for doc in docs:
        content = doc.page_content
        ce_score = get_ce_score(query, content)
        
        # Base score from Cross-Encoder
        final_score = float(ce_score)
        
        # 🔥 FIX: EXPLICIT VISUAL WEIGHTING
        # This ensures images stay competitive against dense text chunks
        if doc.metadata.get("chunk_type") == "image":
            final_score += 0.15 
            
            image_path = doc.metadata.get("image_path")
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    img_t = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_f = model.encode_image(img_t)
                    img_emb = img_f.cpu().numpy()[0]
                    img_emb /= np.linalg.norm(img_emb)
                    
                    clip_sim = cosine_similarity([query_clip_embedding], [img_emb])[0][0]
                    # Hybrid Fusion: Balance visual similarity with semantic relevance
                    final_score = (clip_sim * 0.7) + (ce_score * 0.3) + 0.15
                except:
                    pass

        final_scores.append((final_score, doc))

    # Sort by boosted hybrid scores
    final_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Global duplicate filtering
    seen = set()
    results = []
    for _, d in final_scores:
        key = d.metadata.get("image_path") or d.page_content[:200]
        if key not in seen:
            results.append(d)
            seen.add(key)
    
    return results[:top_k]