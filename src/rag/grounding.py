# ✅ FIX: Add missing import
import torch

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def is_answer_grounded_semantic(answer, context, threshold=0.7):
    """
    Semantic validation using embeddings.
    Tightened threshold to 0.7.
    """
    try:
        embedding_model = get_embedding_model()
        answer_embedding = embedding_model.encode(answer, convert_to_tensor=True)
        context_embedding = embedding_model.encode(context, convert_to_tensor=True)
        
        similarity = cosine_similarity(
            [answer_embedding.cpu().numpy()],
            [context_embedding.cpu().numpy()]
        )[0][0]
        
        if similarity > threshold:
            return True
        
    except Exception as e:
        print(f"Semantic grounding failed: {e}")
    
    return is_answer_grounded_token(answer, context)


def is_answer_grounded_token(answer, context):
    """Token-based grounding (fallback)"""
    stop_words = {"the","is","are","a","an","and","or","in","on","of","to"}
    answer_words = {w for w in answer.lower().split() if w not in stop_words}
    context_words = set(context.lower().split())

    if not answer_words:
        return False

    overlap = len(answer_words & context_words) / len(answer_words)

    if overlap >= 0.08:
        return True

    if len(answer_words) < 20 and overlap >= 0.05:
        return True

    return False


def is_answer_uncertain(answer):
    """
    Detect uncertainty keywords.
    """
    uncertain_words = [
        "may", "might", "could", "possibly", "probably",
        "not clearly", "unclear", "uncertain", "seems", "appears"
    ]
    return any(k in answer.lower() for k in uncertain_words)