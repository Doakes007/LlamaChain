"""
nli_model.py

Loads and caches Natural Language Inference (NLI) models.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------

NLI_MODELS = {
    "deberta": {
        "hf_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "display_name": "DeBERTa-v3",
    },
    "roberta": {
        "hf_name": "FacebookAI/roberta-large-mnli",
        "display_name": "RoBERTa-large-MNLI",
    },
    "modernbert": {
        "hf_name": "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        "display_name": "ModernBERT Large",
    },
}

DEFAULT_MODEL = "deberta"


# ---------------------------------------------------------
# Cached Loader
# ---------------------------------------------------------

@lru_cache(maxsize=None)
def get_nli_model(model_name: str = DEFAULT_MODEL):
    """
    Load and cache an NLI model.

    Parameters
    ----------
    model_name : str
        Key from NLI_MODELS.

    Returns
    -------
    tuple
        (tokenizer, model, device)
    """

    if model_name not in NLI_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(NLI_MODELS.keys())}"
        )

    model_config = NLI_MODELS[model_name]

    hf_model_name = model_config["hf_name"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"[NLI] Loading '{model_config['display_name']}' "
        f"on {device}..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name
    )

    model.to(device)
    model.eval()

    print("[NLI] Model loaded successfully.")

    return tokenizer, model, device