"""
nli.py

Natural Language Inference Strategy

Version 2
"""

from __future__ import annotations

import torch

from src.verification.models import (
    NLILabel,
    VerificationStatus,
    VerificationUnit,
)
from src.verification.strategies.nli_model import (
    DEFAULT_MODEL,
    get_nli_model,
)


def predict_nli(
    claim: str,
    evidence: str,
    model_name: str = DEFAULT_MODEL,
):
    """
    Predict the Natural Language Inference (NLI)
    relationship between a claim and an evidence
    sentence.

    Parameters
    ----------
    claim : str
        Statement to verify.

    evidence : str
        Retrieved evidence sentence/chunk.

    model_name : str, optional
        Model key from NLI_MODELS.

    Returns
    -------
    tuple[NLILabel, float]
        (Predicted NLI label, confidence score)
    """

    tokenizer, model, device = get_nli_model(model_name)

    # -----------------------------------------
    # Tokenize claim + evidence
    # -----------------------------------------

    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # -----------------------------------------
    # Model Inference
    # -----------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    # -----------------------------------------
    # Read label dynamically from model config
    # -----------------------------------------

    hf_label = model.config.id2label[
        prediction.item()
    ].lower()

    if hf_label == "entailment":
        label = NLILabel.ENTAILMENT

    elif hf_label == "neutral":
        label = NLILabel.NEUTRAL

    elif hf_label == "contradiction":
        label = NLILabel.CONTRADICTION

    else:
        raise ValueError(
            f"Unknown NLI label returned by model: {hf_label}"
        )

    return (
        label,
        round(confidence.item(), 4),
    )


# =========================================================
# Verification Strategy
# =========================================================

def verify_with_nli(
    unit: VerificationUnit,
) -> VerificationUnit:
    """
    Verify a VerificationUnit using Natural Language
    Inference (NLI).

    Evidence policy:

    1. The matcher ranks candidate evidence by relevance.
    2. The highest-ranked matched document is used as the
       primary verification evidence.
    3. NLI determines whether that evidence entails,
       contradicts, or is neutral toward the claim.

    This avoids selecting evidence merely because it
    produces a preferred NLI label.
    """

    # -----------------------------------------
    # No retrieved evidence
    # -----------------------------------------

    if not unit.matched_documents:

        unit.verification_status = (
            VerificationStatus.UNKNOWN
        )

        unit.nli_label = NLILabel.UNKNOWN
        unit.confidence = 0.0
        unit.selected_evidence = None

        return unit

    # -----------------------------------------
    # Select strongest matched evidence
    # -----------------------------------------
    #
    # match_documents() already sorts documents
    # from highest to lowest lexical relevance.
    # Therefore index 0 is the strongest candidate.
    # -----------------------------------------

    best_document = unit.matched_documents[0]

    # -----------------------------------------
    # Run NLI
    # -----------------------------------------

    try:

        label, confidence = predict_nli(
            claim=unit.text,
            evidence=best_document.page_content,
        )

    except Exception as e:

        print(
            f"NLI verification failed "
            f"for unit {unit.id}: {e}"
        )

        unit.nli_label = NLILabel.UNKNOWN

        unit.verification_status = (
            VerificationStatus.UNKNOWN
        )

        unit.confidence = 0.0
        unit.selected_evidence = None

        return unit

    # -----------------------------------------
    # Store prediction
    # -----------------------------------------

    unit.nli_label = label
    unit.confidence = round(confidence, 4)
    unit.selected_evidence = best_document

    # -----------------------------------------
    # Convert NLI label to verification status
    # -----------------------------------------

    if label == NLILabel.ENTAILMENT:

        unit.verification_status = (
            VerificationStatus.SUPPORTED
        )

    elif label == NLILabel.NEUTRAL:

        unit.verification_status = (
            VerificationStatus.PARTIALLY_SUPPORTED
        )

    elif label == NLILabel.CONTRADICTION:

        unit.verification_status = (
            VerificationStatus.UNSUPPORTED
        )

    else:

        unit.verification_status = (
            VerificationStatus.UNKNOWN
        )

    return unit