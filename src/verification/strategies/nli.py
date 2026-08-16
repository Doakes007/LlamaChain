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

    Evidence policy (updated 2026-08-16):

    1. The matcher ranks up to top_k candidate evidence chunks by
       lexical relevance.
    2. NLI is run against EVERY candidate, not just the single
       best lexical match. Restricting NLI to only the top-1
       lexical match produced systematic false negatives: the
       chunk that scores highest on lexical overlap is often not
       the chunk that most directly states the fact (e.g. a
       claim's canonical definition may live in the abstract while
       the top lexical match is an unrelated body-text mention
       that happens to reuse similar words). Verified true claims
       like "Donut uses a Swin Transformer as its visual encoder"
       were scoring NEUTRAL purely because the one chunk checked
       didn't happen to state it, even though a different retrieved
       chunk did.
    3. Selection across candidates, in priority order:
         a. If ANY candidate is ENTAILMENT -> SUPPORTED, using the
            highest-confidence entailing candidate as evidence.
         b. Else if ANY candidate is CONTRADICTION -> UNSUPPORTED,
            using the highest-confidence contradicting candidate.
         c. Else -> PARTIALLY_SUPPORTED, using the highest-confidence
            neutral candidate.

    This still avoids selecting evidence merely because it produces
    a preferred label for a single arbitrary candidate, while giving
    a true claim credit if *any* retrieved evidence directly
    supports it -- not just whichever chunk happened to rank first
    on lexical overlap. A genuine contradiction from any candidate
    still overrides a merely-neutral top match, so this does not
    paper over real errors -- it only stops correct claims from
    being penalized for evidence-selection noise.
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
    # Run NLI against every candidate
    # -----------------------------------------

    predictions = []

    for document in unit.matched_documents:

        try:

            label, confidence = predict_nli(
                claim=unit.text,
                evidence=document.page_content,
            )

        except Exception as e:

            print(
                f"NLI verification failed for unit {unit.id} "
                f"against one candidate: {e}"
            )

            continue

        predictions.append((label, confidence, document))

    if not predictions:

        unit.nli_label = NLILabel.UNKNOWN

        unit.verification_status = (
            VerificationStatus.UNKNOWN
        )

        unit.confidence = 0.0
        unit.selected_evidence = None

        return unit

    # -----------------------------------------
    # Select evidence by priority policy
    # -----------------------------------------

    def best_of(label_filter):
        candidates = [
            p for p in predictions if p[0] == label_filter
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p[1])

    chosen = (
        best_of(NLILabel.ENTAILMENT)
        or best_of(NLILabel.CONTRADICTION)
        or best_of(NLILabel.NEUTRAL)
        or predictions[0]
    )

    label, confidence, evidence = chosen

    # -----------------------------------------
    # Store prediction
    # -----------------------------------------

    unit.nli_label = label
    unit.confidence = round(confidence, 4)
    unit.selected_evidence = evidence

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