"""
matcher.py

Matches each VerificationUnit with the most relevant
retrieved documents.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from src.verification.models import VerificationUnit


# =========================================================
# TOKENIZATION
# =========================================================

TOKEN_PATTERN = re.compile(
    r"\b[a-zA-Z0-9]+(?:\.[0-9]+)?%?\b"
)


def _tokenize(text: str) -> set[str]:
    """
    Normalize text into lexical tokens for evidence matching.

    Examples:
        "ImageNet,"   -> "imagenet"
        "accuracy."   -> "accuracy"
        "(CNN)"       -> "cnn"
        "94.2%"       -> "94.2"
    """

    if not text:
        return set()

    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
    }


# =========================================================
# LEXICAL OVERLAP
# =========================================================

def _overlap_score(
    sentence: str,
    document: str,
) -> float:
    """
    Measure how much of the verification claim is represented
    in a candidate evidence document.

    Score:
        |claim_tokens ∩ evidence_tokens|
        --------------------------------
               |claim_tokens|

    Range:
        0.0 -> no lexical overlap
        1.0 -> every claim token occurs in the evidence
    """

    sentence_tokens = _tokenize(sentence)
    document_tokens = _tokenize(document)

    if not sentence_tokens:
        return 0.0

    overlap = sentence_tokens.intersection(
        document_tokens
    )

    return len(overlap) / len(sentence_tokens)


# =========================================================
# DOCUMENT MATCHING
# =========================================================

def match_documents(
    unit: VerificationUnit,
    retrieved_docs: List[Document],
    top_k: int = 3,
) -> VerificationUnit:
    """
    Select the most lexically relevant evidence candidates
    for a verification unit.

    The matcher does not decide whether evidence supports,
    contradicts, or is neutral toward the claim. That decision
    is performed later by the NLI verifier.
    """

    if not retrieved_docs:

        unit.matched_documents = []
        unit.matched_scores = []

        return unit

    scored_docs = []

    for doc in retrieved_docs:

        content = doc.page_content or ""

        score = _overlap_score(
            unit.text,
            content,
        )

        scored_docs.append(
            (
                score,
                doc,
            )
        )

    scored_docs.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top_docs = scored_docs[:top_k]

    unit.matched_documents = [
        doc
        for score, doc in top_docs
    ]

    unit.matched_scores = [
        round(score, 4)
        for score, doc in top_docs
    ]

    return unit