"""
pipeline.py

Evidence Verification Pipeline (Prototype V1)
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from src.verification.models import (
    NLILabel,
    VerificationStatus,
    VerificationUnit,
    VerificationResult,
)
from src.verification.splitter import split_answer
from src.verification.matcher import match_documents
from src.verification.verifier import verify_unit
from .citation import generate_citation


# =========================================================
# ABSENCE CLAIM DETECTION
# =========================================================

ABSENCE_PATTERNS = [

    # -------------------------------------------------
    # DOCUMENT / CONTEXT ACTIVE-VOICE ABSENCE
    #
    # Examples:
    #
    # "The provided context does not specify..."
    # "The document does not mention..."
    # "The documents do not provide..."
    #
    # These patterns are deliberately restricted to
    # context/document subjects so ordinary factual
    # negation is not treated as document-wide absence.
    # -------------------------------------------------

    r"\b(?:the\s+)?(?:provided\s+)?context\s+does\s+not\s+(?:provide|specify|mention|include|contain)\b",

    r"\b(?:the\s+)?(?:provided\s+)?document\s+does\s+not\s+(?:provide|specify|mention|include|contain)\b",

    r"\b(?:the\s+)?(?:provided\s+)?documents\s+do\s+not\s+(?:provide|specify|mention|include|contain)\b",


    # -------------------------------------------------
    # NOT INCLUDED
    # -------------------------------------------------

    r"\bis not included\b",
    r"\bare not included\b",


    # -------------------------------------------------
    # NOT PROVIDED
    # -------------------------------------------------

    r"\bis not provided\b",
    r"\bare not provided\b",
    r"\bnot provided\b",


    # -------------------------------------------------
    # NOT SPECIFIED
    # -------------------------------------------------

    r"\bis not specified\b",
    r"\bare not specified\b",
    r"\bnot specified\b",


    # -------------------------------------------------
    # NOT MENTIONED
    # -------------------------------------------------

    r"\bis not mentioned\b",
    r"\bare not mentioned\b",
    r"\bnot mentioned\b",


    # -------------------------------------------------
    # NOT AVAILABLE
    # -------------------------------------------------

    r"\bis not available\b",
    r"\bare not available\b",
    r"\bnot available\b",


    # -------------------------------------------------
    # EXPLICIT "NO ..." CLAIMS
    # -------------------------------------------------

    r"\bno information\b",
    r"\bno evidence\b",
    r"\bno figure\b",
    r"\bno diagram\b",
    r"\bno table\b",


    # -------------------------------------------------
    # CONTEXT / DOCUMENT COMPLETENESS CLAIMS
    #
    # Examples:
    #
    # "The context only provides information about..."
    # "The document only contains..."
    # "The provided documents only include..."
    #
    # These claims describe the completeness of the
    # available evidence and generally cannot be
    # established from one local chunk.
    # -------------------------------------------------

    r"\b(?:the\s+)?context\s+only\s+(?:provides?|contains?|includes?|mentions?|specifies?)\b",

    r"\b(?:the\s+)?document\s+only\s+(?:provides?|contains?|includes?|mentions?|specifies?)\b",

    r"\b(?:the\s+)?documents\s+only\s+(?:provide|contain|include|mention|specify)\b",

    r"\b(?:the\s+)?provided\s+document\s+only\s+(?:provides?|contains?|includes?|mentions?|specifies?)\b",

    r"\b(?:the\s+)?provided\s+documents\s+only\s+(?:provide|contain|include|mention|specify)\b",
]

def is_absence_claim(text: str) -> bool:
    """
    Detect claims that assert information or an object
    is absent.

    Examples:
        "The document does not contain a CNN diagram."
        "The learning rate is not specified."
        "No table is provided."

    Such claims normally require document-level evidence
    rather than a single locally retrieved chunk.
    """

    text = text.lower().strip()

    return any(
        re.search(pattern, text)
        for pattern in ABSENCE_PATTERNS
    )


# =========================================================
# EXPLICIT ABSENCE EVIDENCE
# =========================================================

def has_explicit_absence_evidence(
    unit: VerificationUnit,
) -> bool:
    """
    Determine whether any matched evidence explicitly
    expresses an absence statement.

    This does NOT determine whether the evidence entails
    the exact claim. NLI performs that step afterward.

    The purpose of this function is only to prevent local
    chunks that merely omit information from being used
    to establish document-wide absence.
    """

    for document in unit.matched_documents:

        evidence = (
            document.page_content
            or ""
        ).lower()

        if any(
            re.search(pattern, evidence)
            for pattern in ABSENCE_PATTERNS
        ):
            return True

    return False


# =========================================================
# MARK UNRESOLVED ABSENCE CLAIM
# =========================================================

def mark_absence_unknown(
    unit: VerificationUnit,
) -> None:
    """
    Mark a document-wide absence claim as UNKNOWN when
    retrieved evidence does not explicitly establish
    the claimed absence.
    """

    unit.nli_label = NLILabel.UNKNOWN

    unit.verification_status = (
        VerificationStatus.UNKNOWN
    )

    unit.confidence = 0.0

    unit.selected_evidence = None

    unit.citation = None


# =========================================================
# MAIN VERIFICATION PIPELINE
# =========================================================

def process_answer(
    answer: str,
    retrieved_docs: List[Document],
    top_k: int = 3,
) -> VerificationResult:

    units = split_answer(answer)

    for unit in units:

        # -------------------------------------------------
        # 1. Match claim against retrieved evidence
        # -------------------------------------------------

        match_documents(
            unit,
            retrieved_docs,
            top_k=top_k,
        )

        # -------------------------------------------------
        # 2. Special handling for absence claims
        # -------------------------------------------------

        if is_absence_claim(unit.text):

            if not has_explicit_absence_evidence(unit):

                print(
                    "[VERIFICATION] "
                    "Document-wide absence claim cannot "
                    "be established from local evidence:"
                )

                print(
                    f"  {unit.text}"
                )

                mark_absence_unknown(unit)

                continue

        # -------------------------------------------------
        # 3. Normal verification
        # -------------------------------------------------

        verify_unit(unit)

        # -------------------------------------------------
        # 4. Citation
        # -------------------------------------------------

        generate_citation(unit)

    return VerificationResult(
        answer=answer,
        verification_units=units,
    )