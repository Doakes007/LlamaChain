"""
models.py

Core data models used by the Evidence Verification Layer (EVL).

Every VerificationUnit starts with only an ID and text.
As it flows through the verification pipeline, each stage
enriches the same object with additional information.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from langchain_core.documents import Document


class VerificationStatus(Enum):
    """
    Final verification status assigned by the verifier.
    """

    UNKNOWN = "UNKNOWN"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"

class NLILabel(Enum):
    """
    Raw prediction returned by the NLI model.
    """

    UNKNOWN = "UNKNOWN"

    ENTAILMENT = "ENTAILMENT"

    NEUTRAL = "NEUTRAL"

    CONTRADICTION = "CONTRADICTION"

@dataclass
class VerificationUnit:
    """
    Represents a single verification unit (currently one sentence).

    The same object flows through the complete verification pipeline.
    Each module enriches this object with new information.
    """

    # ==========================================================
    # Created by splitter.py
    # ==========================================================

    id: int
    text: str

    # ==========================================================
    # Added by matcher.py
    # ==========================================================

    matched_documents: List[Document] = field(default_factory=list)

    matched_scores: List[float] = field(default_factory=list)

    # Added by NLI verifier

    nli_label: NLILabel = NLILabel.UNKNOWN

    # Final user-facing status

    verification_status: VerificationStatus = VerificationStatus.UNKNOWN

    confidence: Optional[float] = None

    selected_evidence: Optional[Document] = None

    # ==========================================================
    # Added by citation.py
    # ==========================================================

    citation: Optional[str] = None

@dataclass
class VerificationResult:
    """
    Output of the complete verification pipeline.
    """

    answer: str

    verification_units: List[VerificationUnit]