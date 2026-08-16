"""
verifier.py

Evidence Verification Module

Version 1

Assigns a verification status to each verification unit
based on the highest evidence matching score.
"""

from __future__ import annotations

from src.verification.models import (
    VerificationUnit,
    VerificationStatus,
)


# ---------------------------------------------------------
# Thresholds
# ---------------------------------------------------------

SUPPORTED_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.45


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def verify_with_threshold(unit: VerificationUnit,) -> VerificationUnit:
    """
    Verify a single VerificationUnit.

    Parameters
    ----------
    unit : VerificationUnit

    Returns
    -------
    VerificationUnit
    """

    # ----------------------------------------
    # No retrieved evidence
    # ----------------------------------------

    if not unit.matched_scores:

        unit.verification_status = VerificationStatus.UNSUPPORTED
        unit.confidence = 0.0
        unit.selected_evidence = None

        return unit

    # ----------------------------------------
    # Highest similarity
    # ----------------------------------------

    best_index = unit.matched_scores.index(max(unit.matched_scores))
    best_score = unit.matched_scores[best_index]

    unit.confidence = round(best_score, 3)

    # matched_documents / matched_scores are parallel lists produced by
    # matcher.py (same index = same candidate). Recorded here purely so
    # this strategy also produces a citation, on equal footing with the
    # NLI strategy, for side-by-side comparison (Section 6.6 ablation).
    # Does not affect the SUPPORTED/PARTIAL/UNSUPPORTED decision below.
    unit.selected_evidence = unit.matched_documents[best_index]

    # ----------------------------------------
    # Decision
    # ----------------------------------------

    if best_score >= SUPPORTED_THRESHOLD:

        unit.verification_status = VerificationStatus.SUPPORTED

    elif best_score >= PARTIAL_THRESHOLD:

        unit.verification_status = (
            VerificationStatus.PARTIALLY_SUPPORTED
        )

    else:

        unit.verification_status = (
            VerificationStatus.UNSUPPORTED
        )

    return unit