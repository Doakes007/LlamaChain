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

        return unit

    # ----------------------------------------
    # Highest similarity
    # ----------------------------------------

    best_score = max(unit.matched_scores)

    unit.confidence = round(best_score, 3)

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