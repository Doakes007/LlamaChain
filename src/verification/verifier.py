"""
verifier.py

Evidence Verification Dispatcher

This module selects which verification strategy
should be used.

Current Strategies
------------------
- threshold
- nli
- llm (future)
"""

from __future__ import annotations

from src.verification.models import VerificationUnit

from src.verification.strategies.threshold import (
    verify_with_threshold,
)

from src.verification.strategies.nli import (
    verify_with_nli,
)


# ---------------------------------------------------------
# Default Verification Strategy
# ---------------------------------------------------------

DEFAULT_METHOD = "nli"


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def verify_unit(
    unit: VerificationUnit,
    method: str = DEFAULT_METHOD,
) -> VerificationUnit:
    """
    Verify one VerificationUnit using the selected strategy.

    Parameters
    ----------
    unit : VerificationUnit
        Verification unit to verify.

    method : str
        Verification strategy to use.

    Returns
    -------
    VerificationUnit
        Updated verification unit.
    """

    if method == "nli":
        return verify_with_nli(unit)

    elif method == "threshold":
        return verify_with_threshold(unit)

    raise ValueError(
        f"Unknown verification method: {method}"
    )