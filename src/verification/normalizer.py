"""
normalizer.py

Verification Answer Normalizer

Converts raw LLM answers into a clean format before
sentence splitting.

Responsibilities
----------------
- Remove numbering (1., 2., ...)
- Remove markdown bullets
- Collapse blank lines
- Normalize whitespace
"""

from __future__ import annotations

import re


def normalize_answer(answer: str) -> str:
    """
    Normalize an LLM answer before sentence splitting.

    Parameters
    ----------
    answer : str

    Returns
    -------
    str
        Cleaned answer.
    """

    if not answer:
        return ""

    text = answer

    # ----------------------------------------------------
    # Normalize newlines
    # ----------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # ----------------------------------------------------
    # Remove numbered lists
    #
    # Example:
    # 1.
    # 2.
    # 15.
    # ----------------------------------------------------

    text = re.sub(
        r'^\s*\d+\.\s*$',
        '',
        text,
        flags=re.MULTILINE
    )

    # ----------------------------------------------------
    # Remove inline numbering
    #
    # Example:
    # 1. Input Layer
    #
    # becomes
    #
    # Input Layer
    # ----------------------------------------------------

    text = re.sub(
        r'(?m)^\s*\d+\.\s+',
        '',
        text
    )

    # ----------------------------------------------------
    # Remove markdown bullets
    #
    # - item
    # * item
    # • item
    # ----------------------------------------------------

    text = re.sub(
        r'(?m)^\s*[-*•]\s+',
        '',
        text
    )

    # ----------------------------------------------------
    # Collapse multiple blank lines
    # ----------------------------------------------------

    text = re.sub(
        r'\n{2,}',
        '\n',
        text
    )

    # ----------------------------------------------------
    # Normalize spaces
    # ----------------------------------------------------

    text = re.sub(
        r'[ \t]+',
        ' ',
        text
    )

    return text.strip()