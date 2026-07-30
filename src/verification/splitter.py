"""
splitter.py

Verification Unit Extraction (VUE)

Converts an LLM answer into individual verification units
that can later be matched against retrieved evidence.

Supports:
- Normal prose sentences
- Markdown table rows
"""

from __future__ import annotations

import re
from typing import List

import spacy

from src.verification.models import VerificationUnit
from src.verification.normalizer import normalize_answer


# ---------------------------------------------------------
# Load spaCy once
# ---------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not installed.\n"
        "Run:\n"
        "python -m spacy download en_core_web_sm"
    )


# ---------------------------------------------------------
# Filtering Patterns
# ---------------------------------------------------------

NUMBER_ONLY_PATTERN = re.compile(
    r"^\d+\.?$"
)


REFERENCE_PATTERN = re.compile(
    r"^\s*[\(\[]?\s*"
    r"(reference|references|source|sources|citation|citations)"
    r"\s*[:\-]",
    re.IGNORECASE,
)


DOCUMENT_METADATA_PATTERN = re.compile(
    r"^\s*document\s*:\s*.+$",
    re.IGNORECASE,
)


PAGE_METADATA_PATTERN = re.compile(
    r"^\s*page\s*:\s*\d+\s*$",
    re.IGNORECASE,
)


# Markdown separator cells such as:
# ---
# ----
# :---
# ---:
# :---:
MARKDOWN_SEPARATOR_CELL = re.compile(
    r"^:?-{3,}:?$"
)


# ---------------------------------------------------------
# Filtering Helpers
# ---------------------------------------------------------

def should_keep_sentence(text: str) -> bool:
    """
    Decide whether a sentence is meaningful enough
    to become a verification unit.

    Verification units should contain factual claims,
    not formatting, numbering, headings, or citation
    metadata.
    """

    text = text.strip()

    if not text:
        return False

    # -----------------------------------------
    # Ignore citation/reference metadata
    # -----------------------------------------

    if (
        REFERENCE_PATTERN.match(text)
        or DOCUMENT_METADATA_PATTERN.match(text)
        or PAGE_METADATA_PATTERN.match(text)
    ):
        return False

    # -----------------------------------------
    # Ignore numbering
    # -----------------------------------------

    if NUMBER_ONLY_PATTERN.fullmatch(text):
        return False

    # -----------------------------------------
    # Ignore standalone bullets
    # -----------------------------------------

    if text in {"-", "*", "•"}:
        return False

    # -----------------------------------------
    # Ignore short headings ending with :
    # -----------------------------------------

    if text.endswith(":"):

        words = text[:-1].split()

        if len(words) <= 3:
            return False

    # -----------------------------------------
    # Ignore tiny fragments
    # -----------------------------------------

    if len(text) < 10:
        return False

    return True


# ---------------------------------------------------------
# Markdown Table Helpers
# ---------------------------------------------------------

def _split_table_row(line: str) -> List[str]:
    """
    Split a Markdown table row into cells.

    Examples:

        Model | Parameters | Accuracy

        | Model | Parameters | Accuracy |

    Both produce:

        ["Model", "Parameters", "Accuracy"]
    """

    line = line.strip()

    if line.startswith("|"):
        line = line[1:]

    if line.endswith("|"):
        line = line[:-1]

    return [
        cell.strip()
        for cell in line.split("|")
    ]


def _is_markdown_separator_row(line: str) -> bool:
    """
    Detect Markdown table separator rows.

    Examples:

        --- | --- | ---
        :--- | ---: | :---:
    """

    if "|" not in line:
        return False

    cells = _split_table_row(line)

    if not cells:
        return False

    return all(
        MARKDOWN_SEPARATOR_CELL.fullmatch(cell)
        is not None
        for cell in cells
    )


def _table_row_to_claim(
    headers: List[str],
    values: List[str],
) -> str | None:
    """
    Convert one Markdown table data row into a
    natural-language verification claim.

    Example:

        Headers:
            Model | Parameters | Val Accuracy | Inference Time

        Row:
            MobileNetV2 | 3.4M | 95.1% | 8.6 ms

        Output:
            MobileNetV2 has Parameters: 3.4M,
            Val Accuracy: 95.1%, and
            Inference Time: 8.6 ms.
    """

    if not headers or not values:
        return None

    if len(headers) != len(values):
        return None

    cleaned_headers = [
        header.strip()
        for header in headers
    ]

    cleaned_values = [
        value.strip()
        for value in values
    ]

    if not all(cleaned_headers):
        return None

    if not all(cleaned_values):
        return None

    # First column is treated as the row entity.
    entity = cleaned_values[0]

    attributes = []

    for header, value in zip(
        cleaned_headers[1:],
        cleaned_values[1:],
    ):

        attributes.append(
            f"{header}: {value}"
        )

    if not attributes:
        return None

    if len(attributes) == 1:

        details = attributes[0]

    elif len(attributes) == 2:

        details = (
            f"{attributes[0]} and "
            f"{attributes[1]}"
        )

    else:

        details = (
            ", ".join(attributes[:-1])
            + f", and {attributes[-1]}"
        )

    return f"{entity} has {details}."


def _extract_markdown_table_claims(
    text: str,
) -> tuple[List[str], str]:
    """
    Extract Markdown table rows from an answer and
    convert each data row into one verification claim.

    Returns:

        (
            table_claims,
            remaining_non_table_text
        )

    The remaining text is later processed normally
    using spaCy.
    """

    lines = text.splitlines()

    table_claims: List[str] = []
    remaining_lines: List[str] = []

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # -------------------------------------------------
        # Detect:
        #
        # header
        # separator
        #
        # Example:
        #
        # Model | Parameters | Accuracy
        # --- | --- | ---
        # -------------------------------------------------

        if (
            line
            and "|" in line
            and i + 1 < len(lines)
            and _is_markdown_separator_row(
                lines[i + 1].strip()
            )
        ):

            headers = _split_table_row(line)

            i += 2

            # ---------------------------------------------
            # Consume table data rows
            # ---------------------------------------------

            while i < len(lines):

                row_line = lines[i].strip()

                if not row_line:
                    break

                if "|" not in row_line:
                    break

                if _is_markdown_separator_row(row_line):
                    i += 1
                    continue

                values = _split_table_row(row_line)

                claim = _table_row_to_claim(
                    headers,
                    values,
                )

                if claim:
                    table_claims.append(claim)

                i += 1

            continue

        # -------------------------------------------------
        # Normal non-table line
        # -------------------------------------------------

        remaining_lines.append(lines[i])

        i += 1

    remaining_text = "\n".join(
        remaining_lines
    ).strip()

    return (
        table_claims,
        remaining_text,
    )


# ---------------------------------------------------------
# Prose Splitting
# ---------------------------------------------------------

def _extract_prose_claims(
    text: str,
) -> List[str]:
    """
    Split ordinary prose using spaCy.
    """

    if not text.strip():
        return []

    doc = nlp(text)

    claims: List[str] = []

    for sent in doc.sents:

        sentence = sent.text.strip()

        if not should_keep_sentence(sentence):
            continue

        claims.append(sentence)

    return claims


# ---------------------------------------------------------
# Main Splitter
# ---------------------------------------------------------

def split_answer(
    answer: str,
) -> List[VerificationUnit]:
    """
    Convert an LLM answer into verification units.

    Processing order:

    1. Normalize answer.
    2. Detect Markdown tables.
    3. Convert each table data row into one factual claim.
    4. Remove the table from prose processing.
    5. Split remaining prose using spaCy.
    6. Create VerificationUnit objects.

    This prevents entire Markdown tables from becoming
    one malformed NLI verification unit.
    """

    if not answer.strip():
        return []

    clean_answer = normalize_answer(answer)

    # -------------------------------------------------
    # Extract structured table claims first
    # -------------------------------------------------

    (
        table_claims,
        remaining_text,
    ) = _extract_markdown_table_claims(
        clean_answer
    )

    # -------------------------------------------------
    # Process remaining prose normally
    # -------------------------------------------------

    prose_claims = _extract_prose_claims(
        remaining_text
    )

    # -------------------------------------------------
    # Preserve answer order approximately.
    #
    # For the current RAG output, tables are normally
    # returned as standalone answers. Table claims are
    # therefore placed first, followed by prose claims.
    # -------------------------------------------------

    claims = (
        table_claims
        + prose_claims
    )

    # -------------------------------------------------
    # Build Verification Units
    # -------------------------------------------------

    units: List[VerificationUnit] = []

    unit_id = 1

    for claim in claims:

        claim = claim.strip()

        if not should_keep_sentence(claim):
            continue

        units.append(
            VerificationUnit(
                id=unit_id,
                text=claim,
            )
        )

        unit_id += 1

    return units