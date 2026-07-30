import os

from langchain.schema import Document

from .models import VerificationUnit


# =====================================================
# Filename Cleaner
# =====================================================

def clean_filename(filename: str) -> str:
    """
    Remove UUID prefix from filenames.

    Example:
        82f14dc3dd2ddcca_TestPaper.pdf
            ->
        TestPaper.pdf
    """

    filename = os.path.basename(filename)

    if "_" in filename:
        filename = filename.split("_", 1)[1]

    return filename


# =====================================================
# Citation Formatter
# =====================================================

def format_citation(document: Document) -> str:
    """
    Convert a LangChain Document into a human-readable citation.

    Example:
        TestPaper.pdf (Page 5)
    """

    source = clean_filename(
        document.metadata.get(
            "source",
            "Unknown Document"
        )
    )

    page = document.metadata.get("page")

    if page is None:
        return source

    return f"{source} (Page {page})"


# =====================================================
# Generate Citation
# =====================================================

def generate_citation(unit: VerificationUnit) -> None:
    """
    Generate a citation for a verification unit.

    The citation is generated ONLY from the selected evidence.

    Updates:
        unit.citation
    """

    if unit.selected_evidence is None:
        unit.citation = None
        return

    unit.citation = format_citation(unit.selected_evidence)


# =====================================================
# Batch Helper
# =====================================================

def generate_citations(units: list[VerificationUnit]) -> None:
    """
    Generate citations for multiple verification units.
    """

    for unit in units:
        generate_citation(unit)