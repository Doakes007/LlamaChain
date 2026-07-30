import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx

from src.ingestion.chunk_schema import Chunk


# =====================================================
# HELPERS
# =====================================================

def clean_filename(path: str) -> str:
    """
    Remove UUID prefix from uploaded filenames.

    Example:
        82f14dc3dd2ddcca_TestPaper.pdf
            ->
        TestPaper.pdf
    """
    filename = Path(path).name

    if "_" in filename:
        filename = filename.split("_", 1)[1]

    return filename


# =====================================================
# CHUNK CREATION
# =====================================================

def make_chunk(
    element,
    modality: str,
    path: str,
    file_type: str,
) -> Optional[Chunk]:

    meta = element.metadata or {}

    text = getattr(element, "text", "").strip()
    page = getattr(meta, "page_number", None)

    content = text

    extra_data: Dict[str, Any] = {
        "category": getattr(element, "category", None),
        "coordinates": getattr(meta, "coordinates", None),
    }

    # -------------------------------------------------
    # IMAGE
    # -------------------------------------------------

    if modality == "image":

        image_path = getattr(meta, "image_path", None)

        if image_path:

            content = image_path
            extra_data["image_path"] = image_path

            caption = text or getattr(meta, "alt_text", "")

            if caption:
                extra_data["caption"] = caption

        else:

            caption = (
                text
                or getattr(meta, "alt_text", "")
                or "Image placeholder"
            )

            content = f"[Image Caption Only]: {caption}"

    # -------------------------------------------------
    # TABLE
    # -------------------------------------------------

    elif modality == "table":

        table_html = getattr(meta, "text_as_html", None)
        table_text = text

        if table_html:

            content = table_html

            extra_data["table_html"] = table_html
            extra_data["caption"] = table_text

        else:

            content = f"[Table Summary]: {table_text}"

    # -------------------------------------------------
    # EMPTY CONTENT
    # -------------------------------------------------

    if not content or content.isspace():
        return None

    # -------------------------------------------------
    # CREATE CHUNK
    # -------------------------------------------------

    return Chunk(
        id=str(uuid.uuid4()),            # Internal chunk ID (keep this)
        content=content,
        modality=modality,
        file_name=clean_filename(path),  # Clean filename
        file_type=file_type,
        page_number=page,
        extra=extra_data,
    )


# =====================================================
# PDF EXTRACTION
# =====================================================

def extract_pdf(path: str) -> List[Chunk]:
    """
    Extract text, tables and images from a PDF.
    """

    print(f"[INFO] Partitioning PDF: {clean_filename(path)}")

    elements = partition_pdf(
        filename=path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        languages=["eng"],
        strategy="hi_res",
    )

    chunks: List[Chunk] = []

    for element in elements:

        category = getattr(element, "category", "")

        if category in ["NarrativeText", "Title", "ListItem"]:
            modality = "text"

        elif category == "Table":
            modality = "table"

        elif category == "Image":
            modality = "image"

        else:
            continue

        chunk = make_chunk(
            element,
            modality,
            path,
            "pdf",
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# =====================================================
# PPTX EXTRACTION
# =====================================================

def extract_pptx(path: str) -> List[Chunk]:

    print(f"[INFO] Partitioning PPTX: {clean_filename(path)}")

    elements = partition_pptx(
        path,
        extract_images_in_pptx=True,
    )

    chunks: List[Chunk] = []

    for element in elements:

        category = getattr(element, "category", "")

        modality = (
            "text"
            if category in ["NarrativeText", "Title", "ListItem"]
            else "table"
            if category == "Table"
            else "image"
            if category == "Image"
            else None
        )

        if modality is None:
            continue

        chunk = make_chunk(
            element,
            modality,
            path,
            "pptx",
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# =====================================================
# MAIN EXTRACTION
# =====================================================

def extract_from_files(file_paths: List[str]) -> List[Chunk]:

    chunks: List[Chunk] = []

    for path in file_paths:

        extension = Path(path).suffix.lower()

        if extension == ".pdf":

            chunks.extend(
                extract_pdf(path)
            )

        elif extension in [".ppt", ".pptx"]:

            chunks.extend(
                extract_pptx(path)
            )

        else:

            print(
                f"[WARN] Unsupported file type: {path}"
            )

    return chunks