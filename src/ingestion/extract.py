

import uuid
from pathlib import Path
from typing import List, Optional, Any, Dict

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx

from src.ingestion.chunk_schema import Chunk



def make_chunk(element, modality: str, path: str, file_type: str) -> Optional[Chunk]:
    
    meta = element.metadata or {}
    text = getattr(element, "text", "").strip()
    page = getattr(meta, "page_number", None)
    
    # Initialize content and extra data
    content = text 
    extra_data: Dict[str, Any] = {
        "category": getattr(element, "category", None),
        "coordinates": getattr(meta, "coordinates", None),
    }

    # --- IMAGE HANDLING ---
    if modality == "image":
        image_path = getattr(meta, "image_path", None)
        
        if image_path:
            content = image_path 
            extra_data["image_path"] = image_path
            
            # Use the image caption or alt text as supplementary info in 'extra'
            caption = text or getattr(meta, "alt_text", "")
            if caption:
                extra_data["caption"] = caption
        else:
            # If no image path, we cannot process it later. Use the text caption as content.
            caption = text or getattr(meta, "alt_text", "") or "Image placeholder"
            content = f"[Image Caption Only]: {caption}"

    # --- TABLE HANDLING ---
    elif modality == "table":
        table_html = getattr(meta, "text_as_html", None)
        table_text = text 
        
        if table_html:
            content = table_html
            extra_data["table_html"] = table_html
            extra_data["caption"] = table_text 
        else:
            # Use the text/summary if structure is missing
            content = f"[Table Summary]: {table_text}"

    # --- TEXT HANDLING ---
    
    if not content or content.isspace():
        return None

    return Chunk(
        id=str(uuid.uuid4()),
        content=content,
        modality=modality,
        file_name=Path(path).name,
        file_type=file_type,
        page_number=page,
        extra=extra_data, 
    )


def extract_pdf(path: str) -> List[Chunk]:
    """Uses full Unstructured extraction: text, tables, and images from a PDF."""
    print(f"[INFO] Partitioning PDF: {Path(path).name}")
    elements = partition_pdf(
        filename=path,
        extract_images_in_pdf=True,
        infer_table_structure=True, 
        languages=["eng"],
        strategy="hi_res", 
    )

    chunks = []
    for e in elements:
        cat = getattr(e, "category", "")
        
        if cat in ["NarrativeText", "Title", "ListItem"]:
            modality = "text"
        elif cat == "Table":
            modality = "table"
        elif cat == "Image":
            modality = "image"
        
        else:
            continue

        chunk = make_chunk(e, modality, path, "pdf")
        if chunk:
            chunks.append(chunk)

    return chunks


def extract_pptx(path: str) -> List[Chunk]:
    
    print(f"[INFO] Partitioning PPTX: {Path(path).name}")
    elements = partition_pptx(path, extract_images_in_pptx=True)
    chunks = []

    for e in elements:
        cat = getattr(e, "category", "")
        modality = (
            "text" if cat in ["NarrativeText", "Title", "ListItem"]
            else "table" if cat == "Table"
            else "image" if cat == "Image"
            else None
        )
        if not modality:
            continue

        chunk = make_chunk(e, modality, path, "pptx")
        if chunk:
            chunks.append(chunk)

    return chunks


def extract_from_files(file_paths: List[str]) -> List[Chunk]:
   
    final: List[Chunk] = []

    for path in file_paths:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            final.extend(extract_pdf(path))
        elif ext in [".pptx", ".ppt"]:
            final.extend(extract_pptx(path))
        else:
            print(f"[WARN] Unsupported file type: {path}")

    return final