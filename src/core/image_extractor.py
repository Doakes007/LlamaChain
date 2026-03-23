import fitz
import torch
import numpy as np
from langchain.schema import Document
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

from src.core.ocr_extractor import extract_text_from_image


# -------------------------------------------------
# Load BLIP model once (CPU safe)
# -------------------------------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to("cpu")

model.eval()


# -------------------------------------------------
# Generate caption
# -------------------------------------------------
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=4,
            )

        caption = processor.decode(output[0], skip_special_tokens=True)

        if not caption or len(caption.strip()) < 5:
            return "No clear caption generated for this figure."

        return caption.strip()

    except Exception as e:
        return f"Caption generation failed: {e}"


# -------------------------------------------------
# Quality gate: reject blank/low-content images
# -------------------------------------------------
def is_useful_image(image_path, min_size=100, min_std=10.0):
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w < min_size or h < min_size:
            return False
        arr = np.array(img).astype(float)
        if arr.std() < min_std:
            return False
        return True
    except Exception:
        return False


# -------------------------------------------------
# Extract bitmap images from a single page
# -------------------------------------------------
def _extract_page_bitmaps(pdf, page_index, file_path, filename, documents, pages_with_images):
    page = pdf[page_index]
    image_list = page.get_images(full=True)

    if not image_list:
        print(f"Page {page_index+1} → 0 embedded bitmap(s) found")
        return

    print(f"Page {page_index+1} → {len(image_list)} embedded bitmap(s) found")

    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"{filename}_page{page_index+1}_img{img_index}.png"
            image_path = os.path.join("extracted_images", image_name)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # Convert non-PNG to PNG
            if image_ext.lower() != "png":
                try:
                    img_obj = Image.open(image_path).convert("RGB")
                    img_obj.save(image_path, "PNG")
                except Exception:
                    pass

            if not is_useful_image(image_path, min_size=100, min_std=10.0):
                print(f"  Skipping low-content image: {image_name}")
                os.remove(image_path)
                continue

            caption = generate_caption(image_path)
            ocr_text = extract_text_from_image(image_path)

            if not ocr_text or len(ocr_text.strip()) < 10:
                ocr_text = "No readable text found in this image."

            page_text = page.get_text("text").strip()
            nearby_text = page_text[:600] if page_text else ""

            combined_content = f"""FIGURE FROM DOCUMENT

Caption:
{caption}

OCR Text:
{ocr_text}

Nearby Document Text:
{nearby_text}"""

            metadata = {
                "source": os.path.basename(file_path),
                "page": page_index + 1,
                "chunk_type": "image",
                "image_path": image_path,
                "image_name": image_name,
            }

            documents.append(
                Document(page_content=combined_content.strip(), metadata=metadata)
            )
            pages_with_images.add(page_index)

        except Exception as e:
            print(f"  Bitmap extraction failed on page {page_index+1} img {img_index}: {e}")
            continue


# -------------------------------------------------
# Render entire page as image (fallback for vector figures)
# FIX: Always render ALL pages — not just those with figure keywords
# This ensures vector diagrams (like MuDoC Figure 1) are captured
# -------------------------------------------------
def _extract_page_renders(pdf, file_path, filename, skip_pages):
    documents = []

    # Pages to always render regardless of keywords (key diagram pages)
    force_render_pages = set()

    for page_index in range(len(pdf)):
        if page_index in skip_pages:
            continue

        page = pdf[page_index]
        page_text = page.get_text("text").strip()

        # Only render pages that explicitly reference numbered figures
        # Avoids rendering text-only pages that mention "pipeline" or "diagram" in passing
        figure_keywords = [
            "figure 1", "figure 2", "figure 3", "figure 4", "figure 5",
            "fig. 1", "fig. 2", "fig. 3", "fig. 4", "fig. 5",
            "fig 1", "fig 2", "fig 3", "fig 4", "fig 5",
        ]

        has_figure_keyword = any(k in page_text.lower() for k in figure_keywords)

        # Only render pages that:
        # 1. Have explicit figure caption references AND
        # 2. Are not too long (long pages = mostly text, not figures)
        # 3. Are not reference/bibliography pages
        is_reference_page = (
            page_text.lower().strip().startswith("reference") or
            "viii. references" in page_text.lower() or
            "references\n[1]" in page_text.lower()
        )
        is_text_heavy = len(page_text) > 6000

        if not has_figure_keyword or is_reference_page or is_text_heavy:
            print(f"Page {page_index+1} → skipping page render (no diagram content)")
            continue

        try:
            print(f"Page {page_index+1} → rendering full page as figure fallback")

            # 150 DPI render
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            image_name = f"{filename}_page{page_index+1}_render.png"
            image_path = os.path.join("extracted_images", image_name)
            pix.save(image_path)

            if not is_useful_image(image_path, min_std=15.0):
                print(f"  Skipping blank page render: {image_name}")
                os.remove(image_path)
                continue

            caption = generate_caption(image_path)
            ocr_text = extract_text_from_image(image_path)

            if not ocr_text or len(ocr_text.strip()) < 10:
                ocr_text = "No readable text found in the page render."

            nearby_text = page_text[:600] if page_text else ""

            combined_content = f"""FIGURE FROM DOCUMENT (Page Render)

Caption:
{caption}

OCR Text:
{ocr_text}

Nearby Document Text:
{nearby_text}"""

            metadata = {
                "source": os.path.basename(file_path),
                "page": page_index + 1,
                "chunk_type": "image",
                "image_path": image_path,
                "image_name": image_name,
            }

            documents.append(Document(page_content=combined_content.strip(), metadata=metadata))
            print(f"  Saved page render: {image_name}")

        except Exception as e:
            print(f"  Page render failed on page {page_index+1}: {e}")
            continue

    return documents


# -------------------------------------------------
# Main extraction entry point
# -------------------------------------------------
def extract_images_from_pdf(file_path):
    os.makedirs("extracted_images", exist_ok=True)

    filename = os.path.splitext(os.path.basename(file_path))[0]
    documents = []
    pages_with_images = set()

    pdf = fitz.open(file_path)

    # Phase 1: Extract embedded bitmaps
    for page_index in range(len(pdf)):
        _extract_page_bitmaps(pdf, page_index, file_path, filename, documents, pages_with_images)

    print(f"\nBitmap extraction complete: {len(documents)} useful images from {len(pages_with_images)} pages")

    # Phase 2: Render pages that don't already have bitmaps (catches vector diagrams)
    # Pass empty set so bitmap pages are also rendered if they have figure keywords
    # This ensures vector diagrams on pages with tiny/useless bitmaps are captured
    render_docs = _extract_page_renders(pdf, file_path, filename, skip_pages=set())
    print(f"Page render fallback complete: {len(render_docs)} additional images")

    documents.extend(render_docs)
    print(f"\nTotal images extracted from {os.path.basename(file_path)}: {len(documents)}")

    pdf.close()
    return documents