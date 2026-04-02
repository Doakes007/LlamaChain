import fitz
import torch
import numpy as np
from langchain.schema import Document
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

from src.core.ocr_extractor import extract_text_from_image


# -------------------------------------------------
# DEVICE SETUP (NEW)
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# LOAD BLIP MODEL (GPU ENABLED)
# -------------------------------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

model.eval()


# -------------------------------------------------
# GENERATE CAPTION (GPU SAFE)
# -------------------------------------------------
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)

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
        print(f"BLIP failed, falling back to CPU: {e}")

        # 🔁 SAFE FALLBACK TO CPU
        try:
            cpu_model = model.to("cpu")
            inputs = processor(image, return_tensors="pt")

            with torch.no_grad():
                output = cpu_model.generate(
                    **inputs,
                    max_new_tokens=60,
                    num_beams=4,
                )

            caption = processor.decode(output[0], skip_special_tokens=True)
            return caption.strip()

        except Exception as e2:
            return f"Caption generation failed: {e2}"


# -------------------------------------------------
# QUALITY FILTER
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
# EXTRACT BITMAP IMAGES
# -------------------------------------------------
def _extract_page_bitmaps(pdf, page_index, file_path, filename, documents, pages_with_images):
    page = pdf[page_index]
    image_list = page.get_images(full=True)

    if not image_list:
        print(f"Page {page_index+1} → 0 embedded bitmap(s)")
        return

    print(f"Page {page_index+1} → {len(image_list)} bitmap(s)")

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

            # Convert to PNG if needed
            if image_ext.lower() != "png":
                try:
                    Image.open(image_path).convert("RGB").save(image_path, "PNG")
                except Exception:
                    pass

            if not is_useful_image(image_path):
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
            print(f"Bitmap extraction failed p{page_index+1}: {e}")


# -------------------------------------------------
# PAGE RENDER (VECTOR FIGURES)
# -------------------------------------------------
def _extract_page_renders(pdf, file_path, filename, skip_pages):
    documents = []

    for page_index in range(len(pdf)):
        if page_index in skip_pages:
            continue

        page = pdf[page_index]
        page_text = page.get_text("text").strip()

        figure_keywords = [
            "figure 1", "figure 2", "figure 3",
            "fig. 1", "fig. 2", "fig 1"
        ]

        has_figure_keyword = any(k in page_text.lower() for k in figure_keywords)

        is_reference_page = page_text.lower().startswith("reference")
        is_text_heavy = len(page_text) > 6000

        if not has_figure_keyword or is_reference_page or is_text_heavy:
            continue

        try:
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            image_name = f"{filename}_page{page_index+1}_render.png"
            image_path = os.path.join("extracted_images", image_name)

            pix.save(image_path)

            if not is_useful_image(image_path, min_std=15.0):
                os.remove(image_path)
                continue

            caption = generate_caption(image_path)
            ocr_text = extract_text_from_image(image_path)

            if not ocr_text:
                ocr_text = "No readable text found."

            nearby_text = page_text[:600]

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

            documents.append(
                Document(page_content=combined_content.strip(), metadata=metadata)
            )

        except Exception as e:
            print(f"Render failed p{page_index+1}: {e}")

    return documents


# -------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------
def extract_images_from_pdf(file_path):
    os.makedirs("extracted_images", exist_ok=True)

    filename = os.path.splitext(os.path.basename(file_path))[0]

    documents = []
    pages_with_images = set()

    pdf = fitz.open(file_path)

    # Phase 1: Bitmaps
    for page_index in range(len(pdf)):
        _extract_page_bitmaps(pdf, page_index, file_path, filename, documents, pages_with_images)

    print(f"Bitmap extraction complete: {len(documents)} images")

    # Phase 2: Page renders
    render_docs = _extract_page_renders(pdf, file_path, filename, skip_pages=set())

    documents.extend(render_docs)

    print(f"Total images: {len(documents)}")

    pdf.close()

    return documents