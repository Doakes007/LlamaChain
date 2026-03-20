import fitz
import torch
from langchain.schema import Document
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# OCR import
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
# Generate caption (STRICT — NO ASSUMPTIONS)
# -------------------------------------------------
def generate_caption(image_path):

    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=40)

        caption = processor.decode(output[0], skip_special_tokens=True)

        # fallback if empty
        if not caption or len(caption.strip()) < 5:
            caption = "No clear caption generated for this figure."

        return caption.strip()

    except Exception:
        return "No caption available."


# -------------------------------------------------
# Extract images from PDF
# -------------------------------------------------
def extract_images_from_pdf(file_path):

    documents = []
    filename = os.path.basename(file_path)

    os.makedirs("extracted_images", exist_ok=True)

    pdf = fitz.open(file_path)

    for page_index in range(len(pdf)):

        page = pdf[page_index]
        images = page.get_images(full=True)

        # Extract nearby page text
        page_text = page.get_text("text")
        page_text = page_text[:500] if page_text else ""

        print(f"Page {page_index+1} → {len(images)} images found")

        for img_index, img in enumerate(images):

            try:
                xref = img[0]
                pix = fitz.Pixmap(pdf, xref)

                # -------------------------------------------------
                # FILTER SMALL IMAGES
                # -------------------------------------------------
                if pix.width < 200 or pix.height < 200:
                    continue

                image_name = f"{filename}_page{page_index+1}_img{img_index}.png"
                image_path = os.path.join("extracted_images", image_name)

                # Save image
                if pix.n < 5:
                    pix.save(image_path)
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(image_path)

                # -------------------------------------------------
                # Step 1 — Caption (NO FAKE LOGIC)
                # -------------------------------------------------
                caption = generate_caption(image_path)

                # -------------------------------------------------
                # Step 2 — OCR extraction
                # -------------------------------------------------
                ocr_text = extract_text_from_image(image_path)

                # -------------------------------------------------
                # Step 3 — CLEAN STRUCTURED CONTENT
                # -------------------------------------------------
                combined_content = f"""
FIGURE FROM DOCUMENT

Caption:
{caption}

OCR Text:
{ocr_text}

Nearby Document Text:
{page_text}
"""

                # -------------------------------------------------
                # Metadata
                # -------------------------------------------------
                metadata = {
                    "source": filename,
                    "page": page_index + 1,
                    "chunk_type": "image",
                    "figure": True,
                    "image_id": f"{page_index+1}_{img_index}",
                    "image_path": image_path,
                    "preview": caption[:200],
                    "ocr_text": ocr_text
                }

                documents.append(
                    Document(
                        page_content=combined_content.strip(),
                        metadata=metadata
                    )
                )

            except Exception as e:
                print("Image extraction failed:", e)
                continue

    return documents