import os
import io
import base64
from pathlib import Path
from typing import Optional

from PIL import Image
from langchain_core.messages import HumanMessage

from src.config import USE_GEMINI
from src.core.llm import get_vision_llm


# =====================================================
# DEBUG
# =====================================================

DEBUG = True


# =====================================================
# IMAGE DESCRIPTION
# =====================================================

def describe_image_with_vision(
    image_path: str,
    context: str = "",
) -> Optional[str]:
    """
    Generate a textual description of an image using the
    configured Vision LLM (Gemini).

    The generated description is stored in the vector database
    and later used during retrieval.
    """

    # -------------------------------------------------
    # INITIAL CHECKS
    # -------------------------------------------------

    if not USE_GEMINI:
        if DEBUG:
            print("[VISION] Gemini disabled. Skipping image description.")
        return None

    if not os.path.exists(image_path):
        if DEBUG:
            print(f"[VISION ERROR] Image not found: {image_path}")
        return None

    # -------------------------------------------------
    # LOAD VISION MODEL
    # -------------------------------------------------

    try:
        vision_llm = get_vision_llm()

    except Exception as e:

        if DEBUG:
            print(f"[VISION ERROR] Failed to initialize Vision LLM: {e}")

        return None

    # -------------------------------------------------
    # DESCRIBE IMAGE
    # -------------------------------------------------

    try:

        img = Image.open(image_path)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        base64_image = base64.b64encode(
            buffered.getvalue()
        ).decode("utf-8")

        prompt_text = f"""
You are an expert document analyst.

Describe this image so it can be indexed inside a
Retrieval-Augmented Generation (RAG) system.

Focus on:

1. Main objects and layout.
2. Visible text (OCR).
3. Charts, graphs, tables and trends.
4. Relationship to the surrounding document.

Document Context:
{context}

Return only a factual description.
"""

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ]
        )

        response = vision_llm.invoke([message])

        description = response.content.strip()

        if DEBUG:
            print(
                f"[VISION] Model: {vision_llm.model}"
            )

            print(
                f"[VISION] Image: {Path(image_path).name}"
            )

            print(
                f"[VISION] Description: "
                f"{description[:100]}..."
            )

        return description

    except Exception as e:

        if DEBUG:
            print(
                f"[VISION ERROR] "
                f"Failed to describe "
                f"{Path(image_path).name}: {e}"
            )

        return None