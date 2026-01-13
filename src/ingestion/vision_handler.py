
import os
from pathlib import Path
from typing import Optional
from PIL import Image
import io 
import base64 


from src.core.llm import get_vision_llm 
from src.config import USE_GEMINI 
from langchain_core.messages import HumanMessage 

def describe_image_with_vision(image_path: str, context: str = "") -> Optional[str]:
    
    # 1. INITIAL CHECKS 
    if not USE_GEMINI:
        print("[VISION] Skipping vision description: Gemini API not configured.")
        return None
    
    if not os.path.exists(image_path):
        print(f"[VISION ERROR] Image file not found at path: {image_path}")
        return None
    
    # 2. GET LLM INSTANCE
    try:
        vision_llm = get_vision_llm()
    except Exception as e:
        print(f"[VISION ERROR] Failed to initialize vision LLM: {e}")
        return None
    
    try:
        # 3. LOAD IMAGE AND CONSTRUCT MULTIMODAL MESSAGE
        img = Image.open(image_path)
        
        # 🚨 FIX: Convert PIL Image to Base64 String for LangChain ChatModel
        # This resolves the 'Unrecognized message part type: image' error.
        buffered = io.BytesIO()
        # Save PIL image to an in-memory byte stream (PNG is usually safe)
        img.save(buffered, format="PNG") 
        # Base64 encode the byte stream
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # --- VISION PROMPT ---
        prompt_text = f"""You are an expert document analyst. Describe this image in detail 
        for a Retrieval-Augmented Generation (RAG) system. The description 
        will be stored as text for vector search.
        
        Focus on:
        1. Main subject, objects, and overall layout.
        2. Explicit text/labels within the image (OCR).
        3. Data visualizations (charts, graphs, tables) and key trends shown.
        4. The image's relevance to its source document section (Context: {context}).

        Provide a clear, detailed, and factual description that summarizes the image's information content."""
        
        # LangChain HumanMessage structure for multimodal input
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                # 🚨 FIX: Use 'image_url' type with the Base64 data URL
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        )
        
        # 4. INVOKE THE LLM
        response = vision_llm.invoke([message])
        description = response.content.strip()
        
        print(f"[VISION] ✓ Used model: {vision_llm.model} for {Path(image_path).name}")
        print(f"[VISION] ✓ Description start: {description[:100]}...")
        return description
            
    except Exception as e:
        # Catches API key errors, generation errors, etc.
        print(f"[VISION ERROR] Failed to get description for {image_path}. Error: {e}")
        return None