
import os
from pathlib import Path
from typing import Dict, Any, List

import io 
import base64
from PIL import Image
from langchain_core.messages import HumanMessage # Used for multimodal messages


from src.rag.rag_chain import build_rag_chain
from src.core.llm import get_vision_llm 
from src.ingestion.extract import extract_from_files
from src.ingestion.to_documents import chunks_to_documents
from src.core.vectorstore import add_documents, delete_collection
from src.config import USE_GEMINI 

# --- Directory where uploaded images are stored for the Vision LLM ---
UPLOAD_DIR = Path("./temp_uploads") 
UPLOAD_DIR.mkdir(exist_ok=True)


def handle_multimodal_query(
    question: str, 
    uploaded_image_paths: List[str] = [],
    is_session_start: bool = False
) -> Dict[str, Any]:
    """
    Main function to run a multimodal RAG query and handle ingestion/cleanup.
    """
    
    # 1. ORCHESTRATION SETUP
    vision_llm = get_vision_llm()
    rag_chain = build_rag_chain()
    
    # --- INGESTION LOGIC ---
    if is_session_start and uploaded_image_paths: 
        print("[ORCHESTRATOR] Starting new session: clearing old index and ingesting new files.")
        delete_collection()
        
        try:
            chunks = extract_from_files(uploaded_image_paths) 
            lc_docs = chunks_to_documents(chunks) 
            add_documents(lc_docs)
            print(f"[ORCHESTRATOR] Ingestion complete. {len(lc_docs)} documents added.")
        except Exception as e:
            print(f"[ORCHESTRATOR ERROR] Error during document ingestion: {e}")
            return {"answer": f"Error during document ingestion: {e}", "source_documents": []}
        
       
        return {"answer": f"Successfully indexed {len(lc_docs)} documents.", "source_documents": []}
        
    # 2. VISION PRE-PROCESSING (Handling images in the user's query)
    if uploaded_image_paths and not is_session_start and USE_GEMINI:
        print("[ORCHESTRATOR] Analyzing user-uploaded images with Vision LLM...")
        
        vision_context_list = []
        for img_path in uploaded_image_paths:
            
            vision_prompt = f"""Analyze this user-uploaded image. Provide a detailed, 
            factual description that can be used as context to answer a user's question: '{question}'."""
            
            try:
                img = Image.open(img_path)
                
               
                buffered = io.BytesIO()
                img.save(buffered, format="PNG") 
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                
                response = vision_llm.invoke([
                    HumanMessage(content=[
                        {"type": "text", "text": vision_prompt},
                        
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ])
                ])
                
                vision_context = f"[USER IMAGE CONTEXT]: {response.content}"
                vision_context_list.append(vision_context)
                
            except Exception as e:
                print(f"[VISION ERROR] Failed to analyze user image {Path(img_path).name}: {e}")
                
        # Prepend the vision context to the user's question for the RAG chain
        if vision_context_list:
            vision_prefix = "\n\n".join(vision_context_list)
            question = f"{vision_prefix}\n\n[FINAL QUESTION]: {question}"
            print("[ORCHESTRATOR] Vision context added to question for RAG chain.")


    # 3. RUN RAG CHAIN
    print("[ORCHESTRATOR] Running main RAG chain...")
    try:
        
        result = rag_chain.invoke({"question": question}) 
        return result
    except Exception as e:
        print(f"[ORCHESTRATOR ERROR] Error during RAG chain execution: {e}")
        return {"answer": f"Error during RAG chain execution: {e}", "source_documents": []}