import os
import re
import numpy as np
from collections import defaultdict
from functools import lru_cache
from langchain.prompts import PromptTemplate

from .intent import classify_query_intent, detect_query_domain, detect_query_source
from .rerank import multimodal_rerank, get_ce_score, clear_ce_cache, get_reranker, encode_query_clip
from .grounding import is_answer_grounded_semantic, is_answer_uncertain
from .prioritization import prioritize_conclusion_chunks
from src.rag.hybrid_retriever import HybridRetriever


# ✅ FIX: CACHED CLIP LOADER (GLOBAL)
@lru_cache(maxsize=1)
def get_clip():
    """Load CLIP once and cache it"""
    import torch
    import open_clip
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    return clip_model, clip_preprocess, device


def expand_query(query):
    """Expand query with relevant synonyms"""
    query_lower = query.lower()
    expansions = {
        "diagram": "figure chart graph illustration visualization",
        "architecture": "system design structure framework components",
        "pipeline": "workflow process methodology steps preprocessing",
        "table": "data values rows columns dataset",
        "explain": "describe detail understanding",
        "compare": "difference versus similar",
    }
    expanded = query
    for key, expansion in expansions.items():
        if key in query_lower:
            expanded += " " + expansion
    return expanded


def is_visual_query(query):
    """Detect if query is about visual/structural content"""
    visual_words = [
        "architecture", "diagram", "flow", "pipeline",
        "structure", "model", "system design", "working",
        "how", "show", "display", "visualize", "layout",
        "framework", "process", "step", "workflow"
    ]
    return any(w in query.lower() for w in visual_words)


def is_figure_query(query):
    """Detect figure queries - simplified, no blocking"""
    q = query.lower()
    
    figure_keywords = [
        "figure", "diagram", "architecture", "flowchart",
        "pipeline", "visual", "show", "display", "illustration",
        "structure", "layout", "flow", "image", "chart", "graph"
    ]
    
    return any(k in q for k in figure_keywords)


def is_show_figure_query(query):
    return "show" in query.lower() and is_figure_query(query)


# ✅ DIAGNOSTIC: Check vectorstore contents
def diagnose_vectorstore(vectorstore):
    """Debug: show what's actually in vectorstore"""
    res = vectorstore.get(include=["documents", "metadatas"])
    
    print("\n" + "="*60)
    print("🔍 VECTORSTORE DIAGNOSIS")
    print("="*60)
    print(f"Total documents: {len(res.get('documents', []))}")
    
    image_count = 0
    text_count = 0
    
    for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
        chunk_type = meta.get("chunk_type", "unknown")
        source = meta.get("source", "unknown")
        
        if chunk_type == "image":
            image_count += 1
            image_path = meta.get("image_path", "NO PATH")
            exists = "✅" if os.path.exists(os.path.abspath(image_path)) else "❌"
            print(f"  IMAGE #{image_count} | Source: {source} | Path exists: {exists}")
            print(f"    Path: {image_path}")
        else:
            text_count += 1
    
    print(f"\n📊 Summary: {image_count} images, {text_count} text chunks")
    print("="*60 + "\n")
    
    return image_count > 0


# ✅ FIX: CLIP-BASED IMAGE RETRIEVAL (WITH DIAGNOSTICS)
def retrieve_images_by_clip(vectorstore, query, top_k=3):
    """
    Retrieve images using CLIP embeddings.
    This is more reliable than text-based retrieval for images.
    """
    try:
        import torch
        from PIL import Image
        from sklearn.metrics.pairwise import cosine_similarity
        
        print(f"\n🔎 Starting CLIP image retrieval for query: '{query}'")
        
        # ✅ DIAGNOSTIC: Check vectorstore
        has_images = diagnose_vectorstore(vectorstore)
        
        if not has_images:
            print("⚠️  No images found in vectorstore!")
            return []
        
        # Get cached CLIP
        print("📦 Loading CLIP model...")
        clip_model, clip_preprocess, device = get_clip()
        print(f"✅ CLIP loaded on device: {device}")
        
        # Get all documents
        res = vectorstore.get(include=["documents", "metadatas"])
        
        image_candidates = []
        
        for content, metadata in zip(res.get("documents", []), res.get("metadatas", [])):
            if metadata.get("chunk_type") == "image":
                image_candidates.append({
                    "content": content,
                    "metadata": metadata
                })
        
        print(f"📊 Found {len(image_candidates)} image candidates")
        
        if not image_candidates:
            print("⚠️  No image candidates after filtering!")
            return []
        
        # Encode query with CLIP
        print("🧠 Encoding query with CLIP...")
        query_embedding = encode_query_clip(query)
        print(f"✅ Query embedding shape: {query_embedding.shape}")
        
        # Score each image
        image_scores = []
        
        for idx, img_dict in enumerate(image_candidates):
            image_path = img_dict["metadata"].get("image_path", "")
            abs_path = os.path.abspath(image_path)
            
            print(f"\n  [{idx+1}/{len(image_candidates)}] Processing: {os.path.basename(image_path)}")
            
            if not os.path.exists(abs_path):
                print(f"    ❌ Path doesn't exist: {abs_path}")
                continue
            
            try:
                print(f"    📂 Opening image...")
                image = Image.open(abs_path).convert("RGB")
                print(f"    ✅ Image loaded: {image.size}")
                
                img_tensor = clip_preprocess(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    img_features = clip_model.encode_image(img_tensor)
                
                img_embedding = img_features.cpu().numpy()[0]
                img_embedding = img_embedding / np.linalg.norm(img_embedding)
                
                clip_score = cosine_similarity(
                    [query_embedding], [img_embedding]
                )[0][0]
                
                print(f"    🎯 CLIP score: {clip_score:.4f}")
                image_scores.append((clip_score, img_dict))
                
            except Exception as e:
                print(f"    ❌ Error processing: {e}")
                continue
        
        print(f"\n📈 Successfully scored {len(image_scores)} images")
        
        if not image_scores:
            print("⚠️  No images were successfully scored!")
            return []
        
        # Sort by score and return top-k
        image_scores.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\n📌 Top {min(top_k, len(image_scores))} images by CLIP score:")
        for i, (score, doc) in enumerate(image_scores[:top_k]):
            print(f"  {i+1}. Score: {score:.4f} | Source: {doc['metadata'].get('source')}")
        
        result = [doc for _, doc in image_scores[:top_k]]
        print(f"\n✅ Returning {len(result)} images\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Image retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ✅ FIX 2: INTERLEAVED CONTEXT BUILDING
def build_interleaved_context(docs):
    """
    Build context with text and images interleaved.
    This improves LLM reasoning about multimodal content.
    """
    image_paths = []
    context_parts = []
    
    # Sort docs by metadata to interleave properly
    sorted_docs = sorted(docs, key=lambda d: (
        d.metadata.get("chunk_type") != "image",  # images first
        -len(d.page_content)  # then by length (longer first)
    ))
    
    for d in sorted_docs:
        chunk_type = d.metadata.get("chunk_type")
        content = d.page_content.strip()
        
        if len(content) < 50:
            continue
        
        if "no readable text" in content.lower() and chunk_type != "image":
            continue
        
        if chunk_type == "image":
            source = d.metadata.get('source', 'unknown')
            page = d.metadata.get('page', 'N/A')
            
            context_parts.append(f"[Image from {source} page {page}]")
            context_parts.append(content)
            
            path = d.metadata.get("image_path", "")
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                image_paths.append(abs_path.replace("\\", "/"))
        
        elif chunk_type == "table":
            context_parts.append(f"[Table]\n{content}")
        
        else:
            context_parts.append(f"[Text]\n{content}")
    
    # Interleaved, not separated
    context = "\n\n".join(context_parts)
    
    return context, image_paths


QA_PROMPT = PromptTemplate(
    input_variables=["context","question"],
    template="""Answer ONLY using provided context.

Rules:
- Prefer explicit conclusions over details
- Combine multiple pieces if needed
- Avoid examples unless asked
- Say "Not specified" if unclear
- Reference both text and images when relevant

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)

DIAGRAM_PROMPT = PromptTemplate(
    input_variables=["context","question"],
    template="""Describe the diagram from context ONLY.

Rules:
- Only describe what is EXPLICITLY shown
- If unclear or not clearly relevant, say "Not clear from diagram"
- DO NOT guess or infer system structure
- Be specific about components and relationships
- Do NOT hallucinate details
- Reference text context if it clarifies the image

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)

# ✅ FIX 3: STRUCTURED OUTPUT FOR COMPARISON
COMPARISON_PROMPT = PromptTemplate(
    input_variables=["context","documents","question"],
    template="""Compare the following documents: {documents}

Rules:
- Include EVERY document
- DO NOT hallucinate
- Be precise and factual
- Use ONLY information from context

STRICT FORMAT:

**Overview:**
[1-2 sentences summary]

**Similarities:**
- [point 1]
- [point 2]
- [point 3 if exists]

**Differences:**
- [point 1]
- [point 2]
- [point 3 if exists]

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)


def llm_rerank(query, docs, top_k=3):
    """Conditional LLM reranking (only for complex queries)"""
    from llm import get_llm
    
    intent = classify_query_intent(query)
    
    # Only rerank for specific intents
    if intent not in ["performance", "comparison"]:
        return docs[:top_k]
    
    if len(docs) <= top_k:
        return docs
    
    options_text = ""
    for i, doc in enumerate(docs[:10]):
        content = doc.page_content[:300].replace("\n", " ")
        options_text += f"{i+1}. {content}\n\n"
    
    prompt = f"""Select the BEST chunk to answer: "{query}"

Rules:
- Prefer direct conclusions
- Ignore examples
- Pick 1-2 numbers max

Chunks:
{options_text}

Answer:"""
    
    try:
        llm = get_llm("rag")
        response = llm.invoke(prompt)
        
        if hasattr(response, "content"):
            response = response.content
        
        response = str(response).strip()
        
        numbers = re.findall(r"\d+", response)
        selected_indices = [int(n)-1 for n in numbers if 0 < int(n) <= len(docs)]
        
        if selected_indices:
            return [docs[i] for i in selected_indices[:top_k]]
        
    except Exception as e:
        print(f"LLM rerank failed: {e}")
    
    return docs[:top_k]


def ask_question(chain, query):
    """Main QA pipeline with retrieval-aware multimodal"""
    from llm import get_llm
    from langchain.schema import Document

    retriever = chain["retriever"]
    vectorstore = chain.get("vectorstore")
    clear_ce_cache()

    docs = retriever.invoke(expand_query(query))

    # ✅ HYBRID: Text retrieval + CLIP image retrieval
    image_docs = retrieve_images_by_clip(vectorstore, query, top_k=3) if vectorstore else []
    text_docs = [d for d in docs if d.metadata.get("chunk_type") != "image"]
    
    # Merge: text from traditional retrieval + images from CLIP
    if image_docs:
        # Convert image_docs dicts back to Document objects
        image_docs_converted = [Document(page_content=d["content"], metadata=d["metadata"]) for d in image_docs[:2]]
        docs = text_docs[:8] + image_docs_converted
    else:
        docs = text_docs[:10]

    # Source filtering
    query_type = detect_query_source(query)
    if query_type:
        filtered = []
        for doc in docs:
            source = doc.metadata.get("source", "").lower()
            if query_type == "nlp" and "nlp" in source:
                filtered.append(doc)
            elif query_type == "cnn" and any(k in source for k in ["cnn", "image"]):
                filtered.append(doc)
        if len(filtered) >= 2:
            docs = filtered

    # Rerank
    reranked_docs = multimodal_rerank(query, docs, top_k=5)

    # NEVER remove images in semantic filtering
    filtered_docs_temp = []
    for d in reranked_docs:
        if d.metadata.get("chunk_type") == "image":
            filtered_docs_temp.append(d)
        elif get_ce_score(query, d.page_content) > 0.65:
            filtered_docs_temp.append(d)

    reranked_docs = filtered_docs_temp if filtered_docs_temp else reranked_docs

    # LLM rerank (conditional)
    reranked_docs = llm_rerank(query, reranked_docs, top_k=3)

    # Re-inject images after LLM rerank
    images = [d for d in reranked_docs if d.metadata.get("chunk_type") == "image"]
    texts = [d for d in reranked_docs if d.metadata.get("chunk_type") != "image"]
    reranked_docs = texts + images

    # Prioritize conclusions
    reranked_docs = prioritize_conclusion_chunks(query, reranked_docs)

    # Force image inclusion for visual queries
    vis_query = is_visual_query(query)
    if vis_query:
        # Smart image selection (best, not first)
        images = sorted(
            [d for d in reranked_docs if d.metadata.get("chunk_type") == "image"],
            key=lambda d: get_ce_score(query, d.page_content),
            reverse=True
        )
        texts = [d for d in reranked_docs if d.metadata.get("chunk_type") != "image"]
        reranked_docs = images[:2] + texts

    # Always include 1–2 images
    images = [d for d in reranked_docs if d.metadata.get("chunk_type") == "image"]
    texts = [d for d in reranked_docs if d.metadata.get("chunk_type") != "image"]

    filtered_docs = texts[:3] + images[:2]
    if not filtered_docs:
        filtered_docs = reranked_docs[: max(3, min(6, len(reranked_docs)))]

    # Interleaved context
    context, image_paths = build_interleaved_context(filtered_docs)

    if len(context) > 4000:
        context_parts = context.split("\n\n")
        context = "\n\n".join(context_parts[:8])

    # Figure check
    if is_show_figure_query(query) and len(image_paths) == 0:
        return "No relevant figure found.", []

    # Use appropriate prompt
    if is_visual_query(query) and image_paths:
        prompt_template = DIAGRAM_PROMPT
    elif is_figure_query(query):
        prompt_template = DIAGRAM_PROMPT
    else:
        prompt_template = QA_PROMPT

    prompt = prompt_template.format(context=context, question=query)

    llm = get_llm("rag")
    answer = llm.invoke(prompt)

    if hasattr(answer, "content"):
        answer = answer.content

    answer = str(answer).strip()

    # Grounding
    if not is_answer_grounded_semantic(answer, context, threshold=0.7):
        answer = "Not specified in the provided documents."

    # Sources
    sources = set()
    for d in filtered_docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        if src and page:
            sources.add(f"{src} (Page {page})")

    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(sorted(sources))

    # Confidence
    pairs = [(query, d.page_content[:300]) for d in filtered_docs]
    ce_scores = get_reranker().predict(pairs)
    ce_scores = np.asarray(ce_scores)

    if len(ce_scores) > 0:
        min_score = float(np.min(ce_scores))
        max_score = float(np.max(ce_scores))
        
        if max_score > min_score:
            ce_scores_normalized = (ce_scores - min_score) / (max_score - min_score)
        else:
            ce_scores_normalized = np.ones_like(ce_scores) * 0.5
        
        avg_normalized = float(np.mean(ce_scores_normalized))
        spread = float(np.max(ce_scores_normalized) - np.min(ce_scores_normalized))
        
        consistency = 1.0 - (spread * 0.2)
        confidence_score = (avg_normalized * 0.6) + (consistency * 0.4)
    else:
        confidence_score = 0.5

    if confidence_score >= 0.55:
        conf = "High"
    elif confidence_score >= 0.40:
        conf = "Medium"
    else:
        conf = "Low"

    if "not specified" in answer.lower():
        conf = "Low"

    if is_answer_uncertain(answer):
        conf = "Low"

    if any(k in answer.lower() for k in ["dog and cat", "bird and fish"]):
        conf = "Low"

    answer += f"\n\n*Confidence: {conf}*"

    return answer, image_paths


def compare_documents(vectorstore, doc_names, aspect=""):
    """Compare multiple documents with structured output"""
    from llm import get_llm

    all_docs = []
    per_doc_limit = 2

    for doc in doc_names:
        try:
            retriever = HybridRetriever(vectorstore=vectorstore)
            
            query = f"{doc} comparison accuracy performance metrics"
            docs = retriever.invoke(query)

            reranked = multimodal_rerank(query, docs, top_k=per_doc_limit)
            all_docs.extend(reranked)

        except Exception as e:
            print(f"Error: {e}")

    if not all_docs:
        return "No content found."

    chunks = []
    for d in all_docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "N/A")
        chunks.append(f"[{src} p{page}]\n{d.page_content[:500]}")

    context = "\n\n".join(chunks)[:4000]

    if aspect:
        context = f"Aspect: {aspect}\n\n{context}"

    # Structured comparison prompt
    prompt = COMPARISON_PROMPT.format(
        context=context,
        documents=", ".join(doc_names),
        question=f"Compare these documents on {aspect if aspect else 'all aspects'}"
    )

    llm = get_llm("rag")

    try:
        answer = llm.invoke(prompt)
        if hasattr(answer, "content"):
            answer = answer.content
        return str(answer).strip()
    except Exception as e:
        return f"Failed: {e}"


def get_indexed_documents(vectorstore):
    """Get all indexed documents"""
    try:
        res = vectorstore.get(include=["metadatas"])
        sources = set()
        for m in res.get("metadatas", []):
            src = m.get("source")
            if src:
                sources.add(src)
        return sorted(list(sources))
    except Exception as e:
        print(f"Error: {e}")
        return []


def build_retrieval_chain(vectorstore):
    """Build retrieval chain"""
    return {
        "retriever": HybridRetriever(vectorstore=vectorstore),
        "vectorstore": vectorstore
    }