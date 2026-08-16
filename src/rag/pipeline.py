import os
import re
import numpy as np
from functools import lru_cache
from collections import defaultdict
from langchain.prompts import PromptTemplate

from .rerank import multimodal_rerank, get_ce_score, clear_ce_cache, encode_query_clip
from .grounding import is_answer_grounded_semantic, is_answer_uncertain
from src.rag.hybrid_retriever import HybridRetriever
from src.verification.verification_pipeline import process_answer

# =====================================================
# CLIP LOADER (CACHED)
# =====================================================
@lru_cache(maxsize=1)
def get_clip():
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    return model.to(device).eval(), preprocess, device


# =====================================================
# QUERY TYPE DETECTION
# =====================================================
def detect_query_type(query: str):
    q = query.lower().strip()

    comparison_patterns = [
        r"\bcompare\b",
        r"\bdifferences?\s+between\b",
        r"\bsimilarities?\s+between\b",
        r"\bversus\b",
        r"\bvs\.?\b",
    ]

    has_comparison_action = any(re.search(pattern, q) for pattern in comparison_patterns)

    if has_comparison_action:

        data_comparison_terms = [
            "parameter", "parameters", "parameter count", "accuracy",
            "validation accuracy", "inference time", "inference speed",
            "precision", "recall", "f1", "f1 score", "support", "loss",
            "latency", "score", "metric", "metrics", "performance",
        ]

        has_data_comparison = any(term in q for term in data_comparison_terms)

        document_terms = [
            "document", "documents", "paper", "papers",
            "pdf", "pdfs", "ppt", "pptx", "file", "files",
        ]

        has_document_reference = any(
            re.search(rf"\b{re.escape(term)}\b", q) for term in document_terms
        )

        if has_document_reference:
            return "comparison"

        if has_data_comparison:
            return "factual"

        return "comparison"

    visual_terms = [
        "figure", "fig.", "diagram", "architecture", "flowchart",
        "flow chart", "graph", "chart", "plot", "image",
        "illustration", "visual", "pipeline", "confusion matrix",
    ]

    if any(term in q for term in visual_terms):
        return "visual"

    analytical_terms = ["why", "how", "reason", "reasons", "analyze", "analyse", "explain"]

    if any(term in q for term in analytical_terms):
        return "analytical"

    return "factual"


# =====================================================
# QUERY EXPANSION
# =====================================================
def expand_query(query):
    query_lower = query.lower()

    expansions = {
        "diagram": "figure chart graph illustration visualization",
        "architecture": "system design structure framework components",
        "pipeline": "workflow process methodology steps preprocessing",
        "table": "data values rows columns dataset",
    }

    expanded = query
    for key, val in expansions.items():
        if key in query_lower:
            expanded += " " + val

    return expanded


def detect_document_scope(query: str, vectorstore):
    if vectorstore is None:
        return None

    try:
        res = vectorstore.get(include=["metadatas"])

        sources = sorted({
            metadata.get("source")
            for metadata in res.get("metadatas", [])
            if metadata and metadata.get("source")
        })

    except Exception as e:
        print(
            f"[DOCUMENT SCOPE] "
            f"Failed to read indexed sources: {e}"
        )
        return None

    if not sources:
        return None

    def normalize(text: str) -> str:
        if not text:
            return ""

        text = os.path.basename(text)
        text = os.path.splitext(text)[0]

        text = re.sub(
            r"^(?:test[\s_\-]*doc(?:ument)?|document|doc)[\s_\-]*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()

    query_normalized = normalize(query)

    print(
        f"[DOCUMENT SCOPE] "
        f"Normalized query: '{query_normalized}'"
    )

    query_tokens = set(query_normalized.split())
    matched_sources = []

    for source in sources:

        normalized_source = normalize(source)

        print(
            f"[DOCUMENT SCOPE] "
            f"Source='{source}' -> "
            f"Normalized='{normalized_source}'"
        )

        if not normalized_source:
            continue

        if normalized_source in query_normalized:

            print(
                f"[DOCUMENT SCOPE] "
                f"EXACT MATCH -> {source}"
            )

            matched_sources.append(source)
            continue

        source_tokens = {
            token
            for token in normalized_source.split()
            if len(token) >= 3
        }

        if not source_tokens:
            continue

        overlap = source_tokens & query_tokens

        print(
            f"[DOCUMENT SCOPE] "
            f"Token overlap for '{source}': "
            f"{sorted(overlap)}"
        )

        if len(overlap) >= 2:

            print(
                f"[DOCUMENT SCOPE] "
                f"TOKEN MATCH -> {source}"
            )

            matched_sources.append(source)

    matched_sources = list(
        dict.fromkeys(matched_sources)
    )

    if not matched_sources:

        print(
            "[DOCUMENT SCOPE] "
            "No reliable document scope detected. "
            "Retrieval remains global."
        )

        return None

    print(
        f"[DOCUMENT SCOPE] "
        f"Final matched sources: {matched_sources}"
    )

    return matched_sources


def retrieve_image_by_figure_number(vectorstore, query):
    match = re.search(r"\b(?:figure|fig\.?)\s*(\d+)\b", query, re.IGNORECASE)

    if not match:
        print("[FIGURE LOOKUP] No explicit figure number detected.")
        return []

    figure_number = match.group(1)

    print("\n" + "=" * 80)
    print("FIGURE NUMBER RETRIEVAL")
    print("=" * 80)
    print(f"Requested Figure : {figure_number}")
    print(f"Query            : {query}")

    try:
        results = vectorstore.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"[FIGURE LOOKUP] Vectorstore lookup failed: {e}")
        print("=" * 80)
        return []

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    def normalize_figure_text(text):
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r"\bfig(?:ure)?\.?\s*", "figure ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    target_reference = f"figure {figure_number}"

    cross_reference_patterns = [
        r"\bsee\s+figure\s+" + re.escape(figure_number),
        r"\bsee\s+the\s+figure\s+" + re.escape(figure_number),
        r"\brefer\s+to\s+figure\s+" + re.escape(figure_number),
        r"\brefer\s+to\s+the\s+figure\s+" + re.escape(figure_number),
        r"\bas\s+(?:shown|seen|illustrated|depicted)\s+in\s+figure\s+" + re.escape(figure_number),
        r"\bshown\s+in\s+figure\s+" + re.escape(figure_number),
        r"\bshown\s+in\s+the\s+figure\s+" + re.escape(figure_number),
        r"\bdescribed\s+in\s+figure\s+" + re.escape(figure_number),
        r"\bdiscussed\s+in\s+figure\s+" + re.escape(figure_number),
        r"\bdetails\s+in\s+figure\s+" + re.escape(figure_number),
        r"\bfigure\s+" + re.escape(figure_number) + r"\s+in\s+the\s+main\s+paper",
        r"\bfigure\s+" + re.escape(figure_number) + r"\s+of\s+the\s+main\s+paper",
        r"\bfigure\s+" + re.escape(figure_number) + r"\s+in\s+the\s+paper",
        r"\bfigure\s+" + re.escape(figure_number) + r"\s+above",
        r"\bfigure\s+" + re.escape(figure_number) + r"\s+below",
    ]

    def is_cross_reference(text):
        return any(re.search(p, text, re.IGNORECASE) for p in cross_reference_patterns)

    matched = []
    rejected = []

    for content, metadata in zip(documents, metadatas):

        metadata = metadata or {}

        if metadata.get("chunk_type") != "image":
            continue

        content = content or ""
        normalized_text = normalize_figure_text(content)

        if target_reference not in normalized_text:
            continue

        if is_cross_reference(normalized_text):
            rejected.append({"content": content, "metadata": metadata})
            continue

        matched.append({"content": content, "metadata": metadata})

    unique_matches = []
    seen = set()

    for item in matched:
        metadata = item["metadata"]
        key = (
            metadata.get("source"), metadata.get("page"),
            metadata.get("image_name"), metadata.get("image_path"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_matches.append(item)

    matched = unique_matches

    print(f"Matched Images   : {len(matched)}")
    print(f"Rejected References : {len(rejected)}")

    for i, item in enumerate(matched, start=1):
        metadata = item["metadata"]
        print(
            f"[MATCH {i}] Source={metadata.get('source')} | "
            f"Page={metadata.get('page')} | Image={metadata.get('image_name')} | "
            f"Path={metadata.get('image_path')}"
        )
        preview = item["content"].replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"    Evidence : {preview}")

    for i, item in enumerate(rejected, start=1):
        metadata = item["metadata"]
        print(
            f"[REJECTED {i}] Source={metadata.get('source')} | "
            f"Page={metadata.get('page')} | Image={metadata.get('image_name')}"
        )
        preview = item["content"].replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"    Reason   : Cross-reference rather than actual Figure {figure_number}")
        print(f"    Evidence : {preview}")

    if not matched:
        print(f"[FIGURE LOOKUP] No high-confidence direct match found for Figure {figure_number}.")
        if rejected:
            print("[FIGURE LOOKUP] Figure references were found, but they were classified as cross-references.")
        print("[FIGURE LOOKUP] Caller may fall back to CLIP retrieval.")

    if matched:
        sources = sorted({i["metadata"].get("source") for i in matched if i["metadata"].get("source")})
        pages = sorted({str(i["metadata"].get("page")) for i in matched if i["metadata"].get("page") is not None})
        print(f"Matched Sources  : {sources}")
        print(f"Matched Pages    : {pages}")
        print("[FIGURE LOOKUP] High-confidence figure reference found. Generic CLIP fallback should not be used.")

    print("=" * 80)

    return matched


def keep_best_figure_image_per_page(images, figure_number):
    """Keep one authoritative image for each source/page figure candidate.

    A PDF page may yield many embedded bitmap chunks, all carrying the same
    page-level Figure N caption in their nearby text. Preserving every one as
    an exact figure crowds out the text evidence that explains the figure.
    Prefer a page render when available because it retains the complete
    figure, caption, and surrounding page evidence.
    """
    if len(images) <= 1:
        return images

    target_caption = re.compile(
        rf"\bfigure\s+{re.escape(str(figure_number))}\s*:",
        re.IGNORECASE,
    )
    grouped = defaultdict(list)

    for image in images:
        metadata = image.get("metadata", {}) or {}
        grouped[(metadata.get("source"), metadata.get("page"))].append(image)

    selected = []

    for page_images in grouped.values():
        def priority(image):
            metadata = image.get("metadata", {}) or {}
            image_name = str(metadata.get("image_name", "")).lower()
            image_path = str(metadata.get("image_path", "")).lower()
            content = image.get("content", "") or ""

            return (
                bool(metadata.get("page_render_fallback")),
                "_render" in image_name or "_render" in image_path,
                bool(metadata.get("figure_page_match")),
                bool(target_caption.search(content)),
                len(content),
            )

        selected.append(max(page_images, key=priority))

    return selected


# =====================================================
# PAGE RENDER FALLBACK
# =====================================================
def render_pdf_page_as_image(source, page, output_dir="extracted_images"):
    """
    Last-resort fallback for explicit figure queries.

    If a figure's page has been identified via text evidence but no
    extracted image chunk exists for that page (i.e. the ingestion
    pipeline missed the figure), render that PDF page directly to a
    PNG using PyMuPDF and return it as an image-evidence dict
    compatible with the rest of the pipeline.

    Source PDFs live under:
        <project_root>/src/evaluation/datasets/benchmark_documents/

    Returns None on any failure so callers can fall through safely.
    """

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[PAGE RENDER FALLBACK] PyMuPDF (fitz) not installed. Skipping. "
              "Install with: pip install pymupdf --break-system-packages")
        return None

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    pdf_path = os.path.join(
        project_root,
        "src",
        "evaluation",
        "datasets",
        "benchmark_documents",
        os.path.basename(source),
    )

    print(f"[PAGE RENDER FALLBACK] PDF path resolved to: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"[PAGE RENDER FALLBACK] Source PDF not found: {pdf_path}")
        return None

    print(f"[PAGE RENDER FALLBACK] PDF found: {pdf_path}")
    print(f"[PAGE RENDER FALLBACK] Rendering page {page}...")

    try:
        page_index = int(page) - 1
    except (TypeError, ValueError):
        print(f"[PAGE RENDER FALLBACK] Invalid page number: {page}")
        return None

    try:
        doc = fitz.open(pdf_path)

        if page_index < 0 or page_index >= len(doc):
            print(f"[PAGE RENDER FALLBACK] Page {page} out of range for {source}")
            doc.close()
            return None

        pdf_page = doc[page_index]
        pix = pdf_page.get_pixmap(dpi=200)

        os.makedirs(output_dir, exist_ok=True)

        safe_source = os.path.splitext(source)[0].replace(" ", "_")
        out_path = os.path.join(
            output_dir,
            f"{safe_source}_page{page}_fallback_render.png",
        )

        pix.save(out_path)
        doc.close()

        print(f"[PAGE RENDER FALLBACK] Rendered {source} page {page} -> {out_path}")

        return {
            "content": (
                f"FALLBACK PAGE RENDER (no extracted figure image was "
                f"available for this page). This is a full-page render "
                f"of {source} page {page}, which contains the requested figure."
            ),
            "metadata": {
                "source": source,
                "page": page,
                "chunk_type": "image",
                "image_name": os.path.basename(out_path),
                "image_path": os.path.abspath(out_path),
                "figure_page_match": True,
                "page_render_fallback": True,
            },
        }

    except Exception as e:
        print(f"[PAGE RENDER FALLBACK] Failed to render {source} page {page}: {e}")
        return None


# =====================================================
# IMAGE RETRIEVAL — HYBRID CLIP + TEXT RELEVANCE
# =====================================================
def retrieve_images_by_clip(vectorstore, query, top_k=3, allowed_sources=None):
    """
    Retrieve image chunks using CLIP + Cross-Encoder.

    ONLY CHANGE (this pass): accepts an optional `allowed_sources` list.
    When provided, the candidate pool is filtered to those sources
    BEFORE scoring/ranking, not after. Previously document-scope
    filtering happened on the already-selected global top_k, which
    meant relevant in-scope images that didn't crack the global top_k
    (out of hundreds of chunks across every indexed paper) were
    silently dropped and never had a chance to be scored at all.
    """
    try:
        from PIL import Image
        from sklearn.metrics.pairwise import cosine_similarity
        import torch

        model, preprocess, device = get_clip()

        res = vectorstore.get(include=["documents", "metadatas"])

        image_candidates = []

        for content, metadata in zip(res.get("documents", []), res.get("metadatas", [])):
            if metadata.get("chunk_type") != "image":
                continue
            image_candidates.append({"content": content, "metadata": metadata})

        total_before_scope = len(image_candidates)

        if allowed_sources:
            allowed_set = set(allowed_sources)
            image_candidates = [
                img for img in image_candidates
                if img["metadata"].get("source") in allowed_set
            ]

        print("\n" + "=" * 80)
        print("IMAGE RETRIEVAL DIAGNOSTIC")
        print("=" * 80)
        print(f"Query                 : {query}")
        if allowed_sources:
            print(f"Allowed Sources       : {sorted(set(allowed_sources))}")
            print(f"Image Candidates      : {total_before_scope} -> {len(image_candidates)} (scoped before ranking)")
        else:
            print(f"Image Candidates      : {len(image_candidates)}")

        if not image_candidates:
            print("RESULT: NO IMAGE CANDIDATES EXIST")
            print("=" * 80)
            return []

        q_emb = encode_query_clip(query)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scored = []

        for index, img in enumerate(image_candidates):

            content = img.get("content", "") or ""
            metadata = img.get("metadata", {})

            try:
                text_score = float(get_ce_score(query, content))
            except Exception as e:
                print(f"[IMAGE CE ERROR] Candidate {index}: {e}")
                text_score = 0.0

            clip_score = 0.0
            image_path = metadata.get("image_path")

            if image_path:
                path = os.path.abspath(image_path)

                if os.path.exists(path):
                    try:
                        image = Image.open(path).convert("RGB")
                        pixel = preprocess(image).unsqueeze(0).to(device)

                        with torch.no_grad():
                            emb = model.encode_image(pixel).cpu().numpy()[0]

                        emb_norm = np.linalg.norm(emb)
                        if emb_norm > 0:
                            emb = emb / emb_norm
                            clip_score = float(cosine_similarity([q_emb], [emb])[0][0])

                    except Exception as e:
                        print(f"[IMAGE CLIP ERROR] {path}: {e}")
                else:
                    print(f"[IMAGE PATH MISSING] {path}")

            hybrid_score = 0.60 * text_score + 0.40 * clip_score
            scored.append((hybrid_score, text_score, clip_score, img))

        scored.sort(key=lambda x: x[0], reverse=True)

        print("\nALL IMAGE CANDIDATES")
        print("-" * 80)

        for rank, (hybrid_score, text_score, clip_score, img) in enumerate(scored, start=1):
            metadata = img["metadata"]
            print(
                f"Rank={rank:02d} | Source={metadata.get('source')} | "
                f"Page={metadata.get('page')} | Image={metadata.get('image_name')} | "
                f"Hybrid={hybrid_score:.4f} | CE={text_score:.4f} | CLIP={clip_score:.4f}"
            )

        print("\nTOP IMAGE RESULTS")
        print("-" * 80)

        for rank, (hybrid_score, text_score, clip_score, img) in enumerate(scored[:top_k], start=1):
            metadata = img["metadata"]
            print(
                f"TOP-{rank} | Source={metadata.get('source')} | "
                f"Page={metadata.get('page')} | Image={metadata.get('image_name')} | "
                f"Hybrid={hybrid_score:.4f} | CE={text_score:.4f} | CLIP={clip_score:.4f}"
            )

        print("=" * 80)

        return [img for _, _, _, img in scored[:top_k]]

    except Exception as e:
        print(f"[IMAGE RETRIEVAL ERROR] {e}")
        return []


# =====================================================
# IMPROVED CONTEXT BUILDER (FREEZE VERSION)
# =====================================================
def build_interleaved_context(query, docs):
    from collections import defaultdict

    image_paths = []
    ordered_sources = []
    grouped = defaultdict(lambda: defaultdict(list))

    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", -1)
        if source not in ordered_sources:
            ordered_sources.append(source)
        grouped[source][page].append(doc)

    context_parts = []

    for source in ordered_sources:

        context_parts.append("\n" + "=" * 80 + f"\nDOCUMENT : {source}\n" + "=" * 80)

        pages = sorted(
            grouped[source].keys(),
            key=lambda p: int(p) if str(p).isdigit() else float("inf")
        )

        for page in pages:

            page_label = page if page != -1 else "Unknown"
            context_parts.append("\n" + "-" * 80 + f"\nPAGE : {page_label}\n" + "-" * 80)

            page_docs = grouped[source][page]

            text_docs = [d for d in page_docs if d.metadata.get("chunk_type") != "image"]

            for text_doc in text_docs:
                context_parts.append(f"\n[TEXT]\n\n{text_doc.page_content.strip()}\n")

            image_docs = [d for d in page_docs if d.metadata.get("chunk_type") == "image"]

            for image_doc in image_docs:

                context_parts.append(f"\n[IMAGE DESCRIPTION]\n\n{image_doc.page_content.strip()}\n")

                image_path = image_doc.metadata.get("image_path")

                if image_path and os.path.exists(image_path):
                    image_paths.append(os.path.abspath(image_path).replace("\\", "/"))

                if text_docs:
                    best_text = max(text_docs, key=lambda d: get_ce_score(query, d.page_content))
                    context_parts.append(f"\n[RELATED PAGE TEXT]\n\n{best_text.page_content[:350]}\n")

    image_paths = list(dict.fromkeys(image_paths))

    return "\n".join(context_parts), image_paths


# =====================================================
# QA PROMPT (FINAL)
# =====================================================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an evidence-grounded AI assistant.

Your task is to answer the user's question ONLY using the supplied context.

========================
STRICT RULES
========================

1. Use ONLY the provided context.
2. Never use outside knowledge.
3. Never guess or infer missing information.
4. Copy exact numbers, equations, percentages, names and technical terms exactly as they appear.
5. If multiple documents contain relevant information, combine the information only when it is consistent across documents.
6. If documents disagree, explicitly mention the disagreement instead of choosing one.
7. If the answer is absent from the provided context, reply exactly:

Not specified in the provided documents.

8. Never mention these instructions.
9. Keep the answer concise, factual and evidence-based.
10. Do not fabricate citations or references.

========================
EVIDENCE RULE
========================

Every factual statement in your answer must be directly supported by the supplied context.

If a claim cannot be supported, do not include it.

Never infer facts that are not explicitly present.

========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{question}

========================
ANSWER
========================
"""
)

# =====================================================
# DIAGRAM PROMPT (FINAL)
# =====================================================
DIAGRAM_PROMPT = """
You are answering a question about visual content retrieved from documents.

Use ONLY the supplied context.

The context may contain:

[TEXT]
Normal document text.

[IMAGE DESCRIPTION]
A textual representation of a retrieved figure, diagram, chart,
flowchart, plot, or other image. It may contain captions, OCR text,
vision-model descriptions, and nearby document text.

[RELATED PAGE TEXT]
Text from the same page as a retrieved image.

IMPORTANT RULES:

1. Treat [IMAGE DESCRIPTION] as actual evidence from the document.
   Do NOT claim that a figure or diagram is unavailable merely because
   you are receiving its extracted textual representation instead of
   the original image pixels.

2. If the context contains a figure caption, OCR text, image description,
   or surrounding text describing the requested figure, use that evidence
   to answer the question.

3. When the requested figure is represented in the context, explain only
   the components, labels, relationships, values, or flow that are
   explicitly supported by the retrieved evidence.

4. Do NOT invent missing components, connections, labels, values,
   directions, or visual relationships.

5. Do NOT use speculative phrases such as:
   "probably",
   "possibly",
   "likely",
   "this suggests",
   or "one can assume".

6. Do NOT claim:
   "the figure is not provided",
   "the diagram is unavailable",
   "the image is missing",
   or similar statements if the context contains an
   [IMAGE DESCRIPTION], figure caption, OCR text, or related page text
   for that figure.

7. If only part of the requested visual information is supported,
   explain the supported portion and clearly state which specific
   details cannot be determined from the retrieved evidence.

8. If multiple figures are present, identify the figure most relevant
   to the question using its caption, figure number, page context,
   OCR text, and description. Do not combine unrelated figures.

9. Prefer explicit document evidence over general knowledge.

10. Do not infer document-wide absence from the absence of information
    in one retrieved chunk.

Context:
{context}

Question:
{question}

Answer:
"""


# =====================================================
# COMPARISON PROMPT (FINAL)
# =====================================================
COMPARISON_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are comparing one or more technical documents.

Use ONLY the supplied context.

========================
STRICT RULES
========================

1. Never use outside knowledge.
2. Never speculate.
3. Never write generic advice.
4. Copy exact numerical values whenever available.
5. Never merge information from different documents.
6. Keep each document completely independent.
7. If information is missing for a document, write exactly:

Not specified in document.

8. Compare ONLY evidence present in the supplied context.
9. Do not fabricate performance metrics.
10. Mention page-specific evidence whenever possible.

========================
OUTPUT FORMAT
========================

For EACH document found in the context, provide the following:

--------------------------------------------------

Document Name:

Architecture

Pipeline

Datasets

Model / Algorithm

Performance Metrics

Advantages

Limitations

Important Observations

--------------------------------------------------

After ALL documents have been summarized, provide:

Technical Comparison

- Architecture Differences

- Pipeline Differences

- Model Differences

- Dataset Differences

- Performance Comparison

- Strengths

- Weaknesses

- Best Performing Document (ONLY if supported by evidence)

- Overall Conclusion

========================
IMPORTANT
========================

If only ONE document is available, do NOT invent comparisons.

Instead, summarize that document using the same structure and explicitly state that no comparison can be made.

========================
EVIDENCE RULE
========================

Every factual statement in your answer must be directly supported by the supplied context.

If a claim cannot be supported, do not include it.

Never infer facts that are not explicitly present.

Never compare information unless the comparison is explicitly supported by the retrieved evidence.

========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{question}

========================
ANSWER
========================
"""
)

# =====================================================
# MAIN PIPELINE
# =====================================================
def ask_question(chain, query):

    from src.rag.summarizer import get_llm
    from langchain.schema import Document

    retriever = chain["retriever"]
    vectorstore = chain.get("vectorstore")

    clear_ce_cache()

    print("\n" + "#" * 100)
    print("FORENSIC CASE START")
    print("#" * 100)

    expanded_query = expand_query(query)
    qtype = detect_query_type(query)

    print(f"Original Query   : {query}")
    print(f"Expanded Query   : {expanded_query}")
    print(f"Query Type       : {qtype}")

    if qtype == "comparison":
        document_scope = None
    else:
        document_scope = detect_document_scope(query, vectorstore)

    print("\n" + "=" * 80)
    print("DOCUMENT SCOPE DEBUG")
    print("=" * 80)
    print(f"Query          : {query}")
    print(f"Expanded Query : {expanded_query}")
    print(f"Query Type     : {qtype}")
    print(f"Document Scope : {document_scope}")
    print("=" * 80)

    # =====================================================
    # STAGE 1 — HYBRID RETRIEVAL
    # =====================================================

    if document_scope:
        docs = retriever.invoke_scoped(expanded_query, document_scope)
    else:
        docs = retriever.invoke(expanded_query)

    print("\n" + "=" * 80)
    print("STAGE 1 — HYBRID RETRIEVAL")
    print("=" * 80)
    print(f"Retrieved Count : {len(docs)}")

    for rank, doc in enumerate(docs, start=1):
        print(
            f"\n[HYBRID #{rank}] Source={doc.metadata.get('source')} | "
            f"Page={doc.metadata.get('page')} | Type={doc.metadata.get('chunk_type')}"
        )
        print(f"Content: {doc.page_content[:500].replace(chr(10), ' ')}")

    print("=" * 80)

    if document_scope:
        scoped_docs = [doc for doc in docs if doc.metadata.get("source") in document_scope]
        print(f"[DOCUMENT SCOPE] Text chunks: {len(docs)} -> {len(scoped_docs)}")
        docs = scoped_docs

    # =====================================================
    # STAGE 2 — FIGURE / IMAGE RETRIEVAL
    # =====================================================

    figure_match = re.search(r"\b(?:figure|fig\.?)\s*(\d+)\b", query, re.IGNORECASE)
    explicit_figure_query = figure_match is not None
    figure_number = figure_match.group(1) if figure_match else None
    exact_figure_match = False
    figure_source_scope = document_scope

    if explicit_figure_query:
        print("\n" + "=" * 80)
        print("EXPLICIT FIGURE QUERY DETECTED")
        print("=" * 80)
        print(f"Requested Figure : {figure_number}")
        print("=" * 80)

        # When the question does not name a document, Figure N alone is not
        # document identity: every indexed paper can have a Figure N. Use the
        # Stage-1 text evidence as the source hint before accepting a direct
        # figure-image match as authoritative.
        if not figure_source_scope:
            target_reference = f"figure {figure_number}"

            def normalize_figure_reference(text):
                if not text:
                    return ""
                text = str(text).lower()
                text = re.sub(r"\bfig(?:ure)?\.?\s*", "figure ", text)
                return re.sub(r"\s+", " ", text).strip()

            figure_source_scope = sorted({
                doc.metadata.get("source")
                for doc in docs
                if doc.metadata.get("source")
                and doc.metadata.get("chunk_type") != "image"
                and target_reference in normalize_figure_reference(doc.page_content)
            })

            print(
                "[FIGURE SOURCE HINT] "
                f"Stage-1 sources mentioning Figure {figure_number}: "
                f"{figure_source_scope or 'None'}"
            )

    images = []

    if explicit_figure_query:

        images = retrieve_image_by_figure_number(vectorstore, query)

        if figure_source_scope:
            images = [
                img for img in images
                if img.get("metadata", {}).get("source") in figure_source_scope
            ]
        elif not document_scope:
            # No source identity was established. Do not let a global Figure N
            # match trigger exact-figure preservation downstream.
            images = []

        if images:
            original_image_count = len(images)
            images = keep_best_figure_image_per_page(images, figure_number)
            print(
                "[FIGURE IMAGE DEDUPLICATION] "
                f"{original_image_count} -> {len(images)} authoritative "
                "image(s), one per source/page."
            )

        exact_figure_match = bool(images)

        print("\n" + "=" * 80)
        print("EXACT FIGURE RETRIEVAL RESULT")
        print("=" * 80)
        print(f"Requested Figure : {figure_number}")
        print(f"Exact Match      : {exact_figure_match}")
        print(f"Matched Images   : {len(images)}")

        for i, image in enumerate(images, start=1):
            metadata = image.get("metadata", {})
            print(
                f"[EXACT FIGURE #{i}] Source={metadata.get('source')} | "
                f"Page={metadata.get('page')} | Image={metadata.get('image_name')} | "
                f"Path={metadata.get('image_path')}"
            )

        print("=" * 80)

    else:
        # ONLY CHANGE: pass document_scope so the CLIP candidate pool is
        # filtered to the target document BEFORE ranking, not after.
        images = retrieve_images_by_clip(
            vectorstore,
            query,
            top_k=3,
            allowed_sources=document_scope,
        )

    # =====================================================
    # FIGURE PAGE → IMAGE ASSOCIATION
    # =====================================================

    if explicit_figure_query and not exact_figure_match:

        print("\n" + "=" * 80)
        print("FIGURE PAGE → IMAGE ASSOCIATION")
        print("=" * 80)

        target_reference = f"figure {figure_number}"

        def normalize_figure_reference(text):
            if not text:
                return ""
            text = str(text).lower()
            text = re.sub(r"\bfig(?:ure)?\.?\s*", "figure ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        figure_pages = set()

        for doc in docs:
            content = doc.page_content or ""
            normalized = normalize_figure_reference(content)

            if target_reference in normalized:
                source = doc.metadata.get("source")
                page = doc.metadata.get("page")
                if source is not None and page is not None:
                    figure_pages.add((source, page))

        if not figure_pages:
            try:
                results = vectorstore.get(include=["documents", "metadatas"])
                store_documents = results.get("documents", [])
                store_metadatas = results.get("metadatas", [])

                for content, metadata in zip(store_documents, store_metadatas):
                    metadata = metadata or {}
                    if metadata.get("chunk_type") == "image":
                        continue

                    normalized = normalize_figure_reference(content)

                    if target_reference in normalized:
                        source = metadata.get("source")
                        page = metadata.get("page")
                        if source is not None and page is not None:
                            figure_pages.add((source, page))

            except Exception as e:
                print(f"[FIGURE PAGE LOOKUP] Vectorstore scan failed: {e}")

        if figure_source_scope:
            figure_pages = {
                (source, page) for source, page in figure_pages
                if source in figure_source_scope
            }

        print(f"Figure Page Candidates : {len(figure_pages)}")

        for source, page in sorted(figure_pages, key=lambda x: (str(x[0]), str(x[1]))):
            print(f"[FIGURE PAGE] Source={source} | Page={page}")

        page_images = []

        try:
            results = vectorstore.get(include=["documents", "metadatas"])
            store_documents = results.get("documents", [])
            store_metadatas = results.get("metadatas", [])

            seen_image_keys = set()

            for content, metadata in zip(store_documents, store_metadatas):
                metadata = metadata or {}

                if metadata.get("chunk_type") != "image":
                    continue

                source = metadata.get("source")
                page = metadata.get("page")

                if (source, page) not in figure_pages:
                    continue

                key = (source, page, metadata.get("image_name"), metadata.get("image_path"))

                if key in seen_image_keys:
                    continue

                seen_image_keys.add(key)

                page_images.append({
                    "content": content or "",
                    "metadata": {
                        **metadata,
                        "figure_page_match": True,
                        "figure_number": figure_number,
                    },
                })

        except Exception as e:
            print(f"[FIGURE PAGE IMAGE LOOKUP] Vectorstore lookup failed: {e}")

        if page_images:

            images = page_images
            exact_figure_match = True

            print(f"[FIGURE PAGE IMAGE LOOKUP] Found {len(images)} image(s) on the Figure {figure_number} page.")

            for i, image in enumerate(images, start=1):
                metadata = image.get("metadata", {})
                print(
                    f"[FIGURE PAGE IMAGE #{i}] Source={metadata.get('source')} | "
                    f"Page={metadata.get('page')} | Image={metadata.get('image_name')}"
                )

            print("[FIGURE PAGE IMAGE LOOKUP] Same-page figure evidence will be treated as authoritative.")

        else:

            print("[FIGURE PAGE IMAGE LOOKUP] No image chunk found on the identified Figure page.")

            if figure_pages:

                print("\n" + "=" * 80)
                print("PAGE RENDER FALLBACK")
                print("=" * 80)

                rendered_images = []

                for source, page in figure_pages:
                    rendered = render_pdf_page_as_image(source, page)
                    if rendered:
                        rendered_images.append(rendered)

                if rendered_images:
                    images = rendered_images
                    exact_figure_match = True
                    print(f"[PAGE RENDER FALLBACK] Recovered {len(images)} page render(s) as figure evidence.")
                else:
                    print("[PAGE RENDER FALLBACK] Could not render any candidate page. Falling through to CLIP retrieval.")

                print("=" * 80)

    # =====================================================
    # CLIP FALLBACK
    # =====================================================

    if not images:

        print("[FIGURE PIPELINE] No exact figure/page image match. Using CLIP retrieval.")
        # ONLY CHANGE: scope this fallback too, preferring whatever source
        # identity we've established (explicit figure source hint, else the
        # global document scope) so it doesn't repeat the same global-top-k
        # problem for figure queries that fell through every other path.
        images = retrieve_images_by_clip(
            vectorstore,
            query,
            top_k=3,
            allowed_sources=figure_source_scope or document_scope,
        )

    elif explicit_figure_query:

        print("[FIGURE PIPELINE] Explicit figure/page image match found. CLIP fallback disabled.")

    if document_scope:
        scoped_images = [
            img for img in images
            if img.get("metadata", {}).get("source") in document_scope
        ]
        print(f"[DOCUMENT SCOPE] Images: {len(images)} -> {len(scoped_images)}")
        images = scoped_images

    print("\n" + "=" * 80)
    print("STAGE 2 — IMAGE RETRIEVAL RESULT")
    print("=" * 80)
    print(f"Image Results       : {len(images)}")
    print(f"Exact Figure Match  : {exact_figure_match}")

    for rank, img in enumerate(images, start=1):
        metadata = img.get("metadata", {})
        print(
            f"\n[IMAGE #{rank}] Source={metadata.get('source')} | "
            f"Page={metadata.get('page')} | Image={metadata.get('image_name')}"
        )
        print(f"Image Path: {metadata.get('image_path')}")
        print(f"Figure Page Match: {metadata.get('figure_page_match', False)}")
        print(f"Content: {img.get('content', '')[:500].replace(chr(10), ' ')}")

    print("=" * 80)

    image_docs = [
        Document(
            page_content=i["content"],
            metadata={**i["metadata"], "chunk_type": "image", "exact_figure_match": exact_figure_match},
        )
        for i in images
    ]

    all_docs = docs + image_docs

    print("\n" + "=" * 80)
    print("COMBINED RETRIEVAL CANDIDATES")
    print("=" * 80)
    print(f"Text Candidates  : {len(docs)}")
    print(f"Image Candidates : {len(image_docs)}")
    print(f"Total Candidates : {len(all_docs)}")
    print("=" * 80)

    # =====================================================
    # STAGE 3 — MULTIMODAL RERANKING
    # =====================================================

    reranked = multimodal_rerank(query=query, docs=all_docs, top_k=12)

    print("\n" + "=" * 80)
    print("STAGE 3 — MULTIMODAL RERANKING")
    print("=" * 80)
    print(f"Input Candidates : {len(all_docs)}")
    print(f"Reranked Output   : {len(reranked)}")

    for rank, doc in enumerate(reranked, start=1):
        print(
            f"\n[RERANK #{rank}] Source={doc.metadata.get('source')} | "
            f"Page={doc.metadata.get('page')} | Type={doc.metadata.get('chunk_type')}"
        )
        print(f"Exact Figure : {doc.metadata.get('exact_figure_match', False)}")
        print(f"Figure Page Match : {doc.metadata.get('figure_page_match', False)}")
        print(f"Content: {doc.page_content[:300].replace(chr(10), ' ')}")

    print("=" * 80)

    if explicit_figure_query and exact_figure_match:

        exact_figure_docs = [d for d in image_docs if d.metadata.get("exact_figure_match", False)]

        reranked_keys = {
            (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("image_path"))
            for d in reranked
        }

        missing_exact_figures = [
            d for d in exact_figure_docs
            if (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("image_path")) not in reranked_keys
        ]

        if missing_exact_figures:

            print("\n" + "=" * 80)
            print("EXACT FIGURE PRESERVATION")
            print("=" * 80)
            print(f"Exact figure candidates missing after reranking: {len(missing_exact_figures)}")

            reranked = missing_exact_figures + reranked

            unique_reranked = []
            seen = set()

            for doc in reranked:
                key = (
                    doc.metadata.get("source"), doc.metadata.get("page"),
                    doc.metadata.get("image_path"), doc.metadata.get("chunk_type"),
                )
                if key in seen:
                    continue
                seen.add(key)
                unique_reranked.append(doc)

            reranked = unique_reranked

            print("Exact figure evidence restored to reranked candidates.")
            print("=" * 80)

    imgs = [d for d in reranked if d.metadata.get("chunk_type") == "image"]
    txts = [d for d in reranked if d.metadata.get("chunk_type") != "image"]

    # =====================================================
    # DOCUMENT SELECTION
    # =====================================================

    if qtype == "comparison":

        grouped = defaultdict(list)
        for doc in reranked:
            grouped[doc.metadata.get("source")].append(doc)

        final_docs = []
        for source_docs in grouped.values():
            final_docs.extend(source_docs[:4])

        final_docs = final_docs[:12]

    else:
        final_docs = reranked[:12]

    # =====================================================
    # EXPLICIT FIGURE FINAL-CONTEXT PROTECTION
    # =====================================================

    if explicit_figure_query and exact_figure_match:

        exact_figure_docs = [
            d for d in reranked
            if d.metadata.get("exact_figure_match", False) and d.metadata.get("chunk_type") == "image"
        ]

        if exact_figure_docs:

            final_keys = {
                (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("image_path"))
                for d in final_docs
            }

            missing_exact = [
                d for d in exact_figure_docs
                if (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("image_path")) not in final_keys
            ]

            if missing_exact:

                print("\n" + "=" * 80)
                print("EXACT FIGURE FINAL-CONTEXT PROTECTION")
                print("=" * 80)
                print(f"Restoring {len(missing_exact)} exact figure evidence item(s).")

                final_docs = missing_exact + final_docs
                final_docs = final_docs[:12]

                print("Exact figure evidence is guaranteed to remain in final context.")
                print("=" * 80)

    print("\n" + "=" * 80)
    print("FINAL CONTEXT DOCUMENTS")
    print("=" * 80)
    print(f"Final context candidates: {len(final_docs)}")

    exact_figure_in_final = False

    for rank, doc in enumerate(final_docs, start=1):
        metadata = doc.metadata or {}
        is_exact_figure = metadata.get("exact_figure_match", False)

        if is_exact_figure:
            exact_figure_in_final = True

        print(
            f"\n[FINAL #{rank}] Source={metadata.get('source')} | "
            f"Page={metadata.get('page')} | Type={metadata.get('chunk_type')}"
        )
        print(f"Exact Figure : {is_exact_figure}")
        print(f"Figure Page Match : {metadata.get('figure_page_match', False)}")
        print(f"Content: {doc.page_content[:300].replace(chr(10), ' ')}")

    print("=" * 80)

    if explicit_figure_query:

        print("\n" + "=" * 80)
        print("FIGURE FINAL-CONTEXT DIAGNOSTIC")
        print("=" * 80)
        print(f"Requested Figure     : {figure_number}")
        print(f"Exact Figure Match   : {exact_figure_match}")
        print(f"Present In Final     : {exact_figure_in_final}")

        if exact_figure_match and exact_figure_in_final:
            print("[FIGURE DIAGNOSTIC] PASS — exact figure/page image evidence survived into final context.")
        elif exact_figure_match:
            print("[FIGURE DIAGNOSTIC] FAIL — figure/page image was retrieved but disappeared before final context.")
        else:
            print("[FIGURE DIAGNOSTIC] FAIL — exact figure/page image was not retrieved.")

        print("=" * 80)

    context, image_paths = build_interleaved_context(query, final_docs)

    print("\n" + "=" * 80)
    print("FINAL CONTEXT SENT TO LLM")
    print("=" * 80)
    print(context)
    print("=" * 80)

    if qtype == "comparison":
        prompt = COMPARISON_PROMPT
    elif qtype == "visual" and image_paths:
        prompt = DIAGRAM_PROMPT
    else:
        prompt = QA_PROMPT

    print("\n" + "=" * 80)
    print("RAG PIPELINE DEBUG")
    print("=" * 80)
    print(f"Query Type        : {qtype}")
    print(f"Retrieved Chunks  : {len(docs)}")
    print(f"Retrieved Images  : {len(image_docs)}")
    print(f"Reranked Chunks   : {len(reranked)}")
    print(f"Reranked Text     : {len(txts)}")
    print(f"Reranked Images   : {len(imgs)}")
    print(f"Final Context Docs: {len(final_docs)}")
    print(f"Exact Figure Query: {explicit_figure_query}")
    print(f"Exact Figure Match: {exact_figure_match}")
    print(f"Exact Figure Final: {exact_figure_in_final}")
    print("=" * 80)

    llm = get_llm("rag")

    response = llm.invoke(prompt.format(context=context, question=query))

    answer = response.content if hasattr(response, "content") else str(response)

    # =====================================================
    # STAGE 4 — EVIDENCE VERIFICATION
    # =====================================================
    #
    # Verify the generated answer against the same evidence
    # (final_docs) that was actually sent to the LLM as
    # context -- not the raw pre-rerank retrieval -- so this
    # checks whether the answer is grounded in what the model
    # actually saw. Wrapped defensively: a verification failure
    # (model load, spaCy, etc.) must not break answer generation,
    # which already succeeded by this point.

    verification_result = None

    try:
        verification_result = process_answer(answer, final_docs, top_k=3)

        from src.verification.models import VerificationStatus

        status_counts = defaultdict(int)
        for unit in verification_result.verification_units:
            status_counts[unit.verification_status.value] += 1

        total_units = len(verification_result.verification_units)
        supported = status_counts.get("SUPPORTED", 0)
        groundedness = (supported / total_units) if total_units else None

        print("\n" + "=" * 80)
        print("STAGE 4 — EVIDENCE VERIFICATION")
        print("=" * 80)
        print(f"Verification Units : {total_units}")
        print(f"Supported          : {status_counts.get('SUPPORTED', 0)}")
        print(f"Partially Supported: {status_counts.get('PARTIALLY_SUPPORTED', 0)}")
        print(f"Unsupported        : {status_counts.get('UNSUPPORTED', 0)}")
        print(f"Unknown            : {status_counts.get('UNKNOWN', 0)}")
        print(
            f"Groundedness Ratio : "
            f"{f'{groundedness:.2f}' if groundedness is not None else 'N/A'}"
        )

        for unit in verification_result.verification_units:
            print(
                f"\n[CLAIM #{unit.id}] {unit.text}"
            )
            print(
                f"  Status={unit.verification_status.value} | "
                f"NLI={unit.nli_label.value} | "
                f"Confidence={unit.confidence} | "
                f"Citation={unit.citation}"
            )

        print("=" * 80)

    except Exception as e:
        print(
            f"[VERIFICATION] Failed, continuing without verification: {e}"
        )
        verification_result = None

    return answer, image_paths, verification_result


# =====================================================
# HELPERS
# =====================================================
def build_retrieval_chain(vectorstore):
    return {
        "retriever": HybridRetriever(vectorstore=vectorstore),
        "vectorstore": vectorstore,
    }

def get_indexed_documents(vectorstore):
    try:
        res = vectorstore.get(include=["metadatas"])
        return sorted({
            m.get("source")
            for m in res.get("metadatas", [])
            if m.get("source")
        })
    except:
        return []