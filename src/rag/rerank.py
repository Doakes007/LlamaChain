import os
from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


device = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_clip_model():
    """Load and cache CLIP model and preprocess."""

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )

    return model.to(device).eval(), preprocess


@lru_cache(maxsize=1)
def get_reranker():
    """Load and cache Cross-Encoder."""

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=device
    )


_ce_score_cache = {}


def get_ce_score(query, content):
    """Cache Cross-Encoder scores to prevent redundant computation."""

    # Use a larger content window for Cross-Encoder scoring.
    # The previous 300-character truncation could hide the
    # actual evidence inside technical/table/figure chunks.
    content_for_reranking = content[:1000]

    cache_key = f"{query}||{content_for_reranking}"

    if cache_key not in _ce_score_cache:

        pairs = [
            (query, content_for_reranking)
        ]

        _ce_score_cache[cache_key] = float(
            get_reranker().predict(pairs)[0]
        )

    return _ce_score_cache[cache_key]

def clear_ce_cache():
    """Clear Cross-Encoder score cache."""

    global _ce_score_cache

    _ce_score_cache = {}


def encode_query_clip(query):
    """Generate CLIP text embedding for the query."""

    model, _ = get_clip_model()

    tokenizer = open_clip.get_tokenizer(
        "ViT-B-32"
    )

    with torch.no_grad():

        text_features = model.encode_text(
            tokenizer([query]).to(device)
        )

    return text_features.cpu().numpy()[0]


def multimodal_rerank(query, docs, top_k=5):
    """
    Rerank retrieved documents using rank fusion.

    Text chunks:
        Ranked using Cross-Encoder semantic relevance.

    Image chunks:
        Ranked using both:
        - Cross-Encoder semantic relevance
        - CLIP visual similarity

    Rank fusion avoids directly combining raw Cross-Encoder
    logits with CLIP cosine similarity, since those scores
    are on different numerical scales.

    Diagnostic instrumentation:
        Prints the complete reranker input and complete
        reranker ranking so that retrieval failures can be
        distinguished from reranking failures.
    """

    if not docs:
        return []

    # =================================================
    # DIAGNOSTIC — RERANKER INPUT
    # =================================================

    print("\n" + "=" * 80)
    print("STEP 3A — RERANKER INPUT")
    print("=" * 80)

    print(f"Query: {query}")
    print(f"Input candidates: {len(docs)}")

    for rank, doc in enumerate(
        docs,
        start=1,
    ):

        metadata = doc.metadata or {}

        print(
            f"{rank:02d}. "
            f"Source={metadata.get('source')} | "
            f"Page={metadata.get('page')} | "
            f"Type={metadata.get('chunk_type', 'text')} | "
            f"Image={metadata.get('image_path', 'N/A')}"
        )

        preview = (
            (doc.page_content or "")
            .replace("\n", " ")
            .strip()
        )

        print(
            f"    Content={preview[:180]}"
        )

    print("=" * 80)

    # -------------------------------------------------
    # 1. CROSS-ENCODER SCORES
    # -------------------------------------------------

    ce_results = []

    for index, doc in enumerate(docs):

        content = doc.page_content or ""

        try:

            ce_score = float(
                get_ce_score(
                    query,
                    content,
                )
            )

        except Exception as e:

            print(
                f"Cross-Encoder reranking failed: {e}"
            )

            ce_score = float("-inf")

        ce_results.append(
            {
                "index": index,
                "doc": doc,
                "ce_score": ce_score,
                "clip_score": None,
            }
        )

    # -------------------------------------------------
    # 2. ASSIGN CROSS-ENCODER RANK
    # -------------------------------------------------

    ce_sorted = sorted(
        ce_results,
        key=lambda x: x["ce_score"],
        reverse=True,
    )

    ce_rank = {
        item["index"]: rank
        for rank, item in enumerate(
            ce_sorted,
            start=1,
        )
    }

    # =================================================
    # DIAGNOSTIC — CROSS-ENCODER RANKING
    # =================================================

    print("\n" + "-" * 80)
    print("CROSS-ENCODER RANKING")
    print("-" * 80)

    for rank, item in enumerate(
        ce_sorted,
        start=1,
    ):

        doc = item["doc"]
        metadata = doc.metadata or {}

        print(
            f"Rank={rank:02d} | "
            f"OriginalIndex={item['index']:02d} | "
            f"Source={metadata.get('source')} | "
            f"Page={metadata.get('page')} | "
            f"Type={metadata.get('chunk_type', 'text')} | "
            f"CE={item['ce_score']:.4f}"
        )

    # -------------------------------------------------
    # 3. CLIP SCORES FOR IMAGE CHUNKS
    # -------------------------------------------------

    image_results = []

    try:

        query_clip_embedding = encode_query_clip(
            query
        )

        query_norm = np.linalg.norm(
            query_clip_embedding
        )

        if query_norm > 0:

            query_clip_embedding = (
                query_clip_embedding / query_norm
            )

        model, preprocess = get_clip_model()

        for item in ce_results:

            doc = item["doc"]

            if (
                doc.metadata.get("chunk_type")
                != "image"
            ):
                continue

            image_path = doc.metadata.get(
                "image_path"
            )

            if (
                not image_path
                or not os.path.exists(image_path)
            ):
                continue

            try:

                img = Image.open(
                    image_path
                ).convert("RGB")

                img_tensor = (
                    preprocess(img)
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.no_grad():

                    image_features = (
                        model.encode_image(
                            img_tensor
                        )
                    )

                image_embedding = (
                    image_features
                    .cpu()
                    .numpy()[0]
                )

                image_norm = np.linalg.norm(
                    image_embedding
                )

                if image_norm == 0:
                    continue

                image_embedding = (
                    image_embedding / image_norm
                )

                clip_score = float(
                    cosine_similarity(
                        [query_clip_embedding],
                        [image_embedding],
                    )[0][0]
                )

                item["clip_score"] = clip_score

                image_results.append(item)

            except Exception as e:

                print(
                    f"CLIP image reranking failed "
                    f"for '{image_path}': {e}"
                )

    except Exception as e:

        print(
            f"CLIP reranking initialization "
            f"failed: {e}"
        )

    # -------------------------------------------------
    # 4. ASSIGN CLIP RANKS
    # -------------------------------------------------

    clip_sorted = sorted(
        image_results,
        key=lambda x: x["clip_score"],
        reverse=True,
    )

    clip_rank = {
        item["index"]: rank
        for rank, item in enumerate(
            clip_sorted,
            start=1,
        )
    }

    # =================================================
    # DIAGNOSTIC — CLIP IMAGE RANKING
    # =================================================

    print("\n" + "-" * 80)
    print("CLIP IMAGE RANKING")
    print("-" * 80)

    if not clip_sorted:

        print(
            "No image candidates received a valid CLIP score."
        )

    else:

        for rank, item in enumerate(
            clip_sorted,
            start=1,
        ):

            doc = item["doc"]
            metadata = doc.metadata or {}

            print(
                f"Rank={rank:02d} | "
                f"OriginalIndex={item['index']:02d} | "
                f"Source={metadata.get('source')} | "
                f"Page={metadata.get('page')} | "
                f"Image={metadata.get('image_path', 'N/A')} | "
                f"CLIP={item['clip_score']:.4f}"
            )

    # -------------------------------------------------
    # 5. RECIPROCAL RANK FUSION
    # -------------------------------------------------

    RRF_K = 60

    final_scores = []

    for item in ce_results:

        index = item["index"]
        doc = item["doc"]

        semantic_rank = ce_rank[index]

        # Every document receives semantic evidence.
        fusion_score = (
            1.0 / (
                RRF_K + semantic_rank
            )
        )

        # Images may additionally receive visual evidence.
        if index in clip_rank:

            visual_rank = clip_rank[index]

            fusion_score += (
                1.0 / (
                    RRF_K + visual_rank
                )
            )

        final_scores.append(
            (
                fusion_score,
                item["ce_score"],
                item["clip_score"],
                doc,
            )
        )

    # -------------------------------------------------
    # 6. SORT
    # -------------------------------------------------

    final_scores.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    # =================================================
    # 7. COMPLETE RERANK DIAGNOSTIC
    # =================================================

    print("\n" + "=" * 80)
    print("STEP 3A — COMPLETE MULTIMODAL RERANK RESULT")
    print("=" * 80)

    for rank, (
        fusion_score,
        ce_score,
        clip_score,
        doc,
    ) in enumerate(
        final_scores,
        start=1,
    ):

        metadata = doc.metadata or {}

        clip_display = (
            f"{clip_score:.4f}"
            if clip_score is not None
            else "N/A"
        )

        print(
            f"Rank={rank:02d} | "
            f"Type={metadata.get('chunk_type', 'text')} | "
            f"Source={metadata.get('source')} | "
            f"Page={metadata.get('page')} | "
            f"RRF={fusion_score:.6f} | "
            f"CE={ce_score:.4f} | "
            f"CLIP={clip_display}"
        )

        preview = (
            (doc.page_content or "")
            .replace("\n", " ")
            .strip()
        )

        print(
            f"    Content={preview[:180]}"
        )

    print("=" * 80)

    # -------------------------------------------------
    # 8. REMOVE DUPLICATES
    # -------------------------------------------------

    seen = set()
    results = []

    for _, _, _, doc in final_scores:

        key = (
            doc.metadata.get("image_path")
            or (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content[:200],
            )
        )

        if key in seen:
            continue

        seen.add(key)

        results.append(doc)

        if len(results) >= top_k:
            break

    # =================================================
    # DIAGNOSTIC — FINAL TOP-K
    # =================================================

    print("\n" + "=" * 80)
    print("STEP 3A — FINAL RERANK TOP-K")
    print("=" * 80)

    print(
        f"Requested top_k: {top_k}"
    )

    print(
        f"Returned documents: {len(results)}"
    )

    for rank, doc in enumerate(
        results,
        start=1,
    ):

        metadata = doc.metadata or {}

        print(
            f"{rank:02d}. "
            f"Source={metadata.get('source')} | "
            f"Page={metadata.get('page')} | "
            f"Type={metadata.get('chunk_type', 'text')} | "
            f"Image={metadata.get('image_path', 'N/A')}"
        )

    print("=" * 80)

    return results