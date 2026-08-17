"""
compute_retrieval_metrics.py

Standard IR ranking metrics (Recall@K, Precision@K, MRR) for the hybrid
retrieval stage, computed as a page-level proxy against the existing
132-question ground truth -- no new annotation, no new pipeline runs.

Why a proxy, stated up front (see PAPER_NOTES.md Limitations): the ground
truth here is a (document, cited page range) per question, not an
exhaustive human relevance judgment over every candidate chunk. A
retrieved chunk is treated as "relevant" if it comes from the cited
document AND its page falls inside the cited page range. This is a
standard, defensible proxy for page-cited QA benchmarks, but it is not
the same as chunk-level human relevance annotation, and undercounts
chunks that are genuinely relevant despite sitting just outside a loosely
cited range.

Source data: the RRF FUSION RANKING block already printed and saved in
every forensic transcript under
    src/evaluation/results/post_verification_full132_v2/<question_id>.txt
This is the ranked candidate pool the hybrid retriever (dense MMR + BM25,
fused) produces before the k=8 cutoff and before multimodal reranking --
i.e. this measures the retrieval component itself, not the final
reranked context.

Usage:
    python -m src.evaluation.compute_retrieval_metrics
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT_CSV = ROOT / "src" / "evaluation" / "benchmark" / "evaluation_ground_truth_FINAL.csv"
RUN_DIR = ROOT / "src" / "evaluation" / "results" / "post_verification_full132_v2"

K_VALUES = [1, 5, 8, 10]

RRF_LINE = re.compile(
    r"^\d+\.\s+RRF=([\d.]+)\s+\|\s+DenseRank=\S+\s+\|\s+BM25Rank=\S+\s+\|\s+"
    r"Source=(.+?)\s+\|\s+Page=(\S+)\s+\|\s+Type=(\S+)",
)


# =====================================================
# Document-name normalization
#
# Same fix as the earlier before/after retrieval regression analysis:
# the ground-truth CSV's NLP-Q rows cite "NLP Paper(2).pdf" while the
# actual indexed/retrieved source is "NLP Paper.pdf".
# =====================================================

def normalize_docname(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\(\d+\)", "", name)   # strip "(2)"-style suffixes
    name = re.sub(r"\.pdf$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# =====================================================
# Ground-truth page-range parsing
#
# Handles the range of formats actually present in
# evaluation_ground_truth_FINAL.csv, e.g.:
#   "Self-RAG.pdf — Pages 1–2"
#   "Self-RAG.pdf — Pages 1, 4–5"
#   "Vision Transformer.pdf, pp. 3, 7"
#   "CNN Paper.pdf, p. 2–3 (Table 1 and Figure 3)"
#   "Donut.pdf, p. 11, Table 3"
# =====================================================

RANGE_RE = re.compile(r"(\d+)\s*(?:[-–—]\s*(\d+))?")


def parse_page_range(source_text: str) -> set[int]:
    # Strip document name / trailing annotations, keep the digits-and-dashes
    # portion that actually refers to pages.
    text = re.split(r"\.pdf", source_text, flags=re.IGNORECASE)
    text = text[1] if len(text) > 1 else source_text
    text = re.sub(r"\([^)]*\)", " ", text)  # drop "(Table 3)"-style notes
    # Drop "Table 3", "Figure 4", "Figure 3(c)", "Figure 6/7" -- these cite a
    # table/figure NUMBER, not a page number, and must not be matched below.
    text = re.sub(r"\b(?:Table|Figure|Fig\.?)\s*\d+[a-zA-Z]?(?:\s*/\s*\d+)?", " ", text)

    pages: set[int] = set()
    for match in RANGE_RE.finditer(text):
        a = int(match.group(1))
        b = int(match.group(2)) if match.group(2) else a
        lo, hi = min(a, b), max(a, b)
        if hi - lo > 20:
            # Guard against accidental garbage matches (e.g. a stray large
            # number from something not actually a page reference).
            continue
        pages.update(range(lo, hi + 1))
    return pages


# =====================================================
# Transcript parsing
# =====================================================

def parse_ranked_candidates(transcript_text: str):
    """Return the RRF-fusion-ranked candidate list as
    [(source_normalized, page_int_or_None, type), ...], in rank order."""

    idx = transcript_text.find("[RRF FUSION RANKING]")
    if idx == -1:
        return None

    block = transcript_text[idx:]
    end = block.find("\n\n", block.find("---\n") + 1) if "---\n" in block else -1
    # Just scan line by line until a non-matching, non-dash line ends the block.
    candidates = []
    for line in block.splitlines():
        m = RRF_LINE.match(line.strip())
        if m:
            _, source, page, ctype = m.groups()
            try:
                page_int = int(page)
            except ValueError:
                page_int = None
            candidates.append((normalize_docname(source), page_int, ctype))
        elif candidates and line.strip().startswith("---"):
            break
    return candidates


# =====================================================
# Metrics
# =====================================================

def compute_metrics_for_question(candidates, gt_doc: str, gt_pages: set[int]):
    relevance = [
        1 if (doc == gt_doc and page in gt_pages) else 0
        for doc, page, _ in candidates
    ]

    result = {}
    for k in K_VALUES:
        top_k = relevance[:k]
        result[f"recall@{k}"] = 1.0 if sum(top_k) > 0 else 0.0
        result[f"precision@{k}"] = (sum(top_k) / k) if top_k else 0.0

    rr = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            rr = 1.0 / rank
            break
    result["rr"] = rr

    return result


def main():
    with open(GT_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    per_question = []
    skipped = []

    for row in rows:
        qid = row["question_id"]
        transcript_path = RUN_DIR / f"{qid}.txt"
        if not transcript_path.exists():
            skipped.append((qid, "no transcript"))
            continue

        text = transcript_path.read_text(encoding="utf-8", errors="replace")
        candidates = parse_ranked_candidates(text)
        if not candidates:
            skipped.append((qid, "no RRF block (unscoped / error case)"))
            continue

        gt_doc = normalize_docname(row["document"])
        gt_pages = parse_page_range(row["ground_truth_source"])
        if not gt_pages:
            skipped.append((qid, "unparsed page range"))
            continue

        metrics = compute_metrics_for_question(candidates, gt_doc, gt_pages)
        metrics["question_id"] = qid
        metrics["n_candidates"] = len(candidates)
        per_question.append(metrics)

    n = len(per_question)
    print("=" * 70)
    print("RETRIEVAL RANKING METRICS -- page-level proxy relevance")
    print("=" * 70)
    print(f"Run              : post_verification_full132_v2")
    print(f"Questions scored : {n} / {len(rows)}")
    if skipped:
        print(f"Skipped          : {len(skipped)} -> {skipped}")
    print()

    print(f"{'Metric':<14}{'Value':>10}")
    for k in K_VALUES:
        avg_r = sum(m[f"recall@{k}"] for m in per_question) / n
        avg_p = sum(m[f"precision@{k}"] for m in per_question) / n
        print(f"Recall@{k:<7}{avg_r:>10.4f}")
        print(f"Precision@{k:<4}{avg_p:>10.4f}")
    mrr = sum(m["rr"] for m in per_question) / n
    print(f"{'MRR':<14}{mrr:>10.4f}")
    print("=" * 70)

    out_path = RUN_DIR.parent / "retrieval_ranking_metrics.csv"
    fieldnames = (
        ["question_id", "n_candidates"]
        + [f"recall@{k}" for k in K_VALUES]
        + [f"precision@{k}" for k in K_VALUES]
        + ["rr"]
    )
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in per_question:
            writer.writerow({k: m.get(k, "") for k in fieldnames})
    print(f"\nPer-question detail written to: {out_path}")


if __name__ == "__main__":
    main()
