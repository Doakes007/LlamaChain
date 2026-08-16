"""
Verification-method ablation: threshold-only vs. NLI-based claim verification.

Section 6.6 of PAPER_NOTES.md, cheapest-first version.

Design goal: compare the two verification *strategies* (lexical-overlap
threshold vs. NLI) against the exact same (answer, retrieved evidence) pair
for every one of the 132 benchmark questions, without paying for 132 fresh
LLM generations again.

How it avoids re-running the LLM:
    The already-completed `post_verification_full132_v2` run saved the full,
    untruncated generated answer for every question (the "RAG ANSWER" block
    in each transcript). This script monkeypatches `get_llm` inside
    `src.rag.pipeline` so that `ask_question()` runs its real retrieval,
    reranking, and context-construction code unmodified, but returns the
    cached answer text instead of calling Ollama. That reproduces the exact
    final_docs the production pipeline would use today, at a fraction of the
    cost (no 7B generation, ~132 x 60s saved).

    It also monkeypatches `process_answer` so that, for each question, BOTH
    verification methods run against the identical final_docs/answer pair
    that real `ask_question()` computed -- a controlled, apples-to-apples
    ablation, not two separate runs that could see different retrieval.

Usage:
    python -m src.evaluation.run_verification_ablation
    python -m src.evaluation.run_verification_ablation --only D-04,BERT-09
"""

import argparse
import contextlib
import csv
import io
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN_DIR = ROOT / "src" / "evaluation" / "results" / "post_verification_full132_v2"
RESULTS_ROOT = ROOT / "src" / "evaluation" / "results"

sys.path.insert(0, str(ROOT))

from src.evaluation.run_benchmark import load_questions, build_chain  # noqa: E402


def extract_cached_answer(question_id: str) -> str | None:
    """Pull the full, untruncated RAG ANSWER text out of an already-completed
    transcript, so this run doesn't need to call the LLM again."""

    path = SOURCE_RUN_DIR / f"{question_id}.txt"
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="replace")

    marker = "=" * 100 + "\nRAG ANSWER\n" + "=" * 100 + "\n"
    idx = text.find(marker)
    if idx == -1:
        return None

    rest = text[idx + len(marker):]
    end = rest.find("\nIMAGE PATHS:")
    answer = rest[:end] if end != -1 else rest
    return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="Threshold vs NLI verification ablation")
    parser.add_argument("--only", default=None,
                         help="Comma-separated question_ids to restrict to.")
    parser.add_argument("--run-id", default="verification_ablation")
    args = parser.parse_args()

    only = args.only.split(",") if args.only else None
    rows = load_questions("132", only)
    if not rows:
        print("No questions matched.")
        return

    out_dir = RESULTS_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ABLATION] Questions: {len(rows)}")
    print("[ABLATION] Building chain (loading embeddings + Chroma index)...")
    chain = build_chain()
    print("[ABLATION] Chain ready.\n")

    import src.rag.pipeline as pipeline_module
    import src.rag.summarizer as summarizer_module

    original_process_answer = pipeline_module.process_answer
    # ask_question() does `from src.rag.summarizer import get_llm` as a LOCAL
    # import inside the function body, re-resolved on every call -- so
    # patching the attribute on the summarizer module (not pipeline_module)
    # is what actually takes effect.
    original_get_llm = summarizer_module.get_llm

    captured = {"pairs": None}

    class _FakeResponse:
        def __init__(self, content):
            self.content = content

    class _FakeLLM:
        def invoke(self, prompt):
            return _FakeResponse(captured["cached_answer"])

    def _fake_get_llm(kind):
        return _FakeLLM()

    def _dual_process_answer(answer, final_docs, top_k=3, method="nli"):
        nli_result = original_process_answer(answer, final_docs, top_k=top_k, method="nli")
        threshold_result = original_process_answer(answer, final_docs, top_k=top_k, method="threshold")
        captured["pairs"] = (nli_result, threshold_result)
        # ask_question()'s own STAGE 4 printing/return uses whatever this
        # returns -- give it the nli result so its console output/behavior
        # is unchanged from production.
        return nli_result

    summarizer_module.get_llm = _fake_get_llm
    pipeline_module.process_answer = _dual_process_answer

    claim_rows = []
    per_question_rows = []
    skipped = []

    try:
        for i, row in enumerate(rows, start=1):
            qid = row["question_id"]
            question = row["question"]

            cached_answer = extract_cached_answer(qid)
            if not cached_answer:
                print(f"[{i}/{len(rows)}] {qid} -- no cached answer found, skipping")
                skipped.append(qid)
                continue

            captured["cached_answer"] = cached_answer
            captured["pairs"] = None

            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    pipeline_module.ask_question(chain, question)
            except Exception as e:
                print(f"[{i}/{len(rows)}] {qid} -- ERROR: {e}")
                skipped.append(qid)
                continue

            if captured["pairs"] is None:
                print(f"[{i}/{len(rows)}] {qid} -- verification did not run, skipping")
                skipped.append(qid)
                continue

            nli_result, threshold_result = captured["pairs"]

            n_units = len(nli_result.verification_units)
            n_supported_nli = sum(
                1 for u in nli_result.verification_units
                if u.verification_status.value == "SUPPORTED"
            )
            n_supported_th = sum(
                1 for u in threshold_result.verification_units
                if u.verification_status.value == "SUPPORTED"
            )
            n_agree = sum(
                1 for a, b in zip(nli_result.verification_units, threshold_result.verification_units)
                if a.verification_status.value == b.verification_status.value
            )

            per_question_rows.append({
                "question_id": qid,
                "document": row.get("document", ""),
                "n_claims": n_units,
                "nli_supported": n_supported_nli,
                "threshold_supported": n_supported_th,
                "agreement_rate": round(n_agree / n_units, 3) if n_units else "",
            })

            for a, b in zip(nli_result.verification_units, threshold_result.verification_units):
                claim_rows.append({
                    "question_id": qid,
                    "document": row.get("document", ""),
                    "claim_id": a.id,
                    "claim_text": a.text,
                    "nli_status": a.verification_status.value,
                    "nli_label": a.nli_label.value,
                    "nli_confidence": a.confidence,
                    "nli_citation": a.citation or "",
                    "threshold_status": b.verification_status.value,
                    "threshold_confidence": b.confidence,
                    "threshold_citation": b.citation or "",
                    "agree": a.verification_status.value == b.verification_status.value,
                })

            print(f"[{i}/{len(rows)}] {qid} -- {n_units} claims, "
                  f"nli_supported={n_supported_nli} threshold_supported={n_supported_th}")

    finally:
        summarizer_module.get_llm = original_get_llm
        pipeline_module.process_answer = original_process_answer

    # ---- write claim-level CSV ----
    claims_path = out_dir / "claims.csv"
    with open(claims_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "question_id", "document", "claim_id", "claim_text",
            "nli_status", "nli_label", "nli_confidence", "nli_citation",
            "threshold_status", "threshold_confidence", "threshold_citation",
            "agree",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(claim_rows)

    # ---- write per-question CSV ----
    per_q_path = out_dir / "per_question.csv"
    with open(per_q_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["question_id", "document", "n_claims", "nli_supported",
                      "threshold_supported", "agreement_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_question_rows)

    # ---- aggregate summary ----
    total_claims = len(claim_rows)
    nli_counts = {}
    th_counts = {}
    for r in claim_rows:
        nli_counts[r["nli_status"]] = nli_counts.get(r["nli_status"], 0) + 1
        th_counts[r["threshold_status"]] = th_counts.get(r["threshold_status"], 0) + 1
    n_agree_total = sum(1 for r in claim_rows if r["agree"])

    summary_lines = [
        f"VERIFICATION ABLATION: threshold vs. NLI",
        f"Questions processed: {len(per_question_rows)}  (skipped: {len(skipped)})",
        f"Total claims: {total_claims}",
        "",
        "Status distribution (claim-level):",
        f"{'Status':<22}{'NLI':>10}{'Threshold':>12}",
    ]
    for status in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "UNKNOWN"):
        summary_lines.append(
            f"{status:<22}{nli_counts.get(status, 0):>10}{th_counts.get(status, 0):>12}"
        )
    summary_lines += [
        "",
        f"Claim-level agreement (same status): {n_agree_total}/{total_claims} "
        f"({n_agree_total/total_claims:.1%})" if total_claims else "Claim-level agreement: N/A",
        "",
        f"Skipped question IDs: {', '.join(skipped) if skipped else '(none)'}",
    ]
    summary_path = out_dir / "SUMMARY.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8", errors="replace")

    print("\n" + "\n".join(summary_lines))
    print(f"\n[ABLATION] Written to {out_dir}")


if __name__ == "__main__":
    main()
