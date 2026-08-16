"""
Standalone benchmark runner for LlamaChain.

Loads the persisted Chroma index the same way app.py does, builds the
retrieval chain, then replays ground-truth questions through
ask_question() one at a time -- automating the "paste a question into
the app, copy the terminal output" workflow that was previously done
by hand.

Nothing about the RAG pipeline itself is touched. This script only
calls the existing ask_question() and captures/parses its stdout.

Usage (run from the repo root, with the project venv active):

    python -m src.evaluation.run_benchmark
    python -m src.evaluation.run_benchmark --set 39
    python -m src.evaluation.run_benchmark --set 132
    python -m src.evaluation.run_benchmark --only D-14,L7,VIT-13
    python -m src.evaluation.run_benchmark --run-id 2026-08-15_scope-fix --force
    python -m src.evaluation.run_benchmark --list

Output layout:

    src/evaluation/results/<run_id>/
        <question_id>.txt   -- full forensic transcript + answer, one per question
        summary.csv          -- one row per question with parsed key fields
        SUMMARY.txt           -- quick skim: pass/fail-style digest
"""

import argparse
import contextlib
import csv
import io
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Make stdout/stderr tolerant of non-ASCII characters (em dashes, arrows)
# regardless of the Windows console code page.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[2]

GT_CSV = ROOT / "src" / "evaluation" / "benchmark" / "evaluation_ground_truth_FINAL.csv"
CASES_CSV = (
    ROOT / "src" / "evaluation" / "benchmark" / "Retrieval deep analysis"
    / "STEP2B_FINAL_39_CASE_ROOT_CAUSE_DATASET.csv"
)
CHROMA_DIR = ROOT / "chroma_db"
RESULTS_ROOT = ROOT / "src" / "evaluation" / "results"


# =====================================================
# Question loading
# =====================================================

def load_questions(question_set: str, only: list[str] | None):
    """Return a list of dicts: question_id, document, question, question_type,
    ground_truth_answer, ground_truth_source -- pulled from the frozen 132-row
    ground truth CSV, optionally filtered to the 39 confirmed forensic cases."""

    with open(GT_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if question_set == "39":
        with open(CASES_CSV, encoding="utf-8-sig", newline="") as f:
            case_ids = {row["question_id"] for row in csv.DictReader(f)}
        rows = [r for r in rows if r["question_id"] in case_ids]

    if only:
        wanted = {q.strip().upper() for q in only}
        rows = [r for r in rows if r["question_id"].strip().upper() in wanted]

    return rows


# =====================================================
# Chain construction (mirrors app.py, no Streamlit)
# =====================================================

def build_chain():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from src.rag import build_retrieval_chain

    if not CHROMA_DIR.exists():
        raise SystemExit(
            f"Chroma index not found at {CHROMA_DIR}. "
            "Run the app once and ingest documents before running the benchmark."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    vectorstore = Chroma(
        collection_name="LlamaChainDocs",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    return build_retrieval_chain(vectorstore)


# =====================================================
# Output capture
# =====================================================

class Tee(io.TextIOBase):
    """Writes to multiple streams at once -- lets the run still scroll live
    in the terminal (like the old manual workflow) while also being saved
    to a buffer for the transcript file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            try:
                stream.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def extract(pattern: str, text: str, default: str = ""):
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else default


def parse_transcript(transcript: str) -> dict:
    """Pull the key forensic fields out of a captured transcript without
    changing anything about how ask_question() logs them."""

    explicit_figure_query = "EXPLICIT FIGURE QUERY DETECTED" in transcript

    figure_diagnostic = extract(r"(\[FIGURE DIAGNOSTIC\].*)", transcript)
    present_in_final = extract(r"Present In Final\s*:\s*(True|False)", transcript)

    return {
        "document_scope": extract(r"^Document Scope\s*:\s*(.*)$", transcript),
        "retrieved_count_stage1": extract(r"Retrieved Count\s*:\s*(\d+)", transcript),
        "explicit_figure_query": explicit_figure_query,
        "present_in_final": present_in_final,
        "figure_diagnostic": figure_diagnostic,
        "final_context_docs": extract(r"Final Context Docs\s*:\s*(\d+)", transcript),
        "verification_units": extract(r"Verification Units\s*:\s*(\d+)", transcript),
        "verification_supported": extract(r"^Supported\s*:\s*(\d+)", transcript),
        "verification_partial": extract(r"Partially Supported:\s*(\d+)", transcript),
        "verification_unsupported": extract(r"^Unsupported\s*:\s*(\d+)", transcript),
        "verification_unknown": extract(r"^Unknown\s*:\s*(\d+)", transcript),
        "groundedness_ratio": extract(r"Groundedness Ratio\s*:\s*([\d.]+|N/A)", transcript),
    }


# =====================================================
# Single-question run
# =====================================================

def run_one(chain, row: dict, out_dir: Path,
            max_retries: int = 2, retry_delay: float = 20.0) -> dict:
    question_id = row["question_id"]
    question = row["question"]

    buffer = io.StringIO()
    tee = Tee(sys.__stdout__, buffer)

    from src.rag import ask_question

    answer = None
    image_paths = []
    verification_result = None
    error = None

    start = time.time()
    for attempt in range(1, max_retries + 2):  # e.g. 1 try + 2 retries = 3 total
        try:
            with contextlib.redirect_stdout(tee):
                answer, image_paths, verification_result = ask_question(chain, question)
            error = None
            break
        except Exception:
            error = traceback.format_exc()
            print(
                f"[RETRY] {question_id} attempt {attempt} failed:\n{error}",
                file=sys.__stdout__,
            )
            if attempt <= max_retries:
                # Likely a transient Ollama memory-allocation failure while
                # loading the model -- give it a moment to settle and retry
                # rather than losing the whole question.
                print(
                    f"[RETRY] Waiting {retry_delay}s before retrying "
                    f"{question_id}...",
                    file=sys.__stdout__,
                )
                time.sleep(retry_delay)
    elapsed = round(time.time() - start, 1)

    transcript = buffer.getvalue()
    parsed = parse_transcript(transcript)

    # ---- per-question transcript file ----
    lines = [
        "=" * 100,
        f"QUESTION_ID   : {question_id}",
        f"DOCUMENT      : {row.get('document', '')}",
        f"QUESTION_TYPE : {row.get('question_type', '')}",
        f"ELAPSED_SEC   : {elapsed}",
        f"ERROR         : {'YES' if error else 'no'}",
        "=" * 100,
        "",
        "QUESTION:",
        question,
        "",
        "GROUND TRUTH ANSWER:",
        row.get("ground_truth_answer", ""),
        "",
        "GROUND TRUTH SOURCE:",
        row.get("ground_truth_source", ""),
        "",
        "=" * 100,
        "FORENSIC TRANSCRIPT",
        "=" * 100,
        transcript,
    ]

    if error:
        lines += ["=" * 100, "EXCEPTION", "=" * 100, error]
    else:
        lines += [
            "=" * 100,
            "RAG ANSWER",
            "=" * 100,
            answer or "",
            "",
            "IMAGE PATHS:",
            *[str(p) for p in (image_paths or [])],
        ]

    (out_dir / f"{question_id}.txt").write_text(
        "\n".join(lines), encoding="utf-8", errors="replace"
    )

    return {
        "question_id": question_id,
        "document": row.get("document", ""),
        "question_type": row.get("question_type", ""),
        "question": question,
        "error": bool(error),
        "elapsed_sec": elapsed,
        **parsed,
        "rag_answer": (answer or "")[:500],
        "ground_truth_answer": (row.get("ground_truth_answer", ""))[:500],
        "image_count": len(image_paths or []),
    }


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="LlamaChain benchmark runner")
    parser.add_argument("--set", choices=["39", "132"], default="39",
                         help="Question set: the 39 confirmed forensic cases "
                              "(default) or the full 132-question ground truth.")
    parser.add_argument("--only", default=None,
                         help="Comma-separated question_ids to restrict to, "
                              "e.g. D-14,L7,VIT-13")
    parser.add_argument("--run-id", default=None,
                         help="Reuse/continue a specific results folder name. "
                              "Defaults to a new UTC timestamp.")
    parser.add_argument("--force", action="store_true",
                         help="Re-run questions even if a transcript already "
                              "exists in this run-id folder.")
    parser.add_argument("--list", action="store_true",
                         help="Print the matched question IDs and exit "
                              "without running anything.")
    parser.add_argument("--retries", type=int, default=2,
                         help="Retries per question on failure (e.g. Ollama "
                              "memory allocation errors). Default 2 "
                              "(3 attempts total).")
    parser.add_argument("--retry-delay", type=float, default=20.0,
                         help="Seconds to wait before retrying a failed "
                              "question. Default 20.")
    parser.add_argument("--sleep-between", type=float, default=5.0,
                         help="Seconds to pause between questions, to let "
                              "memory settle on constrained machines. "
                              "Default 5.")
    args = parser.parse_args()

    only = args.only.split(",") if args.only else None
    rows = load_questions(args.set, only)

    if not rows:
        print("No questions matched the given filters.")
        return

    if args.list:
        for r in rows:
            print(f"{r['question_id']:<10} {r['document']:<20} {r['question']}")
        print(f"\n{len(rows)} question(s) matched.")
        return

    run_id = args.run_id or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BENCHMARK] Question set : {args.set}")
    print(f"[BENCHMARK] Questions    : {len(rows)}")
    print(f"[BENCHMARK] Run ID       : {run_id}")
    print(f"[BENCHMARK] Output dir   : {out_dir}")
    print("[BENCHMARK] Building chain (loading embeddings + Chroma index)...")

    chain = build_chain()

    print("[BENCHMARK] Chain ready. Starting run.\n")

    summary_rows = []
    for i, row in enumerate(rows, start=1):
        question_id = row["question_id"]
        transcript_path = out_dir / f"{question_id}.txt"

        if transcript_path.exists() and not args.force:
            print(f"[{i}/{len(rows)}] {question_id} -- already done, skipping "
                  f"(use --force to re-run)")
            # Re-parse the existing transcript so summary.csv stays complete
            # even on a resumed run.
            text = transcript_path.read_text(encoding="utf-8", errors="replace")
            transcript_only = text.split("FORENSIC TRANSCRIPT\n" + "=" * 100 + "\n", 1)
            transcript_only = transcript_only[1] if len(transcript_only) > 1 else text
            parsed = parse_transcript(transcript_only)
            summary_rows.append({
                "question_id": question_id,
                "document": row.get("document", ""),
                "question_type": row.get("question_type", ""),
                "question": row["question"],
                "error": "EXCEPTION" in text,
                "elapsed_sec": "",
                **parsed,
                "rag_answer": "",
                "ground_truth_answer": (row.get("ground_truth_answer", ""))[:500],
                "image_count": "",
            })
            continue

        print(f"[{i}/{len(rows)}] {question_id} -- running...")
        print("-" * 100)
        result = run_one(chain, row, out_dir,
                          max_retries=args.retries, retry_delay=args.retry_delay)
        print("-" * 100)
        print(f"[{i}/{len(rows)}] {question_id} -- done in {result['elapsed_sec']}s "
              f"(error={result['error']})\n")
        summary_rows.append(result)

        if args.sleep_between > 0 and i < len(rows):
            time.sleep(args.sleep_between)

    # ---- summary.csv ----
    summary_path = out_dir / "summary.csv"
    fieldnames = [
        "question_id", "document", "question_type", "question",
        "document_scope", "explicit_figure_query", "present_in_final",
        "figure_diagnostic", "retrieved_count_stage1", "final_context_docs",
        "verification_units", "verification_supported", "verification_partial",
        "verification_unsupported", "verification_unknown", "groundedness_ratio",
        "image_count", "error", "elapsed_sec", "rag_answer", "ground_truth_answer",
    ]
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # ---- SUMMARY.txt digest ----
    digest_path = out_dir / "SUMMARY.txt"
    digest_lines = [f"BENCHMARK RUN: {run_id}", f"Questions: {len(summary_rows)}", ""]
    for row in summary_rows:
        flag = "ERROR" if row["error"] else (
            row["figure_diagnostic"].replace("[FIGURE DIAGNOSTIC] ", "")
            if row.get("figure_diagnostic") else "ok"
        )
        digest_lines.append(
            f"{row['question_id']:<10} scope={row['document_scope']:<30} {flag}"
        )
    digest_path.write_text("\n".join(digest_lines), encoding="utf-8", errors="replace")

    print(f"\n[BENCHMARK] Done. {len(summary_rows)} question(s) written to:")
    print(f"  {out_dir}")
    print(f"  summary.csv  -- one row per question, parsed key fields")
    print(f"  SUMMARY.txt  -- quick digest")
    print(f"  <id>.txt     -- full forensic transcript per question")


if __name__ == "__main__":
    main()
