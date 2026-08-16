from pathlib import Path
import argparse
import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.verification.strategies.nli import predict_nli
from src.verification.strategies.nli_model import DEFAULT_MODEL

LABELS = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]


DATASET_PATH = (
    Path(__file__).parent
    / "datasets"
    / "gold_nli_dataset_v6_100_examples.csv"
)


def main():

    parser = argparse.ArgumentParser(
        description="Run the NLI benchmark."
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model key from NLI_MODELS",
    )

    args = parser.parse_args()

    df = pd.read_csv(DATASET_PATH)

    correct = 0
    incorrect = 0

    y_true = []
    y_pred = []

    # ---------------------------------------------------------
    # Start Timer
    # ---------------------------------------------------------

    start_time = time.perf_counter()

    print("=" * 70)
    print("NLI BENCHMARK")
    print("=" * 70)

    print(f"Model: {args.model}")

    for _, row in df.iterrows():

        print(f"\nExample {row['id']}")
        print("-" * 60)

        print("Claim:")
        print(row["claim"])

        print()

        print("Evidence:")
        print(row["evidence"])

        print()

        expected = row["label"]

        prediction, confidence = predict_nli(
            row["claim"],
            row["evidence"],
            model_name=args.model,
        )

        predicted = prediction.name

        y_true.append(expected)
        y_pred.append(predicted)

        print(f"Expected   : {expected}")
        print(f"Predicted  : {predicted}")
        print(f"Confidence : {confidence:.4f}")

        if predicted == expected:

            print("\n✓ Correct")
            correct += 1

        else:

            print("\n✗ Incorrect")
            incorrect += 1

        print("-" * 60)

    # ---------------------------------------------------------
    # Stop Timer
    # ---------------------------------------------------------

    end_time = time.perf_counter()
    total_runtime = end_time - start_time

    total = correct + incorrect
    accuracy = (correct / total) * 100

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Model          : {args.model}")
    print(f"Total Examples : {total}")
    print(f"Correct        : {correct}")
    print(f"Incorrect      : {incorrect}")
    print(f"Accuracy       : {accuracy:.2f}%")
    print(f"Runtime (s)    : {total_runtime:.2f}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Per-class Precision / Recall / F1
    #
    # Raw accuracy on a 3-way task can hide a real weakness in one
    # class (e.g. CONTRADICTION), especially when that class is a
    # minority in the gold set. Report the full breakdown so this
    # can't be papered over by a single headline number.
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS PRECISION / RECALL / F1")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("CONFUSION MATRIX (rows=expected, cols=predicted)")
    print(f"{'':>15}" + "".join(f"{l[:11]:>13}" for l in LABELS))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    for label, row in zip(LABELS, cm):
        print(f"{label:>15}" + "".join(f"{v:>13}" for v in row))
    print("=" * 70)


if __name__ == "__main__":
    main()