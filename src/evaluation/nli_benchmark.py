from pathlib import Path
import argparse
import time

import pandas as pd

from src.verification.strategies.nli import predict_nli
from src.verification.strategies.nli_model import DEFAULT_MODEL


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


if __name__ == "__main__":
    main()