import sys
import argparse
import pandas as pd

from predict_main import predict


CSV_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\email-spam-detection\enron.csv"
SAMPLE_SIZE = 1000
RANDOM_SEED = 42
MAX_WRONG_SAMPLES = 5


def _compact(text: str, max_len: int = 220) -> str:
    return " ".join(str(text).split())[:max_len]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current spam model")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on full dataset instead of sampling",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Sample size when not using --full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Model objects are loaded once in predict_main at import-time.
    # predict(...) reuses those globals, so no model reload happens in this loop.
    df = pd.read_csv(CSV_PATH)

    texts = (df["Subject"].fillna("") + " " + df["Message"].fillna("")).astype(str)
    labels = df["Spam/Ham"].astype(str).str.lower().map({"spam": 1, "ham": 0})

    valid = labels.notna()
    eval_df = pd.DataFrame({
        "text": texts[valid].reset_index(drop=True),
        "actual": labels[valid].astype(int).reset_index(drop=True),
    })

    if len(eval_df) == 0:
        print("No valid emails found in dataset.")
        return

    if not args.full and len(eval_df) > args.sample_size:
        eval_df = eval_df.sample(n=args.sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

    total = len(eval_df)
    preds = [0] * total
    scores = [0.0] * total
    wrong_samples = []

    wrong = 0
    false_positives = 0
    false_negatives = 0

    for idx, row in enumerate(eval_df.itertuples(index=False), start=1):
        # Visible progress for each processed mail: 1/1000, 2/1000, ...
        sys.stdout.write(f"\rProcessing: {idx}/{total}")
        sys.stdout.flush()

        out = predict(row.text)
        pred = 1 if out["label"] == "SPAM" else 0
        score = float(out["score"])

        preds[idx - 1] = pred
        scores[idx - 1] = score

        if pred != row.actual:
            wrong += 1
            if row.actual == 0 and pred == 1:
                false_positives += 1
            elif row.actual == 1 and pred == 0:
                false_negatives += 1

            if len(wrong_samples) < MAX_WRONG_SAMPLES:
                wrong_samples.append({
                    "actual": "SPAM" if row.actual == 1 else "HAM",
                    "predicted": "SPAM" if pred == 1 else "HAM",
                    "score": score,
                    "text": _compact(row.text),
                })

    print("\n")
    print(f"TOTAL_EMAILS={total}")
    print(f"WRONGLY_CLASSIFIED={wrong}")
    print(f"FALSE_POSITIVES={false_positives}")
    print(f"FALSE_NEGATIVES={false_negatives}")

    print("\nSAMPLE_WRONGLY_CLASSIFIED_EMAILS=")
    for i in range(1, MAX_WRONG_SAMPLES + 1):
        if i <= len(wrong_samples):
            item = wrong_samples[i - 1]
            print(
                f"{i}. actual={item['actual']} predicted={item['predicted']} "
                f"score={item['score']:.3f} text={item['text']}"
            )
        else:
            print(f"{i}. N/A")


if __name__ == "__main__":
    main()
