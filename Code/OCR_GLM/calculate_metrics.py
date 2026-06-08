import argparse
import json
from collections import Counter
from pathlib import Path


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("The input JSON must contain a list.")
    return data


def normalize_label(value):
    text = str(value).strip().lower()
    if text in {"0", "true", "match", "matched"}:
        return 0
    if text in {"1", "false", "mismatch", "not matched"}:
        return 1
    raise ValueError(f"Unsupported label: {value!r}")


def prediction(record, threshold):
    if record.get("result") is not None:
        return normalize_label(record["result"])
    if record.get("final_match") is not None:
        return 0 if bool(record["final_match"]) else 1
    return 0 if float(record.get("final_score", 0.0)) >= threshold else 1


def class_metrics(y_true, y_pred, label):
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
    support = sum(t == label for t in y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR+GLM pairwise result files.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = read_records(args.input)
    y_true = [normalize_label(row["label"]) for row in records]
    y_pred = [prediction(row, args.threshold) for row in records]
    per_class = {str(label): class_metrics(y_true, y_pred, label) for label in (0, 1)}
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    report = {
        "samples": len(records),
        "label_counts": dict(Counter(y_true)),
        "prediction_counts": dict(Counter(y_pred)),
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_precision": sum(v["precision"] for v in per_class.values()) / 2,
        "macro_recall": sum(v["recall"] for v in per_class.values()) / 2,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 2,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
