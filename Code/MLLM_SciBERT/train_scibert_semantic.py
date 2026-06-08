import argparse
import inspect
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


LOGGER = logging.getLogger(__name__)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def validate_rows(rows, mode, path):
    required = ["label"]
    required.extend(["text_a", "text_b"] if mode == "pair" else ["text"])
    for index, row in enumerate(rows):
        for key in required:
            if key not in row:
                raise ValueError(f"Missing column {key!r} in {path}, row {index}")
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"Expected label 0 or 1 in {path}, row {index}, got {row['label']!r}")


class SemanticMatchDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        max_length,
        mode,
        chunk_text_a=False,
        chunk_overlap=64,
        max_text_b_tokens=0,
        min_text_a_tokens=64,
        text_b_overlap=32,
        chunk_strategy="evidence",
        evidence_overlap=2,
    ):
        self.rows = rows
        self.mode = mode
        self.ids = []
        self.group_ids = []
        self.chunk_indices = []
        self.labels = []
        self.is_chunked = bool(chunk_text_a and mode == "pair")

        if self.is_chunked:
            self.encodings = self._build_chunked_encodings(
                rows,
                tokenizer,
                max_length,
                chunk_overlap,
                max_text_b_tokens,
                min_text_a_tokens,
                text_b_overlap,
                chunk_strategy,
                evidence_overlap,
            )
        elif mode == "pair":
            self.ids = [str(row.get("id", i)) for i, row in enumerate(rows)]
            self.group_ids = list(self.ids)
            self.chunk_indices = [0 for _ in rows]
            self.labels = [int(row["label"]) for row in rows]
            self.encodings = tokenizer(
                [str(row["text_a"]) for row in rows],
                [str(row["text_b"]) for row in rows],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        else:
            self.ids = [str(row.get("id", i)) for i, row in enumerate(rows)]
            self.group_ids = list(self.ids)
            self.chunk_indices = [0 for _ in rows]
            self.labels = [int(row["label"]) for row in rows]
            self.encodings = tokenizer(
                [str(row["text"]) for row in rows],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

    @staticmethod
    def _truncate_head_tail(tokens, max_tokens):
        if len(tokens) <= max_tokens:
            return tokens
        if max_tokens <= 0:
            return []
        head_len = max_tokens // 2
        tail_len = max_tokens - head_len
        return tokens[:head_len] + tokens[-tail_len:]

    @staticmethod
    def _chunk_tokens(tokens, chunk_size, overlap):
        if not tokens:
            return [[]]
        if len(tokens) <= chunk_size:
            return [tokens]

        chunks = []
        step = max(chunk_size - overlap, 1)
        start = 0
        while start < len(tokens):
            chunks.append(tokens[start : start + chunk_size])
            if start + chunk_size >= len(tokens):
                break
            start += step
        return chunks

    @staticmethod
    def _split_evidence_units(text):
        text = str(text).strip()
        if not text:
            return [""]

        units = []
        current = []
        i = 0
        while i < len(text):
            ch = text[i]
            current.append(ch)
            next_ch = text[i + 1] if i + 1 < len(text) else ""

            split_here = ch in "。！？；;："
            if ch in ".!?":
                # Avoid splitting decimals such as 40.2 or abbreviations too aggressively.
                prev_ch = text[i - 1] if i > 0 else ""
                split_here = not (prev_ch.isdigit() and next_ch.isdigit())

            if ch == "\n":
                split_here = True

            if split_here:
                unit = "".join(current).strip()
                if unit:
                    units.append(unit)
                current = []
            i += 1

        tail = "".join(current).strip()
        if tail:
            units.append(tail)

        # Merge very tiny units into their predecessor so punctuation-driven splits do not become too choppy.
        merged = []
        for unit in units:
            if merged and len(unit) <= 8:
                merged[-1] = f"{merged[-1]} {unit}".strip()
            else:
                merged.append(unit)
        return merged or [text]

    @classmethod
    def _chunk_evidence_units(cls, text, tokenizer, chunk_size, unit_overlap, token_overlap):
        units = cls._split_evidence_units(text)
        chunks = []
        current_tokens = []
        current_units = []

        def flush():
            if current_tokens:
                chunks.append(list(current_tokens))

        for unit in units:
            unit_tokens = tokenizer.tokenize(unit)
            if not unit_tokens:
                continue

            if len(unit_tokens) > chunk_size:
                flush()
                current_tokens.clear()
                current_units.clear()
                chunks.extend(cls._chunk_tokens(unit_tokens, chunk_size, token_overlap))
                continue

            if current_tokens and len(current_tokens) + len(unit_tokens) > chunk_size:
                flush()
                overlap_units = current_units[-unit_overlap:] if unit_overlap > 0 else []
                current_tokens = [tok for overlap_unit in overlap_units for tok in overlap_unit]
                current_units = [list(overlap_unit) for overlap_unit in overlap_units]

                while current_tokens and len(current_tokens) + len(unit_tokens) > chunk_size:
                    current_units = current_units[1:]
                    current_tokens = [tok for overlap_unit in current_units for tok in overlap_unit]

            current_tokens.extend(unit_tokens)
            current_units.append(unit_tokens)

        flush()
        return chunks or [[]]

    def _build_chunked_encodings(
        self,
        rows,
        tokenizer,
        max_length,
        chunk_overlap,
        max_text_b_tokens,
        min_text_a_tokens,
        text_b_overlap,
        chunk_strategy,
        evidence_overlap,
    ):
        input_ids = []
        attention_mask = []
        token_type_ids = []

        cls_token = tokenizer.cls_token or "[CLS]"
        sep_token = tokenizer.sep_token or "[SEP]"
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        if min_text_a_tokens <= 0:
            raise ValueError("min_text_a_tokens must be positive when chunking is enabled")
        if min_text_a_tokens > max_length - 4:
            raise ValueError("min_text_a_tokens is too large for max_length")

        full_b_budget = max_length - min_text_a_tokens - 3
        if max_text_b_tokens and max_text_b_tokens > 0:
            full_b_budget = min(full_b_budget, max_text_b_tokens)
        full_b_budget = max(1, full_b_budget)

        for row_index, row in enumerate(rows):
            sample_id = str(row.get("id", row_index))
            label = int(row["label"])

            text_a = str(row["text_a"])
            a_tokens = tokenizer.tokenize(text_a)
            b_tokens = tokenizer.tokenize(str(row["text_b"]))
            b_chunks = self._chunk_tokens(
                b_tokens,
                full_b_budget,
                min(max(text_b_overlap, 0), max(full_b_budget - 1, 0)),
            )

            chunk_index = 0
            for b_chunk_index, b_chunk in enumerate(b_chunks):
                a_budget = max_length - len(b_chunk) - 3
                if a_budget <= 0:
                    raise ValueError(
                        f"text_b chunk leaves no room for text_a. "
                        f"max_length={max_length}, len(text_b_chunk)={len(b_chunk)}"
                    )

                max_reasonable_overlap = max(a_budget // 4, 0)
                effective_overlap = min(max(chunk_overlap, 0), max_reasonable_overlap)
                if chunk_strategy == "evidence":
                    a_chunks = self._chunk_evidence_units(
                        text_a,
                        tokenizer,
                        a_budget,
                        max(evidence_overlap, 0),
                        effective_overlap,
                    )
                elif chunk_strategy == "token":
                    a_chunks = self._chunk_tokens(a_tokens, a_budget, effective_overlap)
                else:
                    raise ValueError(f"Unsupported chunk_strategy: {chunk_strategy}")

                for a_chunk_index, a_chunk in enumerate(a_chunks):
                    tokens = [cls_token] + a_chunk + [sep_token] + b_chunk + [sep_token]
                    ids = tokenizer.convert_tokens_to_ids(tokens)
                    type_ids = [0] * (len(a_chunk) + 2) + [1] * (len(b_chunk) + 1)
                    mask = [1] * len(ids)

                    pad_len = max_length - len(ids)
                    if pad_len < 0:
                        raise ValueError(f"Chunked input exceeded max_length by {-pad_len} tokens")

                    ids += [pad_id] * pad_len
                    type_ids += [0] * pad_len
                    mask += [0] * pad_len

                    input_ids.append(ids)
                    token_type_ids.append(type_ids)
                    attention_mask.append(mask)
                    self.ids.append(f"{sample_id}#b{b_chunk_index}_a{a_chunk_index}")
                    self.group_ids.append(sample_id)
                    self.chunk_indices.append(chunk_index)
                    self.labels.append(label)
                    chunk_index += 1

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return classification_metrics(labels, preds)


def classification_metrics(labels, preds):
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )
    macro_precision = float(np.mean(per_class_precision))
    macro_recall = float(np.mean(per_class_recall))
    macro_f1 = float(np.mean(per_class_f1))

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "label0_precision": float(per_class_precision[0]),
        "label0_recall": float(per_class_recall[0]),
        "label0_f1": float(per_class_f1[0]),
        "label0_support": int(per_class_support[0]),
        "label1_precision": float(per_class_precision[1]),
        "label1_recall": float(per_class_recall[1]),
        "label1_f1": float(per_class_f1[1]),
        "label1_support": int(per_class_support[1]),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def softmax_np(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def aggregate_group_predictions(logits, labels, group_ids, aggregate="mean"):
    probs = softmax_np(np.asarray(logits))
    labels = np.asarray(labels)

    grouped_probs = {}
    grouped_labels = {}
    for index, group_id in enumerate(group_ids):
        grouped_probs.setdefault(group_id, []).append(probs[index])
        grouped_labels.setdefault(group_id, int(labels[index]))

    ordered_ids = list(grouped_probs.keys())
    true_labels = []
    pred_labels = []
    prob_label_1 = []
    chunk_counts = []

    for group_id in ordered_ids:
        group_probs = np.asarray(grouped_probs[group_id])
        if aggregate == "mean":
            p1 = float(np.mean(group_probs[:, 1]))
        elif aggregate == "max_label1":
            p1 = float(np.max(group_probs[:, 1]))
        elif aggregate == "max_label0":
            p1 = float(1.0 - np.max(group_probs[:, 0]))
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")

        true_labels.append(grouped_labels[group_id])
        pred_labels.append(1 if p1 >= 0.5 else 0)
        prob_label_1.append(p1)
        chunk_counts.append(len(group_probs))

    return {
        "ids": ordered_ids,
        "true_labels": np.asarray(true_labels),
        "pred_labels": np.asarray(pred_labels),
        "prob_label_1": np.asarray(prob_label_1),
        "chunk_counts": chunk_counts,
    }


def make_compute_metrics(eval_dataset, aggregate):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if getattr(eval_dataset, "is_chunked", False):
            grouped = aggregate_group_predictions(logits, labels, eval_dataset.group_ids, aggregate)
            metrics = classification_metrics(grouped["true_labels"], grouped["pred_labels"])
            metrics["eval_samples_after_grouping"] = int(len(grouped["true_labels"]))
            metrics["eval_chunks"] = int(len(labels))
            metrics["eval_avg_chunks_per_sample"] = float(np.mean(grouped["chunk_counts"]))
            return metrics

        preds = np.argmax(logits, axis=-1)
        return classification_metrics(labels, preds)

    return compute_metrics


def class_weights(labels):
    counts = np.bincount(labels, minlength=2)
    weights = []
    total = len(labels)
    for count in counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (2.0 * count))
    return weights


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        weights = getattr(model.config, "class_weights", None)
        if weights:
            weight_tensor = torch.tensor(weights, dtype=torch.float, device=logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def make_training_args(args):
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_dir": str(args.output_dir / "logs"),
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "learning_rate": args.learning_rate,
        "fp16": torch.cuda.is_available() and not args.no_fp16,
        "report_to": "none",
        "lr_scheduler_type": "cosine",
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "dataloader_pin_memory" in signature.parameters:
        kwargs["dataloader_pin_memory"] = False

    return TrainingArguments(**kwargs)


def save_predictions(path, dataset, predictions, labels, aggregate):
    if getattr(dataset, "is_chunked", False):
        grouped = aggregate_group_predictions(predictions, labels, dataset.group_ids, aggregate)
        rows = []
        for sample_id, true_label, pred_label, prob, chunk_count in zip(
            grouped["ids"],
            grouped["true_labels"],
            grouped["pred_labels"],
            grouped["prob_label_1"],
            grouped["chunk_counts"],
        ):
            rows.append(
                {
                    "id": sample_id,
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "prob_label_1": float(prob),
                    "chunk_count": int(chunk_count),
                    "correct": int(true_label) == int(pred_label),
                }
            )
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return

    probs = torch.softmax(torch.tensor(predictions), dim=-1).numpy()
    preds = np.argmax(predictions, axis=-1)
    rows = []
    for sample_id, true_label, pred_label, prob in zip(dataset.ids, labels, preds, probs):
        rows.append(
            {
                "id": sample_id,
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "prob_label_1": float(prob[1]),
                "correct": int(true_label) == int(pred_label),
            }
        )
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def parse_args():
    root = Path.cwd()
    default_data_dir = root / "data" / "fold0"
    default_model = root / "scibert_scivocab_uncased"
    default_output_root = root / "outputs"

    parser = argparse.ArgumentParser(description="Train SciBERT for semantic matching classification.")
    parser.add_argument("--mode", choices=["pair", "text"], default="pair")
    parser.add_argument("--train-file", type=Path, default=default_data_dir / "train_scibert_pair.jsonl")
    parser.add_argument("--val-file", type=Path, default=default_data_dir / "val_scibert_pair.jsonl")
    parser.add_argument("--model-name", type=str, default=str(default_model))
    parser.add_argument("--output-dir", type=Path, default=default_output_root / "scibert_semantic_fold0")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--no-chunk-text-a",
        action="store_true",
        help="Disable text_a chunking. By default, pair mode chunks cited facts and preserves all text_b.",
    )
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument(
        "--max-text-b-tokens",
        type=int,
        default=0,
        help="Optional hard cap for text_b tokens. Default 0 means preserve all text_b, splitting it only if needed.",
    )
    parser.add_argument("--min-text-a-tokens", type=int, default=64)
    parser.add_argument("--text-b-overlap", type=int, default=32)
    parser.add_argument("--chunk-strategy", choices=["evidence", "token"], default="evidence")
    parser.add_argument("--evidence-overlap", type=int, default=2)
    parser.add_argument("--chunk-aggregate", choices=["mean", "max_label1", "max_label0"], default="mean")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "train.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    set_seed(args.seed)

    LOGGER.info("Loading data")
    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
    validate_rows(train_rows, args.mode, args.train_file)
    validate_rows(val_rows, args.mode, args.val_file)

    train_labels = [int(row["label"]) for row in train_rows]
    val_labels = [int(row["label"]) for row in val_rows]
    train_label_values, train_label_counts = np.unique(train_labels, return_counts=True)
    val_label_values, val_label_counts = np.unique(val_labels, return_counts=True)
    train_label_dist = {int(label): int(count) for label, count in zip(train_label_values, train_label_counts)}
    val_label_dist = {int(label): int(count) for label, count in zip(val_label_values, val_label_counts)}
    LOGGER.info("Train samples=%d, labels=%s", len(train_rows), train_label_dist)
    LOGGER.info("Val samples=%d, labels=%s", len(val_rows), val_label_dist)
    LOGGER.info("Field mapping: text_a=cited paper facts; text_b=originalkey author_related_content plus citing-table caption")

    LOGGER.info("Loading SciBERT from %s", args.model_name)
    tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    chunk_text_a = args.mode == "pair" and not args.no_chunk_text_a
    train_dataset = SemanticMatchDataset(
        train_rows,
        tokenizer,
        args.max_length,
        args.mode,
        chunk_text_a=chunk_text_a,
        chunk_overlap=args.chunk_overlap,
        max_text_b_tokens=args.max_text_b_tokens,
        min_text_a_tokens=args.min_text_a_tokens,
        text_b_overlap=args.text_b_overlap,
        chunk_strategy=args.chunk_strategy,
        evidence_overlap=args.evidence_overlap,
    )
    val_dataset = SemanticMatchDataset(
        val_rows,
        tokenizer,
        args.max_length,
        args.mode,
        chunk_text_a=chunk_text_a,
        chunk_overlap=args.chunk_overlap,
        max_text_b_tokens=args.max_text_b_tokens,
        min_text_a_tokens=args.min_text_a_tokens,
        text_b_overlap=args.text_b_overlap,
        chunk_strategy=args.chunk_strategy,
        evidence_overlap=args.evidence_overlap,
    )
    LOGGER.info(
        "Chunking: enabled=%s, strategy=%s, train_chunks=%d, val_chunks=%d, avg_train_chunks=%.2f, avg_val_chunks=%.2f, max_text_b_tokens=%d, min_text_a_tokens=%d, text_b_overlap=%d, evidence_overlap=%d, aggregate=%s",
        chunk_text_a,
        args.chunk_strategy,
        len(train_dataset),
        len(val_dataset),
        len(train_dataset) / max(len(train_rows), 1),
        len(val_dataset) / max(len(val_rows), 1),
        args.max_text_b_tokens,
        args.min_text_a_tokens,
        args.text_b_overlap,
        args.evidence_overlap,
        args.chunk_aggregate,
    )
    if not args.no_class_weights:
        weights = class_weights(train_dataset.labels)
        model.config.class_weights = weights
        LOGGER.info("Using chunk-level class weights: %s", weights)

    trainer = WeightedTrainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(val_dataset, args.chunk_aggregate),
    )

    LOGGER.info("Starting training")
    trainer.train()

    LOGGER.info("Saving best model to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    LOGGER.info("Running final evaluation")
    pred_output = trainer.predict(val_dataset)
    labels = pred_output.label_ids
    if getattr(val_dataset, "is_chunked", False):
        grouped = aggregate_group_predictions(
            pred_output.predictions,
            labels,
            val_dataset.group_ids,
            args.chunk_aggregate,
        )
        final_labels = grouped["true_labels"]
        final_preds = grouped["pred_labels"]
        metrics = classification_metrics(final_labels, final_preds)
        metrics["chunks"] = int(len(labels))
        metrics["samples_after_grouping"] = int(len(grouped["true_labels"]))
        metrics["avg_chunks_per_sample"] = float(np.mean(grouped["chunk_counts"]))
    else:
        final_labels = labels
        final_preds = np.argmax(pred_output.predictions, axis=-1)
        metrics = classification_metrics(final_labels, final_preds)
    metrics.update({
        "confusion_matrix": confusion_matrix(final_labels, final_preds).tolist(),
        "total_samples": int(len(final_labels)),
        "mode": args.mode,
        "chunk_text_a": bool(chunk_text_a),
        "chunk_overlap": int(args.chunk_overlap),
        "max_text_b_tokens": int(args.max_text_b_tokens),
        "min_text_a_tokens": int(args.min_text_a_tokens),
        "text_b_overlap": int(args.text_b_overlap),
        "chunk_strategy": args.chunk_strategy,
        "evidence_overlap": int(args.evidence_overlap),
        "chunk_aggregate": args.chunk_aggregate,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "model_name": args.model_name,
        "label_definition": {
            "0": "match=true / citation is factually supported",
            "1": "match=false / citation is not factually supported",
        },
        "field_mapping": {
            "text_a": "cited paper facts from <原文事实 (Ground Truth)>",
            "text_b": "originalkey author_related_content from <待核查的引用句> + originalkey citing table caption",
        },
    })

    metrics_path = args.output_dir / "final_eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_predictions(
        args.output_dir / "val_predictions.json",
        val_dataset,
        pred_output.predictions,
        labels,
        args.chunk_aggregate,
    )
    LOGGER.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
