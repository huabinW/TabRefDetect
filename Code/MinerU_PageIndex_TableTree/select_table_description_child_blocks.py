import argparse
import copy
import json
import re
from pathlib import Path

from build_table_text_classifier_candidates import (
    CONDITION_RE,
    DEFAULT_ANNOTATIONS,
    DEFAULT_MANIFEST,
    lexical_tokens,
    table_reference_pattern,
)
from build_table_text_tree_batch import load_json


RESULT_RE = re.compile(
    r"\b(?:result|show|demonstrate|observe|achiev|outperform|improv|decreas|increase|"
    r"compare|performance|score|gain|best|worse|higher|lower|gap)\w*\b",
    re.I,
)
DATASET_RE = re.compile(
    r"\b(?:dataset|corpus|benchmark|train(?:ing)? set|dev(?:elopment)? set|test set|"
    r"split|sample|example|annotation|topic|dialogue|chart)\w*\b",
    re.I,
)
MODEL_RE = re.compile(
    r"\b(?:model|architecture|encoder|decoder|transformer|resnet|roberta|gru|cnn|"
    r"parameter|layer|attention|embedding|classifier|baseline)\w*\b",
    re.I,
)
METRIC_RE = re.compile(
    r"\b(?:metric|accuracy|precision|recall|f[- ]?score|f1|bleu|pass@1|muc|ceaf|"
    r"blanc|conll|rnss|rms|rating|agreement)\w*\b",
    re.I,
)
TRAINING_RE = re.compile(
    r"\b(?:train|pretrain|finetun|optimizer|adam|learning rate|batch|epoch|step|"
    r"hyperparameter|grid search|cross-validation|pruning rate|shot|prompt|decod|"
    r"quantization|gpu|resolution)\w*\b",
    re.I,
)
METHOD_RE = re.compile(
    r"\b(?:method|approach|task|feature|template|objective|procedure|stage|setting|"
    r"configuration|selection|generation|filtering|evaluation)\w*\b",
    re.I,
)


EVIDENCE_FAMILIES = {
    "dataset": DATASET_RE,
    "model": MODEL_RE,
    "metric": METRIC_RE,
    "training": TRAINING_RE,
    "result": RESULT_RE,
    "method": METHOD_RE,
}


def evidence_family(evidence_type):
    lowered = evidence_type.lower()
    if any(token in lowered for token in ("result", "interpretation", "comparison", "ablation")):
        return "result"
    if any(token in lowered for token in ("dataset", "corpus")):
        return "dataset"
    if any(token in lowered for token in ("model", "architecture")):
        return "model"
    if any(token in lowered for token in ("metric", "evaluation_protocol")):
        return "metric"
    if any(token in lowered for token in ("training", "hyperparameter", "implementation", "comput")):
        return "training"
    return "method"


def annotation_index(annotations):
    result = {}
    for document in annotations["documents"]:
        slug = document["slug"]
        for table in document["tables"]:
            label = table["table_anchor"]["canonical_label"]
            for evidence in table["manual_body_text_evidence"]:
                result[(slug, label, evidence["parent_paragraph_id"])] = evidence
    return result


def child_signals(table, child_text, evidence):
    exact_reference = bool(table_reference_pattern(table["canonical_label"]).search(child_text))
    table_tokens = lexical_tokens(
        " ".join(
            value or ""
            for value in (
                table.get("caption"),
                table.get("table_html"),
                table.get("table_code_body"),
            )
        )
    )
    context_tokens = lexical_tokens(
        f"{evidence.get('manual_rationale', '')} {evidence.get('evidence_type', '')}"
    )
    child_tokens = lexical_tokens(child_text)
    table_overlap = sorted(table_tokens & child_tokens)
    rationale_overlap = sorted(context_tokens & child_tokens)
    family = evidence_family(evidence["evidence_type"])
    family_match = bool(EVIDENCE_FAMILIES[family].search(child_text))
    condition_match = bool(CONDITION_RE.search(child_text))
    result_match = bool(RESULT_RE.search(child_text))
    score = 0
    score += 100 if exact_reference else 0
    score += min(30, len(table_overlap) * 5)
    score += min(25, len(rationale_overlap) * 5)
    score += 25 if family_match else 0
    score += 15 if condition_match else 0
    score += 15 if result_match and family == "result" else 0

    return {
        "score": score,
        "exact_table_reference": exact_reference,
        "table_text_overlap": table_overlap,
        "annotation_rationale_overlap": rationale_overlap,
        "evidence_family": family,
        "evidence_family_match": family_match,
        "condition_match": condition_match,
        "result_match": result_match,
    }


def select_children(children, table, evidence):
    output = []
    for child in children:
        signals = child_signals(table, child["child_text"], evidence)
        enriched = dict(child)
        enriched.update(
            {
                "child_label": 0,
                "child_label_meaning": "high-recall candidate from a correct/relevant parent paragraph",
                "child_label_source": "code_high_recall_v1",
                "child_review_status": "awaiting_codex_semantic_review",
                "child_selection_signals": signals,
                "child_selection_reason": "all_children_preserved_from_relevant_parent_for_recall",
            }
        )
        output.append(enriched)
    return output


def label_dataset(dataset, annotations):
    evidence_by_key = annotation_index(annotations)
    labeled = copy.deepcopy(dataset)
    selected_rows = []

    for document in labeled["documents"]:
        slug = document["slug"]
        for table_item in document["tables"]:
            table = table_item["table_anchor"]
            label = table["canonical_label"]
            for candidate in table_item["candidates"]:
                key = (slug, label, candidate["parent_paragraph_id"])
                if candidate["manual_parent_label"] == 1:
                    candidate["children"] = [
                        {
                            **child,
                            "child_label": 1,
                            "child_label_meaning": "incorrect/irrelevant table-description child block",
                            "child_label_source": "inherited_from_incorrect_parent",
                            "child_review_status": "deterministic",
                            "child_selection_signals": {},
                            "child_selection_reason": "parent_relation_is_incorrect",
                        }
                        for child in candidate["children"]
                    ]
                    continue

                evidence = evidence_by_key.get(key)
                if evidence is None:
                    raise ValueError(f"Missing manual parent evidence for {key}")
                candidate["parent_evidence_type"] = evidence["evidence_type"]
                candidate["parent_manual_rationale"] = evidence["manual_rationale"]
                candidate["children"] = select_children(candidate["children"], table, evidence)
                for child in candidate["children"]:
                    row = {
                        "slug": slug,
                        "table_anchor": table,
                        "candidate_id": candidate["candidate_id"],
                        "parent_paragraph_id": candidate["parent_paragraph_id"],
                        "full_parent_text": candidate["full_parent_text"],
                        "parent_raw_content_indices": candidate["raw_content_indices"],
                        "parent_page_indices": candidate["page_indices"],
                        "parent_node_id": candidate["parent_node_id"],
                        "parent_title": candidate["parent_title"],
                        "parent_evidence_type": evidence["evidence_type"],
                        "parent_manual_rationale": evidence["manual_rationale"],
                        "child": child,
                    }
                    if child["child_label"] == 0:
                        selected_rows.append(row)
    labeled["dataset_version"] = "1.2"
    labeled["label_semantics"]["child_label"] = {
        "0": "correct/relevant table-description child block",
        "1": "incorrect/irrelevant child block",
    }
    labeled["child_label_policy"] = {
        "version": "code_high_recall_v1",
        "incorrect_parent": "All children inherit label 1.",
        "correct_parent": "Preserve every child from each correct/relevant parent paragraph. Signals and scores only rank candidates; they never remove a child. Final precision is decided by Codex.",
        "gold_status": "High-recall candidates only; not final labels or human gold.",
    }
    return labeled, selected_rows


def audit(labeled, selected_rows):
    failures = []
    selected_ids = {
        (row["slug"], row["table_anchor"]["canonical_label"], row["candidate_id"], row["child"]["child_id"])
        for row in selected_rows
    }
    label_counts = {0: 0, 1: 0}
    for document in labeled["documents"]:
        for table in document["tables"]:
            for candidate in table["candidates"]:
                correct_children = 0
                for child in candidate["children"]:
                    label = child["child_label"]
                    if label not in (0, 1):
                        failures.append(f"{candidate['candidate_id']} has invalid child label {label}")
                        continue
                    label_counts[label] += 1
                    correct_children += label == 0
                    if candidate["full_parent_text"][child["char_start"]:child["char_end"]] != child["child_text"]:
                        failures.append(f"{candidate['candidate_id']} child offsets changed text")
                    key = (
                        document["slug"],
                        table["table_anchor"]["canonical_label"],
                        candidate["candidate_id"],
                        child["child_id"],
                    )
                    if (label == 0) != (key in selected_ids):
                        failures.append(f"{candidate['candidate_id']} selected output mismatch")
                if candidate["manual_parent_label"] == 1 and correct_children:
                    failures.append(f"{candidate['candidate_id']} has correct child under incorrect parent")
                if candidate["manual_parent_label"] == 0 and not correct_children:
                    failures.append(f"{candidate['candidate_id']} correct parent has no selected child")
    return failures, label_counts


def render_readable_selected_children(selected_rows):
    lines = ["# Table Description Child Blocks (Label 0)", ""]
    current_slug = None
    current_table = None
    item_number = 0
    for row in selected_rows:
        slug = row["slug"]
        table_label = row["table_anchor"]["canonical_label"]
        if slug != current_slug:
            current_slug = slug
            current_table = None
            lines.extend([f"## Paper ID: {slug}", ""])
        if table_label != current_table:
            current_table = table_label
            item_number = 0
            lines.extend([f"### Tab ID: {table_label}", ""])
        item_number += 1
        score = row["child"]["child_selection_signals"]["score"]
        lines.extend(
            [
                f"{item_number}. {row['child']['child_text'].strip()}",
                f"   - Score: {score}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Select table-description child blocks.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    output_dir = Path(manifest["output_root"]) / "classifier_candidate_dataset"
    candidates_path = args.candidates or output_dir / "batch_table_text_classifier_candidates.json"
    annotations_path = args.annotations or (
        Path(manifest["output_root"])
        / "manual_table_body_text_annotations"
        / "batch_manual_table_body_text_annotations.json"
    )
    dataset = load_json(candidates_path)
    annotations = load_json(annotations_path)
    labeled, selected_rows = label_dataset(dataset, annotations)
    failures, label_counts = audit(labeled, selected_rows)
    if failures:
        raise ValueError("\n".join(failures))

    labeled_path = output_dir / "batch_table_text_classifier_candidates_with_child_labels.json"
    selected_path = output_dir / "selected_table_description_child_blocks.json"
    readable_path = output_dir / "selected_table_description_child_blocks_readable.md"
    jsonl_path = output_dir / "batch_table_text_classifier_parent_child_labeled.jsonl"
    labeled_path.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_path.write_text(json.dumps(selected_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    readable_path.write_text(
        render_readable_selected_children(selected_rows),
        encoding="utf-8",
    )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for document in labeled["documents"]:
            for table in document["tables"]:
                for candidate in table["candidates"]:
                    for child in candidate["children"]:
                        handle.write(
                            json.dumps(
                                {
                                    "slug": document["slug"],
                                    "table": table["table_anchor"],
                                    "candidate_id": candidate["candidate_id"],
                                    "manual_parent_label": candidate["manual_parent_label"],
                                    "parent": {
                                        "parent_paragraph_id": candidate["parent_paragraph_id"],
                                        "full_parent_text": candidate["full_parent_text"],
                                        "raw_content_indices": candidate["raw_content_indices"],
                                        "page_indices": candidate["page_indices"],
                                        "parent_node_id": candidate["parent_node_id"],
                                        "parent_title": candidate["parent_title"],
                                    },
                                    "child": child,
                                    "candidate_signals": candidate["candidate_signals"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

    summary = {
        "labeled_dataset": str(labeled_path),
        "selected_children": str(selected_path),
        "readable_selected_children": str(readable_path),
        "jsonl": str(jsonl_path),
        "child_label_0_correct": label_counts[0],
        "child_label_1_incorrect": label_counts[1],
        "selected_parent_relations": len(
            {
                (row["slug"], row["table_anchor"]["canonical_label"], row["candidate_id"])
                for row in selected_rows
            }
        ),
        "audit_failures": failures,
    }
    (output_dir / "table_description_child_selection_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
