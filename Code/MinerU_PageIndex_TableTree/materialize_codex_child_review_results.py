import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, load_json
from prepare_codex_child_review_packages import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, review_key, text_hash


DEFAULT_RESULT_DIR = None
ALLOWED_ROLES = {
    "table_introduction",
    "experimental_condition",
    "dataset",
    "model",
    "metric",
    "training_setting",
    "method",
    "result_interpretation",
    "comparison",
    "limitation",
    "other_support",
    "irrelevant",
}
ALLOWED_SUPPORT = {"direct", "indirect", "none"}


def load_decisions(decision_dir):
    decisions = {}
    sources = {}
    paths = sorted(decision_dir.glob("*_codex_decisions.json"))
    if not paths:
        raise ValueError(f"No Codex decision files found in {decision_dir}")
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if document.get("review_stage") != "codex_semantic_precision_review":
            raise ValueError(f"{path.name}: invalid review_stage")
        for decision in document.get("decisions", []):
            key = decision.get("review_key")
            if not key:
                raise ValueError(f"{path.name}: decision missing review_key")
            if key in decisions:
                raise ValueError(f"Duplicate decision for {key}")
            decisions[key] = decision
            sources[key] = str(path)
    return decisions, sources, paths


def validate_decision(row, decision):
    child = row["child"]
    failures = []
    if decision.get("child_id") != child["child_id"]:
        failures.append("child_id mismatch")
    if decision.get("child_text_sha256") != text_hash(child["child_text"]):
        failures.append("child_text_sha256 mismatch")
    if decision.get("codex_label") not in (0, 1):
        failures.append("codex_label must be 0 or 1")
    if decision.get("semantic_role") not in ALLOWED_ROLES:
        failures.append("invalid semantic_role")
    if decision.get("citation_support") not in ALLOWED_SUPPORT:
        failures.append("invalid citation_support")
    if not str(decision.get("rationale") or "").strip():
        failures.append("rationale is required")
    if decision.get("codex_label") == 1 and decision.get("citation_support") != "none":
        failures.append("label 1 must use citation_support=none")
    if decision.get("codex_label") == 0 and decision.get("semantic_role") == "irrelevant":
        failures.append("label 0 cannot use semantic_role=irrelevant")
    return failures


def render_readable(reviewed_rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in reviewed_rows:
        grouped[row["slug"]][row["table_anchor"]["canonical_label"]].append(row)

    lines = ["# Codex-Reviewed Table Description Child Blocks", ""]
    for slug, tables in grouped.items():
        lines.extend([f"## Paper ID: {slug}", ""])
        for table_label, rows in tables.items():
            retained = [row for row in rows if row["codex_review"]["codex_label"] == 0]
            demoted = [row for row in rows if row["codex_review"]["codex_label"] == 1]
            lines.extend([f"### Tab ID: {table_label}", "", "#### Retained Label 0", ""])
            if retained:
                for index, row in enumerate(retained, 1):
                    review = row["codex_review"]
                    score = row["child"]["child_selection_signals"]["score"]
                    lines.extend(
                        [
                            f"{index}. {row['child']['child_text'].strip()}",
                            f"   - Score: {score}",
                            f"   - Role: {review['semantic_role']}",
                            f"   - Citation support: {review['citation_support']}",
                            f"   - Codex rationale: {review['rationale']}",
                            "",
                        ]
                    )
            else:
                lines.extend(["None.", ""])

            lines.extend(["#### Codex-Demoted Label 1", ""])
            if demoted:
                for index, row in enumerate(demoted, 1):
                    review = row["codex_review"]
                    score = row["child"]["child_selection_signals"]["score"]
                    lines.extend(
                        [
                            f"{index}. {row['child']['child_text'].strip()}",
                            f"   - Score: {score}",
                            f"   - Reason: {review['rationale']}",
                            "",
                        ]
                    )
            else:
                lines.extend(["None.", ""])
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Validate and materialize Codex child decisions.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--decision-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    output_root = Path(manifest["output_root"])
    input_path = args.input or (
        output_root
        / "classifier_candidate_dataset"
        / "selected_table_description_child_blocks.json"
    )
    decision_dir = args.decision_dir or output_root / "codex_child_review_packages"
    output_dir = args.output_dir or output_root / "codex_child_review_results"
    rows = load_json(input_path)
    decisions, sources, decision_paths = load_decisions(decision_dir)
    expected = {review_key(row): row for row in rows}
    missing = sorted(set(expected) - set(decisions))
    unexpected = sorted(set(decisions) - set(expected))
    failures = []
    for key, row in expected.items():
        if key in decisions:
            failures.extend(f"{key}: {message}" for message in validate_decision(row, decisions[key]))
    if missing:
        failures.append(f"Missing {len(missing)} decisions; first: {missing[:5]}")
    if unexpected:
        failures.append(f"Unexpected {len(unexpected)} decisions; first: {unexpected[:5]}")
    if failures:
        raise ValueError("\n".join(failures))

    reviewed_rows = []
    retained_rows = []
    for row in rows:
        enriched = dict(row)
        key = review_key(row)
        review = dict(decisions[key])
        review["decision_source_file"] = sources[key]
        review["decision_source"] = "codex_semantic_review_v1"
        enriched["codex_review"] = review
        enriched["final_child_label"] = review["codex_label"]
        enriched["final_child_label_source"] = "codex_semantic_review_v1"
        reviewed_rows.append(enriched)
        if review["codex_label"] == 0:
            retained_rows.append(enriched)

    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = output_dir / "all_codex_reviewed_child_blocks.json"
    retained_path = output_dir / "retained_table_description_child_blocks.json"
    readable_path = output_dir / "codex_reviewed_child_blocks_readable.md"
    jsonl_path = output_dir / "codex_reviewed_parent_child_samples.jsonl"
    reviewed_path.write_text(json.dumps(reviewed_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    retained_path.write_text(json.dumps(retained_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    readable_path.write_text(render_readable(reviewed_rows), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in reviewed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = Counter(row["final_child_label"] for row in reviewed_rows)
    summary = {
        "schema_version": "1.0",
        "decision_source": "codex_semantic_review_v1",
        "input_candidates": str(input_path),
        "decision_files": [str(path) for path in decision_paths],
        "reviewed_output": str(reviewed_path),
        "retained_output": str(retained_path),
        "readable_output": str(readable_path),
        "jsonl_output": str(jsonl_path),
        "reviewed_children": len(reviewed_rows),
        "final_label_0_retained": label_counts[0],
        "final_label_1_demoted": label_counts[1],
        "audit_failures": [],
    }
    status_path = output_dir / "codex_child_review_status.json"
    status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
