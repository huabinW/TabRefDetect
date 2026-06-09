import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, load_json


DEFAULT_INPUT = None
DEFAULT_OUTPUT_DIR = None


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def review_key(row):
    child = row["child"]
    return "::".join(
        (
            row["slug"],
            row["table_anchor"]["canonical_label"],
            row["candidate_id"],
            child["child_id"],
        )
    )


def build_package(slug, rows):
    tables = defaultdict(list)
    for row in rows:
        tables[row["table_anchor"]["canonical_label"]].append(row)

    packaged_tables = []
    for table_label, table_rows in tables.items():
        anchor = table_rows[0]["table_anchor"]
        parents = defaultdict(list)
        for row in table_rows:
            parents[row["candidate_id"]].append(row)

        packaged_parents = []
        for candidate_id, parent_rows in parents.items():
            first = parent_rows[0]
            review_items = []
            for row in parent_rows:
                child = row["child"]
                review_items.append(
                    {
                        "review_key": review_key(row),
                        "child_id": child["child_id"],
                        "child_text": child["child_text"],
                        "child_text_sha256": text_hash(child["child_text"]),
                        "char_start": child["char_start"],
                        "char_end": child["char_end"],
                        "code_recall_score": child["child_selection_signals"]["score"],
                        "code_recall_signals": child["child_selection_signals"],
                        "codex_decision_template": {
                            "codex_label": None,
                            "semantic_role": None,
                            "citation_support": None,
                            "rationale": None,
                        },
                    }
                )
            packaged_parents.append(
                {
                    "candidate_id": candidate_id,
                    "parent_paragraph_id": first["parent_paragraph_id"],
                    "full_parent_text": first["full_parent_text"],
                    "full_parent_text_sha256": text_hash(first["full_parent_text"]),
                    "parent_title": first["parent_title"],
                    "parent_page_indices": first["parent_page_indices"],
                    "parent_raw_content_indices": first["parent_raw_content_indices"],
                    "parent_evidence_type": first["parent_evidence_type"],
                    "parent_manual_rationale": first["parent_manual_rationale"],
                    "review_items": review_items,
                }
            )

        packaged_tables.append(
            {
                "table_label": table_label,
                "caption": anchor.get("caption"),
                "table_html": anchor.get("table_html"),
                "table_code_body": anchor.get("table_code_body"),
                "page_index": anchor.get("page_index"),
                "bbox": anchor.get("bbox"),
                "parent_node_id": anchor.get("parent_node_id"),
                "parent_title": anchor.get("parent_title"),
                "parents": packaged_parents,
            }
        )

    return {
        "schema_version": "1.0",
        "review_stage": "codex_semantic_precision_review",
        "slug": slug,
        "label_semantics": {
            "0": "retain: the child describes, supplements, introduces, or interprets this table and can support later citation-reference judgment",
            "1": "demote: the child is generic, redundant, unrelated, or insufficiently table-scoped",
        },
        "decision_policy": [
            "Judge meaning from the table, complete parent paragraph, and exact child text.",
            "Retain experimental conditions, datasets, models, metrics, settings, comparisons, limitations, and result interpretations scoped to the table.",
            "Retain a table-introduction sentence when it identifies what the table contains.",
            "Demote headings, generic transitions, unrelated background, and fragments that add no useful table-scoped evidence.",
            "Treat code_recall_score only as a recall/ranking aid, never as the semantic decision.",
            "Use 0 for correct/relevant and 1 for incorrect/irrelevant.",
        ],
        "tables": packaged_tables,
    }


def render_markdown(package):
    lines = [
        f"# Codex Child Review Package: {package['slug']}",
        "",
        "Label 0 = retain; label 1 = demote. The code score is reference only.",
        "",
    ]
    for table in package["tables"]:
        lines.extend(
            [
                f"## {table['table_label']}",
                "",
                f"Caption: {table.get('caption') or ''}",
                "",
            ]
        )
        for parent in table["parents"]:
            lines.extend(
                [
                    f"### {parent['candidate_id']}",
                    "",
                    f"Evidence type: {parent['parent_evidence_type']}",
                    "",
                    f"Parent: {parent['full_parent_text']}",
                    "",
                ]
            )
            for item in parent["review_items"]:
                lines.extend(
                    [
                        f"- Key: `{item['review_key']}`",
                        f"  Score: {item['code_recall_score']}",
                        f"  Child: {item['child_text'].strip()}",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Prepare high-recall child candidates for Codex review.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    output_root = Path(manifest["output_root"])
    input_path = args.input or (
        output_root
        / "classifier_candidate_dataset"
        / "selected_table_description_child_blocks.json"
    )
    output_dir = args.output_dir or output_root / "codex_child_review_packages"
    rows = load_json(input_path)
    by_slug = defaultdict(list)
    for row in rows:
        by_slug[row["slug"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"schema_version": "1.0", "input": str(input_path), "documents": []}
    for slug, document_rows in by_slug.items():
        package = build_package(slug, document_rows)
        json_path = output_dir / f"{slug}_codex_review_package.json"
        md_path = output_dir / f"{slug}_codex_review_package.md"
        decision_path = output_dir / f"{slug}_codex_decisions.json"
        json_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(package), encoding="utf-8")
        status["documents"].append(
            {
                "slug": slug,
                "tables": len(package["tables"]),
                "review_items": len(document_rows),
                "package_json": str(json_path),
                "package_markdown": str(md_path),
                "expected_decisions": str(decision_path),
            }
        )

    status["review_items"] = sum(item["review_items"] for item in status["documents"])
    status_path = output_dir / "codex_child_review_package_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
