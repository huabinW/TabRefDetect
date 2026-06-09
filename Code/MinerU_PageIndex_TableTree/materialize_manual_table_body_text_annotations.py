import argparse
import json
from pathlib import Path

from build_table_text_tree_batch import DEFAULT_MANIFEST, load_json


DEFAULT_SELECTIONS = Path(__file__).with_name("manual_table_body_text_selections.json")


def iter_leaves(nodes):
    for node in nodes:
        if node.get("node_type"):
            yield node
        else:
            yield from iter_leaves(node.get("nodes", []) or [])


def table_anchor(leaf):
    return {
        "table_node_id": leaf["node_id"],
        "canonical_label": leaf["caption_label"],
        "caption": leaf.get("caption"),
        "page_index": leaf.get("page_index"),
        "bbox": leaf.get("bbox"),
        "order_key": leaf.get("order_key"),
        "raw_content_index": leaf.get("raw_content_index"),
        "parent_node_id": leaf.get("parent_node_id"),
        "parent_title": leaf.get("parent_title"),
        "assignment_reason": leaf.get("assignment_reason"),
        "table_html": leaf.get("html"),
        "table_code_body": leaf.get("code_body"),
        "image_path": leaf.get("image_path"),
    }


def paragraph_annotation(paragraph, selection):
    return {
        "annotation_source": "manual_selection",
        "manual_parent_label": 0,
        "label_meaning": "correct/relevant table-text relation",
        "anchor_raw_content_index": selection["anchor_raw_content_index"],
        "evidence_type": selection["evidence_type"],
        "relevance": selection["relevance"],
        "manual_rationale": selection["rationale"],
        "parent_paragraph_id": paragraph["node_id"],
        "full_parent_text": paragraph["text"],
        "raw_content_indices": paragraph["raw_content_indices"],
        "page_indices": paragraph["page_indices"],
        "page_start": paragraph["page_start"],
        "page_end": paragraph["page_end"],
        "bboxes": paragraph["bboxes"],
        "order_key": paragraph["order_key"],
        "tree_parent_node_id": paragraph["parent_node_id"],
        "tree_parent_title": paragraph["parent_title"],
        "merge_reasons": paragraph["merge_reasons"],
        "raw_blocks": paragraph["raw_blocks"],
    }


def materialize_document(selection_doc, output_root):
    slug = selection_doc["slug"]
    tree_path = output_root / "local_paragraph_table_text_trees" / (
        f"{slug}_local_paragraph_table_text_tree.json"
    )
    tree_payload = load_json(tree_path)
    leaves = list(iter_leaves(tree_payload["structure"]))
    tables = {
        leaf["caption_label"]: leaf
        for leaf in leaves
        if leaf.get("node_type") == "table_leaf"
    }
    paragraphs_by_raw_index = {}
    for paragraph in tree_payload["paragraph_corpus"]:
        for raw_index in paragraph["raw_content_indices"]:
            if raw_index in paragraphs_by_raw_index:
                raise ValueError(f"{slug}: raw text index {raw_index} belongs to multiple paragraphs")
            paragraphs_by_raw_index[raw_index] = paragraph

    selected_labels = [table["table_label"] for table in selection_doc["tables"]]
    if set(selected_labels) != set(tables):
        raise ValueError(
            f"{slug}: selection table labels {selected_labels} do not match tree labels {list(tables)}"
        )

    output_tables = []
    for table_selection in selection_doc["tables"]:
        label = table_selection["table_label"]
        seen_paragraphs = set()
        evidence = []
        for selection in table_selection["evidence"]:
            anchor_index = selection["anchor_raw_content_index"]
            paragraph = paragraphs_by_raw_index.get(anchor_index)
            if paragraph is None:
                raise ValueError(f"{slug} {label}: no paragraph contains raw index {anchor_index}")
            if paragraph["node_id"] in seen_paragraphs:
                raise ValueError(
                    f"{slug} {label}: paragraph {paragraph['node_id']} selected more than once"
                )
            seen_paragraphs.add(paragraph["node_id"])
            evidence.append(paragraph_annotation(paragraph, selection))

        evidence.sort(key=lambda item: (item["page_start"], item["order_key"], item["raw_content_indices"][0]))
        output_tables.append(
            {
                "table_anchor": table_anchor(tables[label]),
                "manual_body_text_evidence": evidence,
                "evidence_count": len(evidence),
            }
        )

    return {
        "slug": slug,
        "source_pdf": tree_payload["source_pdf"],
        "mineru_source": tree_payload["mineru_source"],
        "paragraph_tree_source": str(tree_path),
        "annotation_policy": {
            "selection": "Manual reading of original paper; no automatic relevance classifier or remote LLM.",
            "text": "Verbatim full parent paragraphs copied from the local MinerU paragraph tree.",
            "label_semantics": {
                "0": "correct/relevant table-text relation",
                "1": "incorrect/irrelevant table-text relation",
            },
            "include": [
                "direct numbered-table references",
                "table-specific result interpretations",
                "datasets, models, metrics, prompts, shots, splits, baselines, and settings needed to interpret the table",
            ],
            "exclude": [
                "table caption and body as positive prose evidence",
                "reference-list entries",
                "unrelated nearby prose",
            ],
        },
        "table_count": len(output_tables),
        "tables": output_tables,
    }


def render_markdown(payload):
    lines = [
        f"# Manual Table Body-Text Annotations: {payload['slug']}",
        "",
        "- Selection: manual original-paper review",
        "- Text: full verbatim MinerU parent paragraphs",
        "",
    ]
    for table in payload["tables"]:
        anchor = table["table_anchor"]
        lines.extend(
            [
                f"## {anchor['canonical_label']}",
                "",
                f"- Caption: {anchor['caption']}",
                f"- Table page: {anchor['page_index']}",
                f"- Evidence paragraphs: {table['evidence_count']}",
                "",
            ]
        )
        for item in table["manual_body_text_evidence"]:
            lines.extend(
                [
                    f"### {item['parent_paragraph_id']} | raw {item['raw_content_indices']}",
                    "",
                    f"- Type: `{item['evidence_type']}`",
                    f"- Relevance: `{item['relevance']}`",
                    f"- Rationale: {item['manual_rationale']}",
                    f"- Tree section: `{item['tree_parent_node_id']} {item['tree_parent_title']}`",
                    "",
                    item["full_parent_text"],
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Materialize manually selected table-body evidence.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    selections = load_json(args.selections)
    output_root = Path(manifest["output_root"])
    output_dir = output_root / "manual_table_body_text_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        materialize_document(selection_doc, output_root)
        for selection_doc in selections["documents"]
    ]
    combined = {
        "annotation_version": selections["annotation_version"],
        "selection_method": selections["selection_method"],
        "unit": selections["unit"],
        "document_count": len(documents),
        "table_count": sum(document["table_count"] for document in documents),
        "documents": documents,
    }

    for document in documents:
        slug = document["slug"]
        (output_dir / f"{slug}_table_body_text_annotations.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{slug}_table_body_text_annotations.md").write_text(
            render_markdown(document),
            encoding="utf-8",
        )

    combined_path = output_dir / "batch_manual_table_body_text_annotations.json"
    combined_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(combined_path),
                "documents": combined["document_count"],
                "tables": combined["table_count"],
                "evidence_paragraphs": sum(
                    table["evidence_count"]
                    for document in documents
                    for table in document["tables"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
