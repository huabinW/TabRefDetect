import json
from pathlib import Path

from table_context_pipeline.v2.build_scope_candidate_inventory import (
    BATCH_FILENAME,
    STATUS_FILENAME,
    _content_text,
    build_document_inventory,
    build_from_config,
)
from table_context_pipeline.v2.common import (
    is_reference_path,
    parse_table_references,
    split_child_spans,
)


def test_reference_section_matching_requires_a_reference_title():
    assert is_reference_path(["Paper", "References"])
    assert is_reference_path(["Paper", "A. Bibliography"])
    assert not is_reference_path(
        ["Paper", "F Reference-free Evaluation in Direct Assessment Formats"]
    )


def test_content_text_keeps_footnotes_and_lists_but_not_layout_noise():
    assert _content_text(
        {"type": "page_footnote", "text": "Experimental caveat."}
    ) == "Experimental caveat."
    assert _content_text({"type": "list", "list_items": ["A", "B"]}) == "A\nB"
    assert _content_text({"type": "page_header", "text": "Conference"}) is None


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def table_anchor(table_id, label, bbox, page=1):
    return {
        "table_node_id": table_id,
        "canonical_label": label,
        "caption": f"{label}: Scores on the test set.",
        "page_index": page,
        "bbox": bbox,
        "raw_content_index": 20,
        "table_html": "<table><tr><td>BLEU</td></tr></table>",
    }


def text_node(title, content, block_ids, page, bbox, children=None):
    return {
        "type": "text",
        "title": title,
        "content": content,
        "block_ids": block_ids,
        "location": [{"page": page, "bbox": bbox}],
        "children": children or [],
    }


def table_node(block_id, page, bbox):
    return {
        "type": "table",
        "title": "",
        "content": "",
        "block_ids": [block_id],
        "location": [{"page": page, "bbox": bbox}],
        "children": [],
    }


def fixture_files(tmp_path):
    slug = "paper-a"
    manifest_path = tmp_path / "anchors.json"
    tree_path = tmp_path / "trees" / f"{slug}.json"
    mineru_path = tmp_path / "mineru" / "content_list.json"
    overrides_path = tmp_path / "overrides.json"
    document = {
        "slug": slug,
        "mineru_source": "mineru/content_list.json",
        "tables": [
            {"table_anchor": table_anchor("table-0001", "Table 1", [100, 300, 400, 500])},
            {"table_anchor": table_anchor("table-0002", "Table 2", [500, 300, 900, 500], page=9)},
        ],
    }
    tree = {
        "type": "root",
        "title": "",
        "content": "",
        "block_ids": [],
        "location": [],
        "children": [
            text_node(
                "Experiments",
                "Dr. Smith used Dataset α.<|txt_split|>Tables 1 and 2 report BLEU.",
                [11],
                1,
                [0.1, 0.05, 0.9, 0.25],
                children=[
                    table_node(21, 1, [0.1, 0.3, 0.4, 0.5]),
                    table_node(22, 1, [0.5, 0.3, 0.9, 0.5]),
                ],
            ),
            text_node(
                "References",
                "[1] A reference entry. It must not become a child.",
                [90],
                2,
                [0.1, 0.2, 0.9, 0.9],
            ),
        ],
    }
    content_list = [
        {
            "type": "text",
            "text": "Dr. Smith used Dataset α.",
            "block_id": 11,
            "page_idx": 0,
            "bbox": [100, 50, 900, 120],
        },
        {
            "type": "text",
            "text": "Dataset α.",
            "page_idx": 0,
            "bbox": [100, 130, 900, 160],
        },
        {
            "type": "text",
            "text": "Table 1: Scores on the test set.",
            "page_idx": 0,
            "bbox": [100, 280, 400, 300],
        },
        {
            "type": "text",
            "text": "For all experiments, beam size was 4.",
            "page_idx": 0,
            "bbox": [100, 600, 900, 650],
        },
        {
            "type": "text",
            "text": "[1] A reference entry.",
            "page_idx": 1,
            "bbox": [100, 300, 900, 350],
        },
        {"type": "text", "text": "", "page_idx": 0, "bbox": [0, 0, 0, 0]},
    ]
    overrides = {
        "schema_version": "1.0",
        "overrides": [
            {
                "slug": slug,
                "canonical_label": "Table 2",
                "page_index": 1,
                "bbox": [500, 300, 900, 500],
                "caption": "Table 2: Corrected caption.",
                "reason": "test correction",
            }
        ],
    }
    write_json(manifest_path, {"documents": [document]})
    write_json(tree_path, tree)
    write_json(mineru_path, content_list)
    write_json(overrides_path, overrides)
    return document, manifest_path, tree_path, mineru_path, overrides_path, overrides


def test_document_inventory_is_complete_unique_and_auditable(tmp_path):
    document, manifest, tree, mineru, overrides_path, overrides = fixture_files(tmp_path)

    inventory = build_document_inventory(
        document,
        tree,
        mineru,
        manifest,
        overrides["overrides"],
        overrides_path,
        workspace_root=tmp_path,
    )

    assert inventory["table_anchor_audit"]["status"] == "pass"
    applied = inventory["table_anchor_audit"]["override_audit"]["applied"]
    assert applied[0]["before"]["page_index"] == 9
    assert applied[0]["after"]["page_index"] == 1
    assert applied[0]["after"]["caption"] == "Table 2: Corrected caption."
    assert inventory["review_eligible_table_ids"] == ["table-0001", "table-0002"]
    assert inventory["policy"]["pre_review_cap"] is None
    assert inventory["policy"]["score_deletes_candidates"] is False

    parents = {row["parent_id"]: row for row in inventory["parents"]}
    references = [row for row in parents.values() if row["outline_only"]]
    assert len(references) == 1
    assert references[0]["full_text"] is None
    assert references[0]["child_ids"] == []

    orphans = [row for row in parents.values() if row["source_origin"] == "mineru_orphan_fallback"]
    assert [row["full_text"] for row in orphans] == ["For all experiments, beam size was 4."]
    assert inventory["source_coverage"]["mineru_orphan_parent_count"] == 1
    statuses = [row["status"] for row in inventory["source_coverage"]["mineru_text_block_audit"]]
    assert statuses == [
        "covered_by_popo_block_id",
        "covered_by_popo_content",
        "excluded_known_table_caption",
        "mineru_orphan_fallback",
        "excluded_reference_area",
        "excluded_empty_text",
    ]

    child_ids = [row["child_id"] for row in inventory["children"]]
    assert len(child_ids) == len(set(child_ids))
    for child in inventory["children"]:
        parent = parents[child["parent_id"]]
        assert child["text"] == parent["full_text"][child["char_start"] : child["char_end"]]
        assert child["review_eligible_table_ids"] == ["table-0001", "table-0002"]

    multi_table_child = next(row for row in inventory["children"] if row["text"].startswith("Tables"))
    assert multi_table_child["structured_table_references"][0]["resolved_table_ids"] == [
        "table-0001",
        "table-0002",
    ]
    assert {
        row["table_id"] for row in multi_table_child["table_suggestions"]
    } == {"table-0001", "table-0002"}
    assert all(row["binding"] is False for row in multi_table_child["table_suggestions"])


def test_segmentation_prefers_popo_boundaries_and_preserves_abbreviations():
    text = "Dr. Smith used e.g. BLEU v1.2.<|txt_split|>结果稳定。下一句。"

    spans = split_child_spans(text)
    children = [text[start:end] for start, end in spans]

    assert children == ["Dr. Smith used e.g. BLEU v1.2.", "结果稳定。", "下一句。"]
    assert all(text[start:end] == child for (start, end), child in zip(spans, children))


def test_structured_table_reference_parser_handles_lists_ranges_and_supplements():
    tables = [
        {"table_id": f"t{number}", "canonical_label": f"Table {number}"}
        for number in range(1, 4)
    ] + [{"table_id": "ts1", "canonical_label": "Table S1"}]

    mentions = parse_table_references("See Tables 1–3 and Tbl. S1.", tables)

    assert mentions[0]["resolved_table_ids"] == ["t1", "t2", "t3"]
    assert mentions[1]["resolved_table_ids"] == ["ts1"]


def test_failed_anchor_audit_publishes_only_status_and_returns_failure(tmp_path):
    document = {
        "slug": "bad-paper",
        "mineru_source": "mineru.json",
        "tables": [
            {"table_anchor": table_anchor("table-0001", "Table 1", [100, 100, 400, 300])}
        ],
    }
    write_json(tmp_path / "manifest.json", {"documents": [document]})
    write_json(tmp_path / "mineru.json", [])
    write_json(
        tmp_path / "trees" / "bad-paper.json",
        {"type": "root", "children": []},
    )
    write_json(
        tmp_path / "config.json",
        {
            "test_anchor_manifest": "manifest.json",
            "test_popo_tree_dir": "trees",
            "output_dir": "output",
        },
    )
    stale_inventory = tmp_path / "output" / "inventory" / "bad-paper.json"
    stale_batch = tmp_path / "output" / BATCH_FILENAME
    write_json(stale_inventory, {"stale": True})
    write_json(stale_batch, {"stale": True})

    status, batch = build_from_config(
        tmp_path / "config.json", workspace_root=tmp_path, publish=True
    )

    assert status["status"] == "fail"
    assert status["documents"][0]["reason"] == "table_anchor_match_failed"
    assert batch is None
    assert not stale_inventory.exists()
    assert not stale_batch.exists()
    assert (tmp_path / "output" / STATUS_FILENAME).exists()
    assert list((tmp_path / "output").glob("*.json")) == [
        tmp_path / "output" / STATUS_FILENAME
    ]
