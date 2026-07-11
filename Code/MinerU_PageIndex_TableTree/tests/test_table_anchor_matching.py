import sys
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from popo_workflow_helpers import match_table_anchors  # noqa: E402
from prepare_popo_strict_human_child_annotation_packages import (  # noqa: E402
    build_selected_rows,
)


def anchor(label, page, bbox):
    return {
        "canonical_label": label,
        "page_index": page,
        "bbox": bbox,
    }


def table_node(node_id, page, bbox, block_id):
    return {
        "node_id": node_id,
        "page_indices": [page],
        "block_ids": [block_id],
        "location": [{"page": page, "bbox": bbox}],
    }


def tree_table(page, bbox, block_id):
    return {
        "type": "table",
        "block_ids": [block_id],
        "location": [{"page": page, "bbox": bbox}],
        "children": [],
    }


def test_reordered_nodes_match_by_page_and_bbox():
    anchors = [
        anchor("Table 1", 2, [100, 100, 400, 300]),
        anchor("Table 2", 5, [500, 500, 900, 800]),
    ]
    nodes = [
        table_node("popo-b", 5, [0.501, 0.499, 0.899, 0.802], 20),
        table_node("popo-a", 2, [0.101, 0.098, 0.401, 0.301], 10),
    ]

    audit = match_table_anchors(anchors, nodes)

    assert audit["status"] == "pass"
    assert [item["popo_node_id"] for item in audit["matches"]] == [
        "popo-a",
        "popo-b",
    ]


def test_same_page_tables_match_by_geometry_not_list_order():
    anchors = [
        anchor("Table 1", 3, [100, 100, 450, 300]),
        anchor("Table 2", 3, [100, 600, 450, 850]),
    ]
    nodes = [
        table_node("bottom", 3, [0.10, 0.60, 0.45, 0.85], 2),
        table_node("top", 3, [0.10, 0.10, 0.45, 0.30], 1),
    ]

    audit = match_table_anchors(anchors, nodes)

    assert audit["status"] == "pass"
    assert [item["popo_node_id"] for item in audit["matches"]] == [
        "top",
        "bottom",
    ]


def test_count_mismatch_fails_without_partial_assignment():
    audit = match_table_anchors(
        [anchor("Table 1", 1, [100, 100, 400, 300])],
        [],
    )

    assert audit["status"] == "fail"
    assert audit["matches"] == []
    assert audit["failures"][0]["reason"] == "table_count_mismatch"


def test_wrong_page_fails_closed():
    audit = match_table_anchors(
        [anchor("Table 1", 4, [100, 100, 400, 300])],
        [table_node("wrong-page", 5, [0.10, 0.10, 0.40, 0.30], 1)],
    )

    assert audit["status"] == "fail"
    assert audit["matches"] == []
    assert audit["failures"][0]["reason"] == (
        "anchor_has_no_eligible_popo_table"
    )


def test_duplicate_geometry_fails_as_ambiguous():
    anchors = [
        anchor("Table 1", 1, [100, 100, 400, 300]),
        anchor("Table 2", 1, [100, 100, 400, 300]),
    ]
    nodes = [
        table_node("duplicate-a", 1, [0.10, 0.10, 0.40, 0.30], 1),
        table_node("duplicate-b", 1, [0.10, 0.10, 0.40, 0.30], 2),
    ]

    audit = match_table_anchors(anchors, nodes)

    assert audit["status"] == "fail"
    assert audit["matches"] == []
    assert audit["failures"][0]["reason"] == (
        "ambiguous_multiple_complete_matchings"
    )


def test_failed_document_produces_no_strict_candidates(tmp_path):
    manual = {
        "documents": [
            {
                "slug": "paper-a",
                "tables": [
                    {
                        "table_anchor": anchor(
                            "Table 1", 4, [100, 100, 400, 300]
                        )
                    }
                ],
            }
        ]
    }
    tree = {
        "type": "root",
        "children": [tree_table(5, [0.10, 0.10, 0.40, 0.30], 1)],
    }
    (tmp_path / "paper-a.json").write_text(
        __import__("json").dumps(tree), encoding="utf-8"
    )

    rows, failures, audits = build_selected_rows(
        manual,
        tmp_path,
        parent_score_threshold=120,
        child_score_threshold=60,
        min_children_per_table=3,
        max_children_per_table=11,
    )

    assert rows == []
    assert failures[0]["reason"] == "table_anchor_match_failed"
    assert audits[0]["status"] == "fail"
