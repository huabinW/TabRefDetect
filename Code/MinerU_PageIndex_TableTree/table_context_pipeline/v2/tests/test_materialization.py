from __future__ import annotations

import copy
import json

import pytest

from table_context_pipeline.v2.materialize_scope_reviews import (
    REVIEW_STAGE,
    materialize_from_paths,
    materialize_scope_reviews,
    validate_decision_payload,
    write_materialized_outputs,
)
from table_context_pipeline.v2.prepare_scope_review_packages import (
    build_document_package,
    prepare_review_packages,
)


@pytest.fixture
def inventory() -> dict:
    parent_text = "All runs use five seeds. Table results follow."
    first = "All runs use five seeds."
    second = "Table results follow."
    return {
        "schema_version": "scope-candidate-inventory-v2",
        "candidate_child_count": 2,
        "documents": [
            {
                "paper_id": "paper/one",
                "title": "A paper",
                "review_eligible_table_ids": ["T1", "T2", "T3"],
                "tables": [
                    {
                        "table_id": "T1",
                        "canonical_label": "Table 1",
                        "caption": "Primary results",
                        "table_body": "<table><tr><td>1</td></tr></table>",
                        "anchor": {
                            "caption": "Primary results",
                            "table_html": "<table><tr><td>1</td></tr></table>",
                            "page_index": 3,
                            "bbox": [1, 2, 3, 4],
                        },
                    },
                    {
                        "table_id": "T2",
                        "table_label": "Table 2",
                        "table_caption": "Ablation results",
                        "table_body": "model | score",
                        "page": 4,
                    },
                    {
                        "table_id": "T3",
                        "table_label": "Table 3",
                        "table_caption": "Self-contained table",
                        "table_body": "setting | value",
                        "page": 5,
                    },
                ],
                "scopes": [
                    {
                        "scope_id": "S1",
                        "title": "Experiments",
                        "parent_ids": ["P1"],
                        "table_ids": ["T1", "T2"],
                        "codex_review": {"old": True},
                    }
                ],
                "parents": [
                    {
                        "parent_id": "P1",
                        "full_text": parent_text,
                        "page_indices": [2],
                        "bboxes": [[10, 20, 30, 40]],
                        "source_origin": "popo_text",
                        "child_ids": ["C1", "C2"],
                    },
                    {
                        "parent_id": "P-ref",
                        "full_text": None,
                        "full_text_sha256": "reference-source-hash",
                        "outline_only": True,
                        "content_omitted_reason": "reference_outline_only",
                        "source_origin": "popo_text",
                        "child_ids": [],
                    },
                ],
                "children": [
                    {
                        "child_id": "C1",
                        "parent_id": "P1",
                        "text": first,
                        "char_start": parent_text.index(first),
                        "char_end": parent_text.index(first) + len(first),
                        "scope_id": "S1",
                        "source_origin": "popo_text",
                        "review_eligible_table_ids": ["T1", "T2", "T3"],
                        "human_label": 0,
                        "human_rationale": "historical gold must not leak",
                        "codex_label": 0,
                    },
                    {
                        "child_id": "C2",
                        "parent_id": "P1",
                        "text": second,
                        "char_start": parent_text.index(second),
                        "char_end": parent_text.index(second) + len(second),
                        "scope_id": "S1",
                        "table_suggestions": [{"table_id": "T1", "ordering_score": 8}],
                        "page_indices": [2],
                    },
                ],
            }
        ],
    }


@pytest.fixture
def decisions() -> dict:
    return {
        "schema_version": "scope-review-decisions-v2",
        "review_stage": REVIEW_STAGE,
        "paper_id": "paper/one",
        "decisions": [
            {
                "child_id": "C1",
                "parent_id": "P1",
                "relevant_tables": [
                    {
                        "table_id": "T1",
                        "label": 0,
                        "confidence": 1.0,
                        "evidence_role": "experimental_condition",
                        "rationale": "The seed count is not visible in Table 1.",
                    },
                    {
                        "table_id": "T2",
                        "label": 0,
                        "confidence": 0.75,
                        "evidence_role": "experimental_condition",
                        "rationale": "The same scope governs the ablation.",
                    },
                ],
                "rejection_reason": None,
            },
            {
                "child_id": "C2",
                "parent_id": "P1",
                "relevant_tables": [],
                "rejection_reason": "Pure table pointer.",
            },
        ],
    }


def test_package_has_unique_children_all_tables_and_no_historical_labels(
    inventory: dict, tmp_path
) -> None:
    package = build_document_package(inventory, inventory["documents"][0])

    assert [item["child_id"] for item in package["review_items"]] == ["C1", "C2"]
    assert {table["table_id"] for table in package["tables"]} == {"T1", "T2", "T3"}
    assert package["tables"][0]["table_caption"] == "Primary results"
    assert package["tables"][0]["table_body"].startswith("<table>")
    assert package["task_definition"]["caption_policy"].startswith("Captions are table-anchor")
    assert "Pure table pointers" in " ".join(package["task_definition"]["reject"])
    serialized = json.dumps(package, ensure_ascii=False)
    assert "human_label" not in serialized
    assert "human_rationale" not in serialized
    assert "codex_label" not in serialized
    assert "historical gold must not leak" not in serialized

    input_path = tmp_path / "batch_scope_candidate_inventory.json"
    input_path.write_text(json.dumps(inventory), encoding="utf-8")
    status = prepare_review_packages(input_path, tmp_path / "packages")
    assert status["review_item_count"] == 2
    assert (tmp_path / "packages" / "paper_one_scope_review_package.json").exists()
    assert (tmp_path / "packages" / "paper_one_scope_review_package.md").exists()


def test_decision_validation_requires_complete_known_unique_ids_and_confidence(
    inventory: dict, decisions: dict
) -> None:
    package = build_document_package(inventory, inventory["documents"][0])
    validated = validate_decision_payload(package, decisions)
    assert set(validated) == {"C1", "C2"}
    assert validated["C1"]["relevant_tables"][0]["confidence"] == 1.0

    missing = copy.deepcopy(decisions)
    missing["decisions"].pop()
    with pytest.raises(ValueError, match="missing decisions"):
        validate_decision_payload(package, missing)

    duplicate = copy.deepcopy(decisions)
    duplicate["decisions"].append(copy.deepcopy(duplicate["decisions"][0]))
    with pytest.raises(ValueError, match="duplicate child_id"):
        validate_decision_payload(package, duplicate)

    unknown_table = copy.deepcopy(decisions)
    unknown_table["decisions"][0]["relevant_tables"][0]["table_id"] = "T99"
    with pytest.raises(ValueError, match="unknown table_id"):
        validate_decision_payload(package, unknown_table)

    invalid_confidence = copy.deepcopy(decisions)
    invalid_confidence["decisions"][0]["relevant_tables"][0]["confidence"] = 1.01
    with pytest.raises(ValueError, match="confidence must be in"):
        validate_decision_payload(package, invalid_confidence)


def test_materialization_preserves_zero_tables_and_builds_compact_human_rows(
    inventory: dict, decisions: dict, tmp_path
) -> None:
    outputs = materialize_scope_reviews(
        inventory,
        {"paper/one": decisions},
        {"paper/one": "fresh/paper_one_scope_review_decisions.json"},
    )

    audit = outputs["full_audit"]
    assert audit["child_count"] == 2
    assert len(audit["documents"][0]["children"]) == 2
    assert audit["documents"][0]["children"][1]["subagent_decision"]["rejection_reason"] == "Pure table pointer."

    selected = outputs["selected_relations"]
    assert selected["selected_relation_count"] == 2
    summaries = {table["table_id"]: table for table in selected["documents"][0]["tables"]}
    assert summaries["T1"]["selected_relation_count"] == 1
    assert summaries["T2"]["selected_relation_count"] == 1
    assert summaries["T3"]["selected_relation_count"] == 0

    human_tables = {
        table["table_id"]: table
        for table in outputs["human_template"]["documents"][0]["tables"]
    }
    assert human_tables["T3"]["records"] == []
    row = human_tables["T1"]["records"][0]
    assert row["paper_id"] == "paper/one"
    assert row["table_caption"] == "Primary results"
    assert row["table_body"].startswith("<table>")
    assert row["full_parent_text"][row["char_start"] : row["char_end"]] == row["child_text"]
    assert row["page"] == [2]
    assert row["bbox"] == [[10, 20, 30, 40]]
    assert row["source"] == "popo_text"
    assert row["subagent_confidence"] == 1.0
    assert row["subagent_role"] == "experimental_condition"
    assert row["human_label"] is None
    assert row["human_rationale"] == ""
    assert "table_caption" not in decisions["decisions"][0]

    status = write_materialized_outputs(outputs, tmp_path / "results")
    assert status["tables_without_relations"] == 1
    assert (tmp_path / "results" / "scope_review_full_audit.json").exists()
    assert (tmp_path / "results" / "scope_review_selected_relations.json").exists()
    assert "Table 3" in (tmp_path / "results" / "scope_review_human_template.md").read_text(encoding="utf-8")


def test_path_materialization_reads_only_configured_fresh_decision_glob(
    inventory: dict, decisions: dict, tmp_path
) -> None:
    input_path = tmp_path / "batch_scope_candidate_inventory.json"
    input_path.write_text(json.dumps(inventory), encoding="utf-8")
    decision_dir = tmp_path / "decisions"
    decision_dir.mkdir()
    (decision_dir / "paper_one_scope_review_decisions.json").write_text(
        json.dumps(decisions), encoding="utf-8"
    )
    (decision_dir / "old_historical_decisions.json").write_text(
        json.dumps({"paper_id": "unknown", "decisions": []}), encoding="utf-8"
    )

    status = materialize_from_paths(input_path, decision_dir, tmp_path / "results")

    assert status["reviewed_child_count"] == 2
    assert status["selected_relation_count"] == 2
    assert status["validation_failures"] == []
