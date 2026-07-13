#!/usr/bin/env python3
"""Audit source coverage against an explicit historical reference set.

This is a regression-only tool. Reference labels and decisions are never copied
into candidate inventories or subagent review packages.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from table_context_pipeline.v2.common import load_json, normalize_text, write_json  # noqa: E402


DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "outputs"
    / "table_context_pipeline_v2"
    / "example_run"
    / "batch_scope_candidate_inventory.json"
)
DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT / "inputs" / "historical_reference"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "table_context_pipeline_v2"
    / "example_run"
    / "regression"
    / "historical_reference_coverage.json"
)


def _reference_slug(payload: dict[str, Any], path: Path) -> str:
    slug = str(payload.get("slug") or "").strip()
    suffix = "_fulltext_conditions"
    if not slug:
        slug = path.stem
    if slug.endswith(suffix):
        slug = slug[: -len(suffix)]
    return slug


def _match_text(value: str) -> str:
    value = re.sub(r"<\|(?:txt_)?split\|>", " ", value or "", flags=re.I)
    value = normalize_text(value)
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    return re.sub(r"\s+", " ", value).strip()


def audit_reference_coverage(
    inventory: dict[str, Any], reference_paths: list[Path]
) -> dict[str, Any]:
    documents = {
        str(document["paper_id"]): document
        for document in inventory.get("documents", [])
    }
    rows: list[dict[str, Any]] = []
    missing_documents: list[str] = []
    origin_counts: Counter[str] = Counter()

    for path in sorted(reference_paths):
        reference = load_json(path)
        paper_id = _reference_slug(reference, path)
        document = documents.get(paper_id)
        if document is None:
            missing_documents.append(paper_id)
            continue

        parents = document.get("parents", [])
        parents_by_id = {
            str(parent.get("parent_id")): parent
            for parent in parents
            if parent.get("parent_id")
        }
        normalized_parents = [
            (parent, _match_text(parent.get("full_text") or ""))
            for parent in parents
            if not parent.get("outline_only") and parent.get("full_text")
        ]
        for table in reference.get("tables", []):
            table_label = table.get("table_label")
            for evidence in table.get("evidence", []):
                evidence_text = str(evidence.get("evidence_text") or "")
                needle = _match_text(evidence_text)
                matches = [
                    parent
                    for parent, haystack in normalized_parents
                    if needle and needle in haystack
                ]
                match_mode = "normalized_text" if matches else None
                match_score = 1.0 if matches else None
                if not matches and needle:
                    combined = " ".join(haystack for _, haystack in normalized_parents)
                    if needle in combined:
                        matches = [
                            parent
                            for parent, haystack in normalized_parents
                            if set(needle.split()) & set(haystack.split())
                        ]
                        match_mode = "normalized_consecutive_parent_text"
                        match_score = 1.0
                if not matches and needle:
                    scored = [
                        (difflib.SequenceMatcher(None, needle, haystack).ratio(), parent)
                        for parent, haystack in normalized_parents
                    ]
                    best_score, best_parent = max(
                        scored, default=(0.0, None), key=lambda row: row[0]
                    )
                    if best_parent is not None and best_score >= 0.80:
                        matches = [best_parent]
                        match_mode = "format_tolerant_text"
                        match_score = round(best_score, 6)
                source_parent_id = str(evidence.get("popo_node_id") or "")
                if not matches and source_parent_id in parents_by_id:
                    source_parent = parents_by_id[source_parent_id]
                    if not source_parent.get("outline_only") and source_parent.get("full_text"):
                        matches = [source_parent]
                        match_mode = "source_popo_node_trace"
                        match_score = None
                if not matches:
                    source_block_ids = {
                        int(value)
                        for value in evidence.get("block_ids", [])
                        if isinstance(value, int) or str(value).isdigit()
                    }
                    block_matches = [
                        parent
                        for parent in parents
                        if not parent.get("outline_only")
                        and source_block_ids.intersection(
                            int(value)
                            for value in parent.get("block_ids", [])
                            if isinstance(value, int) or str(value).isdigit()
                        )
                    ]
                    if block_matches:
                        matches = block_matches
                        match_mode = "source_block_trace"
                        match_score = None
                origins = sorted(
                    {str(parent.get("source_origin") or "unknown") for parent in matches}
                )
                if matches:
                    for origin in origins:
                        origin_counts[origin] += 1
                rows.append(
                    {
                        "paper_id": paper_id,
                        "table_label": table_label,
                        "evidence_id": evidence.get("evidence_id"),
                        "evidence_text": evidence_text,
                        "covered": bool(matches),
                        "match_mode": match_mode,
                        "match_score": match_score,
                        "matching_parent_ids": [
                            parent.get("parent_id") for parent in matches
                        ],
                        "source_origins": origins,
                    }
                )

    covered_count = sum(row["covered"] for row in rows)
    unique_evidence: dict[str, bool] = {}
    for row in rows:
        key = normalize_text(row["evidence_text"])
        unique_evidence[key] = unique_evidence.get(key, False) or row["covered"]
    unique_covered = sum(unique_evidence.values())
    status = "pass" if covered_count == len(rows) and not missing_documents else "fail"
    return {
        "schema_version": "historical-reference-source-coverage-v2",
        "purpose": "regression_only_not_a_review_input",
        "status": status,
        "document_count": len({row["paper_id"] for row in rows}),
        "missing_documents": missing_documents,
        "association_count": len(rows),
        "covered_association_count": covered_count,
        "association_coverage": covered_count / len(rows) if rows else 0.0,
        "unique_evidence_count": len(unique_evidence),
        "covered_unique_evidence_count": unique_covered,
        "unique_evidence_coverage": unique_covered / len(unique_evidence)
        if unique_evidence
        else 0.0,
        "covered_association_origin_counts": dict(sorted(origin_counts.items())),
        "uncovered": [row for row in rows if not row["covered"]],
        "records": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Historical Reference Coverage Audit",
        "",
        "This is a regression-only audit. Historical decisions were not exposed to reviewers.",
        "",
        f"- Status: {report['status']}",
        f"- Table-condition associations: {report['covered_association_count']}/{report['association_count']} ({report['association_coverage']:.2%})",
        f"- Unique evidence spans: {report['covered_unique_evidence_count']}/{report['unique_evidence_count']} ({report['unique_evidence_coverage']:.2%})",
        f"- Covered origins: {json.dumps(report['covered_association_origin_counts'], ensure_ascii=False, sort_keys=True)}",
        "",
        "## Uncovered Evidence",
        "",
    ]
    if not report["uncovered"]:
        lines.append("None.")
    else:
        for row in report["uncovered"]:
            lines.extend(
                [
                    f"- {row['paper_id']} / {row['table_label']} / {row['evidence_id']}",
                    f"  - {row['evidence_text']}",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = audit_reference_coverage(
        load_json(args.inventory),
        list(args.reference_dir.glob("*_fulltext_conditions.json")),
    )
    write_json(args.output, report)
    args.output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status",
        "association_count",
        "covered_association_count",
        "association_coverage",
        "unique_evidence_count",
        "covered_unique_evidence_count",
        "unique_evidence_coverage",
    )}, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
