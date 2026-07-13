#!/usr/bin/env python3
"""Compare fresh selected relations with a historical full-text reference audit.

The comparison is post-hoc only. It does not modify decisions or human labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from table_context_pipeline.v2.common import load_json, normalize_text, write_json  # noqa: E402


DEFAULT_SELECTED = (
    PROJECT_ROOT / "outputs" / "table_context_pipeline_v2"
    / "example_run" / "scope_review_results" / "scope_review_selected_relations.json"
)
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "inputs" / "historical_reference"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "table_context_pipeline_v2"
    / "example_run" / "regression" / "selected_vs_historical_reference.json"
)
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def match_text(value: str) -> str:
    value = re.sub(r"<\|(?:txt_)?split\|>", " ", value or "", flags=re.I)
    value = normalize_text(value)
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def text_match_score(child_text: str, evidence_text: str) -> float:
    child = match_text(child_text)
    evidence = match_text(evidence_text)
    if not child or not evidence:
        return 0.0
    if child in evidence or evidence in child:
        return 1.0
    child_tokens = TOKEN_RE.findall(child)
    evidence_tokens = set(TOKEN_RE.findall(evidence))
    if len(child_tokens) < 4:
        return 0.0
    overlap = sum(token in evidence_tokens for token in child_tokens)
    return overlap / len(child_tokens)


def reference_rows(reference_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(reference_dir.glob("*_fulltext_conditions.json")):
        payload = load_json(path)
        paper_id = str(payload.get("slug") or path.stem.replace("_fulltext_conditions", ""))
        for table in payload.get("tables", []):
            for evidence in table.get("evidence", []):
                rows.append(
                    {
                        "paper_id": paper_id,
                        "table_label": table.get("table_label"),
                        "evidence_id": evidence.get("evidence_id"),
                        "evidence_text": evidence.get("evidence_text") or "",
                        "condition_type": evidence.get("condition_type"),
                    }
                )
    return rows


def compare(selected: dict[str, Any], references: list[dict[str, Any]]) -> dict[str, Any]:
    relations = selected.get("relations", [])
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        by_key[(relation["paper_id"], relation["table_label"])].append(relation)

    reference_audit = []
    matched_relation_keys: set[tuple[str, str, str]] = set()
    paper_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for reference in references:
        candidates = by_key[(reference["paper_id"], reference["table_label"])]
        scored = sorted(
            (
                (text_match_score(relation["child_text"], reference["evidence_text"]), relation)
                for relation in candidates
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        best_score, best_relation = scored[0] if scored else (0.0, None)
        matched = best_relation is not None and best_score >= 0.75
        if matched:
            matched_relation_keys.add(
                (best_relation["paper_id"], best_relation["table_id"], best_relation["child_id"])
            )
        paper_counts[reference["paper_id"]]["reference"] += 1
        paper_counts[reference["paper_id"]]["matched_reference"] += int(matched)
        reference_audit.append(
            {
                **reference,
                "matched": matched,
                "best_match_score": round(best_score, 6),
                "best_child_id": best_relation.get("child_id") if best_relation else None,
                "best_child_text": best_relation.get("child_text") if best_relation else None,
            }
        )

    relation_audit = []
    for relation in relations:
        candidates = [
            reference
            for reference in references
            if reference["paper_id"] == relation["paper_id"]
            and reference["table_label"] == relation["table_label"]
        ]
        best_score = max(
            (text_match_score(relation["child_text"], reference["evidence_text"]) for reference in candidates),
            default=0.0,
        )
        overlaps = best_score >= 0.75
        paper_counts[relation["paper_id"]]["selected_relation"] += 1
        paper_counts[relation["paper_id"]]["reference_overlap_relation"] += int(overlaps)
        relation_audit.append(
            {
                "paper_id": relation["paper_id"],
                "table_id": relation["table_id"],
                "table_label": relation["table_label"],
                "child_id": relation["child_id"],
                "child_text": relation["child_text"],
                "subagent_role": relation.get("subagent_role"),
                "subagent_confidence": relation.get("subagent_confidence"),
                "overlaps_historical_reference": overlaps,
                "best_match_score": round(best_score, 6),
            }
        )

    matched_reference_count = sum(row["matched"] for row in reference_audit)
    overlap_relation_count = sum(row["overlaps_historical_reference"] for row in relation_audit)
    per_paper = []
    for paper_id, counts in sorted(paper_counts.items()):
        per_paper.append(
            {
                "paper_id": paper_id,
                **counts,
                "reference_recall": counts["matched_reference"] / counts["reference"]
                if counts["reference"] else None,
                "selected_reference_overlap_rate": counts["reference_overlap_relation"]
                / counts["selected_relation"] if counts["selected_relation"] else None,
            }
        )
    return {
        "schema_version": "selected-vs-historical-reference-v2",
        "purpose": "post_hoc_audit_only_not_human_gold_mutation",
        "matching_rule": "same paper/table and normalized child-to-evidence token coverage >= 0.75",
        "reference_association_count": len(reference_audit),
        "matched_reference_association_count": matched_reference_count,
        "reference_recall": matched_reference_count / len(reference_audit) if reference_audit else 0.0,
        "selected_relation_count": len(relation_audit),
        "reference_overlap_selected_relation_count": overlap_relation_count,
        "selected_reference_overlap_rate": overlap_relation_count / len(relation_audit) if relation_audit else 0.0,
        "interpretation_warning": "Non-overlap is not automatically a false positive because the historical reference may be incomplete; inspect differences manually.",
        "per_paper": per_paper,
        "unmatched_reference": [row for row in reference_audit if not row["matched"]],
        "selected_without_reference_overlap": [row for row in relation_audit if not row["overlaps_historical_reference"]],
        "reference_audit": reference_audit,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fresh Review vs Historical Full-text Reference",
        "",
        "This is a post-hoc audit. Historical evidence was not exposed to fresh subagents.",
        "",
        f"- Historical association recall: {report['matched_reference_association_count']}/{report['reference_association_count']} ({report['reference_recall']:.2%})",
        f"- Fresh selected relations overlapping the historical reference: {report['reference_overlap_selected_relation_count']}/{report['selected_relation_count']} ({report['selected_reference_overlap_rate']:.2%})",
        "- A non-overlap is an audit target, not automatically a false positive.",
        "",
        "## Per Paper",
        "",
        "| Paper | Historical recall | Selected/reference overlap |",
        "|---|---:|---:|",
    ]
    for row in report["per_paper"]:
        lines.append(
            f"| {row['paper_id']} | {row['matched_reference']}/{row['reference']} ({row['reference_recall']:.2%}) | "
            f"{row['reference_overlap_relation']}/{row['selected_relation']} ({row['selected_reference_overlap_rate']:.2%}) |"
        )
    lines.extend(["", "## Unmatched Historical Evidence", ""])
    if not report["unmatched_reference"]:
        lines.append("None.")
    else:
        for row in report["unmatched_reference"]:
            lines.extend(
                [
                    f"- {row['paper_id']} / {row['table_label']} / {row['evidence_id']} (best={row['best_match_score']:.3f})",
                    f"  - Evidence: {row['evidence_text']}",
                    f"  - Best fresh child: {row['best_child_text'] or ''}",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = compare(load_json(args.selected), reference_rows(args.reference_dir))
    write_json(args.output, report)
    args.output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "reference_recall": report["reference_recall"],
        "matched_reference_association_count": report["matched_reference_association_count"],
        "reference_association_count": report["reference_association_count"],
        "selected_reference_overlap_rate": report["selected_reference_overlap_rate"],
        "reference_overlap_selected_relation_count": report["reference_overlap_selected_relation_count"],
        "selected_relation_count": report["selected_relation_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
