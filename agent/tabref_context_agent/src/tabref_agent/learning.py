from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": "1.0",
    "policy_version": "candidate-policy-1.0.0",
    "source": "built_in_high_recall_default",
    "weights": {
        "exact_table_reference": 100.0,
        "table_text_overlap_per_token": 5.0,
        "table_text_overlap_cap": 30.0,
        "annotation_rationale_overlap_per_token": 5.0,
        "annotation_rationale_overlap_cap": 25.0,
        "evidence_family_match": 25.0,
        "condition_match": 15.0,
        "result_match": 15.0,
    },
    "review_threshold": 0.0,
    "force_review_signals": ["exact_table_reference"],
    "guardrails": {
        "min_examples": 30,
        "min_recall": 0.98,
        "min_table_coverage": 1.0,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(payload)
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    table = row.get("table") or row.get("table_anchor") or {}
    child = row.get("child") or {}
    return (
        str(row.get("slug", "")),
        str(table.get("canonical_label", row.get("table_label", ""))),
        str(row.get("candidate_id", "")),
        str(child.get("child_id", row.get("child_id", ""))),
    )


def feedback_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("slug", "")),
        str(row.get("table_label", "")),
        str(row.get("candidate_id", "")),
        str(row.get("child_id", "")),
    )


def validate_feedback(rows: list[dict[str, Any]]) -> None:
    required = {"slug", "table_label", "candidate_id", "child_id", "gold_label"}
    seen: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(rows, start=1):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Feedback row {index} is missing fields: {missing}")
        if row["gold_label"] not in (0, 1):
            raise ValueError(f"Feedback row {index} has invalid gold_label")
        key = feedback_key(row)
        if key in seen:
            raise ValueError(f"Duplicate human feedback key: {key}")
        seen.add(key)


def load_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return deepcopy(DEFAULT_POLICY)
    policy = read_json(path)
    if "weights" not in policy or "review_threshold" not in policy:
        raise ValueError(f"Invalid candidate policy: {path}")
    return policy


def score_signals(signals: dict[str, Any], policy: dict[str, Any]) -> float:
    weights = policy["weights"]
    table_overlap = len(signals.get("table_text_overlap", []))
    rationale_overlap = len(signals.get("annotation_rationale_overlap", []))
    score = (
        weights["exact_table_reference"]
        if signals.get("exact_table_reference")
        else 0.0
    )
    score += min(
        weights["table_text_overlap_cap"],
        table_overlap * weights["table_text_overlap_per_token"],
    )
    score += min(
        weights["annotation_rationale_overlap_cap"],
        rationale_overlap * weights["annotation_rationale_overlap_per_token"],
    )
    score += (
        weights["evidence_family_match"]
        if signals.get("evidence_family_match")
        else 0.0
    )
    score += weights["condition_match"] if signals.get("condition_match") else 0.0
    score += (
        weights["result_match"]
        if signals.get("result_match")
        and signals.get("evidence_family") == "result"
        else 0.0
    )
    return round(float(score), 6)


def is_selected(signals: dict[str, Any], policy: dict[str, Any]) -> bool:
    if any(signals.get(name) for name in policy.get("force_review_signals", [])):
        return True
    return score_signals(signals, policy) >= float(policy["review_threshold"])


def evaluate_policy(
    examples: list[tuple[dict[str, Any], dict[str, Any]]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    positive_tables: set[tuple[str, str]] = set()
    covered_tables: set[tuple[str, str]] = set()
    scored = []
    for candidate, feedback in examples:
        signals = candidate.get("child", {}).get("child_selection_signals", {})
        scored.append(
            {
                "candidate": candidate,
                "feedback": feedback,
                "score": score_signals(signals, policy),
                "selected": is_selected(signals, policy),
            }
        )
    by_parent: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in scored:
        feedback = item["feedback"]
        parent_key = (
            feedback["slug"],
            feedback["table_label"],
            feedback["candidate_id"],
        )
        by_parent.setdefault(parent_key, []).append(item)
    for parent_items in by_parent.values():
        if not any(item["selected"] for item in parent_items):
            max(parent_items, key=lambda item: item["score"])["selected"] = True

    for item in scored:
        feedback = item["feedback"]
        selected = item["selected"]
        relevant = feedback["gold_label"] == 0
        table_key = feedback["slug"], feedback["table_label"]
        if relevant:
            positive_tables.add(table_key)
            if selected:
                covered_tables.add(table_key)
                true_positive += 1
            else:
                false_negative += 1
        elif selected:
            false_positive += 1
        else:
            true_negative += 1
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    coverage = (
        len(covered_tables) / len(positive_tables) if positive_tables else 0.0
    )
    return {
        "examples": len(examples),
        "positive_examples": true_positive + false_negative,
        "negative_examples": false_positive + true_negative,
        "selected_examples": true_positive + false_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "table_coverage": round(coverage, 6),
    }


def _policy_candidates(
    base_policy: dict[str, Any],
    examples: list[tuple[dict[str, Any], dict[str, Any]]],
) -> Iterable[dict[str, Any]]:
    weight_names = [
        "table_text_overlap_per_token",
        "annotation_rationale_overlap_per_token",
        "evidence_family_match",
        "condition_match",
        "result_match",
    ]
    weight_sets = [deepcopy(base_policy["weights"])]
    for name in weight_names:
        for factor in (0.5, 1.5, 2.0):
            weights = deepcopy(base_policy["weights"])
            weights[name] = round(weights[name] * factor, 6)
            weight_sets.append(weights)

    for weights in weight_sets:
        policy = deepcopy(base_policy)
        policy["weights"] = weights
        scores = sorted(
            {
                score_signals(
                    candidate.get("child", {}).get("child_selection_signals", {}),
                    policy,
                )
                for candidate, _ in examples
            }
        )
        for threshold in [0.0, *scores]:
            proposal = deepcopy(policy)
            proposal["review_threshold"] = threshold
            yield proposal


def optimize_policy(
    candidates_path: Path,
    feedback_path: Path,
    active_policy_path: Path,
    output_root: Path,
    min_examples: int,
    min_recall: float,
    min_table_coverage: float,
) -> dict[str, Any]:
    feedback = load_jsonl(feedback_path)
    validate_feedback(feedback)
    candidate_rows = load_jsonl(candidates_path)
    candidate_by_key = {candidate_key(row): row for row in candidate_rows}
    unmatched = [
        feedback_key(row) for row in feedback if feedback_key(row) not in candidate_by_key
    ]
    if unmatched:
        raise ValueError(f"Human feedback has unmatched candidate keys: {unmatched[:5]}")
    examples = [(candidate_by_key[feedback_key(row)], row) for row in feedback]
    active = load_policy(active_policy_path)
    active["guardrails"] = {
        "min_examples": min_examples,
        "min_recall": min_recall,
        "min_table_coverage": min_table_coverage,
    }
    baseline_metrics = evaluate_policy(examples, active)
    enough_data = (
        len(examples) >= min_examples
        and baseline_metrics["positive_examples"] > 0
        and baseline_metrics["negative_examples"] > 0
    )
    result: dict[str, Any] = {
        "status": "skipped_insufficient_human_feedback",
        "feedback_path": str(feedback_path),
        "candidate_path": str(candidates_path),
        "active_policy_path": str(active_policy_path),
        "human_examples": len(examples),
        "baseline_metrics": baseline_metrics,
        "guardrails": active["guardrails"],
        "policy_activated": False,
    }
    if not enough_data:
        return result

    viable: list[tuple[tuple[float, float, float], dict[str, Any], dict[str, Any]]] = []
    for proposal in _policy_candidates(active, examples):
        metrics = evaluate_policy(examples, proposal)
        if (
            metrics["recall"] >= min_recall
            and metrics["table_coverage"] >= min_table_coverage
        ):
            objective = (
                metrics["precision"],
                -metrics["selected_examples"],
                metrics["recall"],
            )
            viable.append((objective, proposal, metrics))
    if not viable:
        result["status"] = "skipped_no_policy_passed_guardrails"
        return result

    _, best_policy, best_metrics = max(viable, key=lambda item: item[0])
    improves = (
        best_metrics["precision"] > baseline_metrics["precision"]
        or best_metrics["selected_examples"] < baseline_metrics["selected_examples"]
    )
    if not improves:
        result.update(
            {
                "status": "kept_active_policy_no_improvement",
                "proposed_metrics": best_metrics,
            }
        )
        return result

    old_version = active.get("policy_version", "candidate-policy-1.0.0")
    prefix, numeric = old_version.rsplit("-", 1)
    major, minor, patch = (int(part) for part in numeric.split("."))
    best_policy.update(
        {
            "policy_version": f"{prefix}-{major}.{minor}.{patch + 1}",
            "source": "human_gold_optimization",
            "created_at_utc": utc_now(),
            "parent_policy_version": old_version,
            "metrics": best_metrics,
            "guardrails": active["guardrails"],
        }
    )
    history_dir = output_root / "candidate_policy/history"
    if active_policy_path.exists():
        history_path = history_dir / f"{old_version}_{safe_timestamp()}.json"
        write_json(history_path, active)
    write_json(active_policy_path, best_policy)
    result.update(
        {
            "status": "activated",
            "policy_activated": True,
            "new_policy_version": best_policy["policy_version"],
            "proposed_metrics": best_metrics,
        }
    )
    return result


def build_reflection(
    feedback_path: Path,
    candidate_path: Path,
    reviewed_path: Path,
    learning_root: Path,
    optimization: dict[str, Any],
) -> dict[str, Any]:
    feedback = load_jsonl(feedback_path)
    candidates = {candidate_key(row): row for row in load_jsonl(candidate_path)}
    reviewed_rows = read_json(reviewed_path) if reviewed_path.exists() else []
    reviewed = {candidate_key(row): row for row in reviewed_rows}
    error_counts: dict[str, int] = {}
    memory_events = []
    for gold in feedback:
        key = feedback_key(gold)
        predicted_row = reviewed.get(key)
        predicted = (
            predicted_row.get("final_child_label") if predicted_row is not None else None
        )
        if predicted is None:
            category = "unreviewed_human_example"
        elif predicted == gold["gold_label"]:
            category = "agreement"
        elif predicted == 0:
            category = "false_positive"
        else:
            category = "false_negative"
        error_counts[category] = error_counts.get(category, 0) + 1
        signals = (
            candidates.get(key, {})
            .get("child", {})
            .get("child_selection_signals", {})
        )
        memory_events.append(
            {
                "recorded_at_utc": utc_now(),
                "source": "human_gold",
                "key": list(key),
                "gold_label": gold["gold_label"],
                "codex_label": predicted,
                "outcome": category,
                "error_category": gold.get("error_category"),
                "comment": gold.get("comment"),
                "signals": signals,
            }
        )
    if memory_events:
        append_jsonl(learning_root / "memory/events.jsonl", memory_events)

    lessons = []
    if error_counts.get("false_positive"):
        lessons.append(
            {
                "rule": "Require table-scoped explanatory value; a low-score generic sentence may be demoted.",
                "support": error_counts["false_positive"],
                "source": "human_gold_false_positives",
            }
        )
    if error_counts.get("false_negative"):
        lessons.append(
            {
                "rule": "Preserve implicit experimental-condition evidence even without an explicit table reference.",
                "support": error_counts["false_negative"],
                "source": "human_gold_false_negatives",
            }
        )
    reflection = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "human_feedback_count": len(feedback),
        "error_counts": error_counts,
        "optimization": optimization,
        "lessons": lessons,
        "codex_is_training_gold": False,
    }
    path = learning_root / "reflections" / f"reflection_{safe_timestamp()}.json"
    write_json(path, reflection)
    write_json(learning_root / "memory/lessons.json", {"lessons": lessons})
    reflection["reflection_path"] = str(path)
    return reflection


def _bump_patch(version: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def stage_skill_proposal(
    skill_dir: Path,
    learning_root: Path,
    reflection: dict[str, Any],
) -> dict[str, Any]:
    lessons = reflection.get("lessons", [])
    if not lessons:
        return {"status": "skipped_no_human_supported_lessons", "proposal_staged": False}
    version_path = skill_dir / "references/version.json"
    skill_path = skill_dir / "SKILL.md"
    if not version_path.exists() or not skill_path.exists():
        return {
            "status": "skipped_skill_not_found",
            "proposal_staged": False,
            "skill_dir": str(skill_dir),
        }
    current_version = read_json(version_path)["version"]
    proposed_version = _bump_patch(current_version)
    proposal_id = f"selector-{proposed_version}-{safe_timestamp()}"
    proposal_root = learning_root / "skill_proposals/pending" / proposal_id
    files_root = proposal_root / "files"
    shutil.copytree(skill_dir, files_root)

    skill_text = (files_root / "SKILL.md").read_text(encoding="utf-8")
    skill_text = skill_text.replace(
        f"Current version: `{current_version}`.",
        f"Current version: `{proposed_version}`.",
    )
    learned_lines = [
        "## Human-Supported Learned Rules",
        "",
        "Apply these rules only because they were derived from preserved human gold:",
        "",
    ]
    for lesson in lessons:
        learned_lines.append(
            f"- {lesson['rule']} (support: {lesson['support']} human corrections)"
        )
    learned_lines.extend(
        [
            "",
            "Do not edit this section from Codex predictions alone.",
            "",
        ]
    )
    marker = "## Updating With User Gold"
    if "## Human-Supported Learned Rules" in skill_text:
        before, remainder = skill_text.split("## Human-Supported Learned Rules", 1)
        _, after = remainder.split(marker, 1)
        skill_text = before + "\n".join(learned_lines) + marker + after
    else:
        skill_text = skill_text.replace(marker, "\n".join(learned_lines) + marker)
    (files_root / "SKILL.md").write_text(skill_text, encoding="utf-8")
    version_payload = read_json(files_root / "references/version.json")
    version_payload["version"] = proposed_version
    version_payload["approval_status"] = "pending_human_approval"
    write_json(files_root / "references/version.json", version_payload)
    manifest = {
        "schema_version": "1.0",
        "proposal_id": proposal_id,
        "status": "pending_human_approval",
        "created_at_utc": utc_now(),
        "skill": "tabref-table-text-child-selector",
        "skill_dir": str(skill_dir),
        "base_version": current_version,
        "proposed_version": proposed_version,
        "source_reflection": reflection.get("reflection_path"),
        "allowed_files": ["SKILL.md", "references/version.json"],
        "approval_required": True,
        "auto_apply_forbidden": True,
    }
    write_json(proposal_root / "manifest.json", manifest)
    return {
        "status": "pending_human_approval",
        "proposal_staged": True,
        "proposal_id": proposal_id,
        "proposal_dir": str(proposal_root),
        "base_version": current_version,
        "proposed_version": proposed_version,
    }


def list_skill_proposals(learning_root: Path) -> list[dict[str, Any]]:
    pending_root = learning_root / "skill_proposals/pending"
    if not pending_root.exists():
        return []
    return [
        read_json(path)
        for path in sorted(pending_root.glob("*/manifest.json"))
    ]


def approve_skill_proposal(
    learning_root: Path,
    proposal_id: str,
    approver: str,
) -> dict[str, Any]:
    proposal_root = learning_root / "skill_proposals/pending" / proposal_id
    manifest_path = proposal_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pending Skill proposal not found: {proposal_id}")
    manifest = read_json(manifest_path)
    skill_dir = Path(manifest["skill_dir"]).expanduser().resolve()
    current_version = read_json(skill_dir / "references/version.json")["version"]
    if current_version != manifest["base_version"]:
        raise ValueError(
            f"Skill changed since proposal: expected {manifest['base_version']}, "
            f"found {current_version}"
        )
    history_dir = (
        learning_root
        / "skill_history/tabref-table-text-child-selector"
        / f"{current_version}_{safe_timestamp()}"
    )
    shutil.copytree(skill_dir, history_dir)
    files_root = proposal_root / "files"
    for relative in manifest["allowed_files"]:
        source = (files_root / relative).resolve()
        if files_root.resolve() not in source.parents:
            raise ValueError(f"Unsafe proposal path: {relative}")
        destination = skill_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest.update(
        {
            "status": "approved",
            "approved_at_utc": utc_now(),
            "approved_by": approver,
            "history_snapshot": str(history_dir),
        }
    )
    write_json(manifest_path, manifest)
    approved_root = learning_root / "skill_proposals/approved" / proposal_id
    approved_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(proposal_root), str(approved_root))
    return manifest


def reject_skill_proposal(
    learning_root: Path,
    proposal_id: str,
    approver: str,
    reason: str,
) -> dict[str, Any]:
    proposal_root = learning_root / "skill_proposals/pending" / proposal_id
    manifest_path = proposal_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pending Skill proposal not found: {proposal_id}")
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "status": "rejected",
            "rejected_at_utc": utc_now(),
            "rejected_by": approver,
            "rejection_reason": reason,
        }
    )
    write_json(manifest_path, manifest)
    rejected_root = learning_root / "skill_proposals/rejected" / proposal_id
    rejected_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(proposal_root), str(rejected_root))
    return manifest
