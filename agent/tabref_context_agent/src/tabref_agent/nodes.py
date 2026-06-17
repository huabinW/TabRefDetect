from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from .config import AgentConfig
from .learning import (
    build_active_memory_pack,
    build_reflection,
    optimize_policy,
    stage_skill_proposal,
)
from .state import AgentState
from .tools import ProjectToolRunner, append_event, load_json


REVIEW_STATUS = Path(
    "batch_table_text_tree/codex_child_review_packages/"
    "codex_child_review_package_status.json"
)
REVIEW_RESULTS = Path(
    "batch_table_text_tree/codex_child_review_results/"
    "codex_child_review_status.json"
)
CANDIDATE_STATUS = Path(
    "batch_table_text_tree/classifier_candidate_dataset/"
    "batch_table_text_classifier_candidates_status.json"
)
SELECTION_STATUS = Path(
    "batch_table_text_tree/classifier_candidate_dataset/"
    "table_description_child_selection_status.json"
)
PARAGRAPH_AUDIT = Path(
    "batch_table_text_tree/local_paragraph_table_text_tree_audits/"
    "batch_local_paragraph_table_text_tree_audit.json"
)
LABELED_CHILDREN = Path(
    "batch_table_text_tree/classifier_candidate_dataset/"
    "batch_table_text_classifier_parent_child_labeled.jsonl"
)
REVIEWED_CHILDREN = Path(
    "batch_table_text_tree/codex_child_review_results/"
    "all_codex_reviewed_child_blocks.json"
)


def _artifact_map(config: AgentConfig) -> dict[str, str]:
    root = config.workspace_root
    return {
        "manifest": str(config.resolved_manifest_path),
        "paragraph_tree_status": str(
            root / "batch_table_text_tree/local_paragraph_table_text_tree_status.json"
        ),
        "paragraph_audit": str(root / PARAGRAPH_AUDIT),
        "manual_annotations": str(
            root
            / "batch_table_text_tree/manual_table_body_text_annotations/"
            "batch_manual_table_body_text_annotations.json"
        ),
        "candidate_status": str(root / CANDIDATE_STATUS),
        "selection_status": str(root / SELECTION_STATUS),
        "review_package_status": str(root / REVIEW_STATUS),
        "review_result_status": str(root / REVIEW_RESULTS),
        "human_learning_feedback": str(config.resolved_learning_feedback_path),
        "active_candidate_policy": str(config.resolved_candidate_policy_path),
        "learning_output": str(config.resolved_learning_output_dir),
        "memory_db": str(config.resolved_memory_db_path),
        "core_memory": str(config.resolved_core_memory_path),
        "active_memory_pack": str(config.resolved_active_memory_pack_path),
    }


def decisions_ready(review_status_path: Path) -> tuple[bool, list[str]]:
    if not review_status_path.exists():
        return False, ["review package status is missing"]
    payload = load_json(review_status_path)
    missing = [
        document["expected_decisions"]
        for document in payload.get("documents", [])
        if not Path(document["expected_decisions"]).exists()
    ]
    return not missing, missing


def load_review_status(review_status_path: Path) -> dict[str, Any]:
    if not review_status_path.exists():
        raise FileNotFoundError(f"Review package status is missing: {review_status_path}")
    return load_json(review_status_path)


def documents_needing_codex_review(
    review_status_path: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    status = load_review_status(review_status_path)
    return [
        document
        for document in status.get("documents", [])
        if force or not Path(document["expected_decisions"]).exists()
    ]


def load_review_prompt(workspace_root: Path):
    module_path = workspace_root / "run_codex_child_semantic_review.py"
    module_name = "_tabref_project_codex_review"
    if module_name in sys.modules:
        return sys.modules[module_name].review_prompt
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Codex review prompt from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.review_prompt


def decide_review_action(
    review_mode: str,
    ready: bool,
    force_codex_review: bool = False,
) -> str:
    if review_mode == "prepare":
        return "stop"
    if review_mode == "codex":
        if force_codex_review:
            return "codex"
        return "materialize" if ready else "codex"
    if review_mode == "existing":
        return "materialize" if ready else "stop"
    if review_mode == "manual":
        return "manual"
    raise ValueError(f"Unsupported review mode: {review_mode}")


class AgentNodes:
    def __init__(self, config: AgentConfig, runner: ProjectToolRunner):
        self.config = config
        self.runner = runner
        self.review_prompt = None

    @staticmethod
    def _log_name(state: AgentState, step_name: str) -> str:
        job_id = state.get("job_id", "unscoped")
        safe_job_id = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in job_id
        )
        return f"{safe_job_id}_{step_name}"

    def validate_workspace(self, state: AgentState) -> dict[str, Any]:
        manifest_path = self.config.resolved_manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        required_scripts = [
            "build_local_paragraph_table_text_tree_batch.py",
            "audit_local_paragraph_table_text_trees.py",
            "materialize_manual_table_body_text_annotations.py",
            "build_table_text_classifier_candidates.py",
            "select_table_description_child_blocks.py",
            "prepare_codex_child_review_packages.py",
            "run_codex_child_semantic_review.py",
            "materialize_codex_child_review_results.py",
        ]
        missing_scripts = [
            name for name in required_scripts
            if not (self.config.workspace_root / name).exists()
        ]
        if missing_scripts:
            raise FileNotFoundError(f"Missing project tools: {missing_scripts}")

        manifest = load_json(manifest_path)
        documents = [document["slug"] for document in manifest.get("documents", [])]
        self.config.resolved_output_dir.mkdir(parents=True, exist_ok=True)
        active_memory = build_active_memory_pack(
            learning_root=self.config.resolved_learning_output_dir,
            db_path=self.config.resolved_memory_db_path,
            core_memory_path=self.config.resolved_core_memory_path,
            active_pack_path=self.config.resolved_active_memory_pack_path,
            query="table context evidence selection parent child threshold policy",
            task_key="table_context_selection",
            max_items=self.config.max_active_memory_items,
        )
        return {
            "stage": "validate_workspace",
            "status": "running",
            "documents": documents,
            "artifacts": _artifact_map(self.config),
            "metrics": {
                "document_count": len(documents),
                "active_memory_items": active_memory["budget"]["selected_items"],
                "available_memory_items": active_memory["budget"]["available_items"],
            },
            "events": append_event(
                state,
                "validate_workspace",
                "Manifest, project tools, and active memory pack validated.",
                document_count=len(documents),
                active_memory_items=active_memory["budget"]["selected_items"],
            ),
        }

    def build_evidence_trees(self, state: AgentState) -> dict[str, Any]:
        self.runner.run_python_script(
            self._log_name(state, "01_build_paragraph_trees"),
            "build_local_paragraph_table_text_tree_batch.py",
            ["--manifest", str(self.config.resolved_manifest_path)],
        )
        return {
            "stage": "build_evidence_trees",
            "events": append_event(
                state,
                "build_evidence_trees",
                "Built traceable MinerU paragraph and table-text trees.",
            ),
        }

    def audit_evidence_trees(self, state: AgentState) -> dict[str, Any]:
        self.runner.run_python_script(
            self._log_name(state, "02_audit_paragraph_trees"),
            "audit_local_paragraph_table_text_trees.py",
            ["--manifest", str(self.config.resolved_manifest_path)],
        )
        audit_path = self.config.workspace_root / PARAGRAPH_AUDIT
        audit = load_json(audit_path)
        failures = [
            item["slug"] for item in audit
            if item.get("status") != "pass"
        ]
        if failures:
            raise ValueError(f"Paragraph-tree audit failed: {failures}")
        metrics = dict(state.get("metrics", {}))
        metrics["paragraph_tree_audit_passed"] = len(audit)
        return {
            "stage": "audit_evidence_trees",
            "metrics": metrics,
            "events": append_event(
                state,
                "audit_evidence_trees",
                "All paragraph trees passed traceability audit.",
                passed=len(audit),
            ),
        }

    def materialize_annotations(self, state: AgentState) -> dict[str, Any]:
        result = self.runner.run_python_script(
            self._log_name(state, "03_materialize_annotations"),
            "materialize_manual_table_body_text_annotations.py",
            ["--manifest", str(self.config.resolved_manifest_path)],
        )
        summary = json.loads(result.stdout)
        metrics = dict(state.get("metrics", {}))
        metrics.update(
            {
                "table_count": summary.get("tables"),
                "positive_parent_relations": summary.get("evidence_paragraphs"),
            }
        )
        return {
            "stage": "materialize_annotations",
            "metrics": metrics,
            "events": append_event(
                state,
                "materialize_annotations",
                "Materialized the current human parent-paragraph annotations.",
                tables=summary.get("tables"),
                evidence_paragraphs=summary.get("evidence_paragraphs"),
            ),
        }

    def build_recall_candidates(self, state: AgentState) -> dict[str, Any]:
        candidate_result = self.runner.run_python_script(
            self._log_name(state, "04_build_candidates"),
            "build_table_text_classifier_candidates.py",
            ["--manifest", str(self.config.resolved_manifest_path)],
        )
        selection_result = self.runner.run_python_script(
            self._log_name(state, "05_select_children"),
            "select_table_description_child_blocks.py",
            ["--manifest", str(self.config.resolved_manifest_path)],
        )
        candidate_summary = json.loads(candidate_result.stdout)
        selection_summary = json.loads(selection_result.stdout)
        metrics = dict(state.get("metrics", {}))
        metrics.update(
            {
                "parent_candidates": candidate_summary.get("parent_candidates"),
                "child_candidates": candidate_summary.get("child_candidates"),
                "preserved_positive_child_candidates": selection_summary.get(
                    "child_label_0_correct"
                ),
                "review_candidates": selection_summary.get(
                    "semantic_review_candidates",
                    selection_summary.get("child_label_0_correct"),
                ),
            }
        )
        return {
            "stage": "build_recall_candidates",
            "metrics": metrics,
            "events": append_event(
                state,
                "build_recall_candidates",
                "Generated high-recall parent and child candidates.",
                parent_candidates=candidate_summary.get("parent_candidates"),
                child_candidates=candidate_summary.get("child_candidates"),
                review_candidates=selection_summary.get(
                    "semantic_review_candidates",
                    selection_summary.get("child_label_0_correct"),
                ),
            ),
        }

    def prepare_review_packages(self, state: AgentState) -> dict[str, Any]:
        result = self.runner.run_python_script(
            self._log_name(state, "06_prepare_review_packages"),
            "prepare_codex_child_review_packages.py",
        )
        summary = json.loads(result.stdout)
        metrics = dict(state.get("metrics", {}))
        metrics["review_items"] = summary.get("review_items")
        return {
            "stage": "prepare_review_packages",
            "metrics": metrics,
            "events": append_event(
                state,
                "prepare_review_packages",
                "Prepared complete table-parent-child packages for semantic review.",
                review_items=summary.get("review_items"),
            ),
        }

    def review_gate(self, state: AgentState) -> dict[str, Any]:
        status_path = self.config.workspace_root / REVIEW_STATUS
        ready, missing = decisions_ready(status_path)
        action = decide_review_action(
            self.config.review_mode,
            ready,
            force_codex_review=self.config.force_codex_review,
        )

        if action == "manual":
            response = interrupt(
                {
                    "kind": "tabref_semantic_review",
                    "message": "Complete or verify Codex decision files, then resume this thread.",
                    "review_package_status": str(status_path),
                    "missing_decision_files": missing,
                    "label_semantics": {
                        "0": "correct/relevant",
                        "1": "incorrect/irrelevant",
                    },
                }
            )
            approved = bool(
                response.get("approved") if isinstance(response, dict) else response
            )
            ready, missing = decisions_ready(status_path)
            action = "materialize" if approved and ready else "stop"

        if action == "stop":
            status = (
                "awaiting_codex_review"
                if self.config.review_mode == "prepare"
                else "blocked_missing_decisions"
            )
        else:
            status = "running"

        review_documents = []
        if action == "codex":
            review_documents = documents_needing_codex_review(
                status_path,
                force=self.config.force_codex_review,
            )

        return {
            "stage": "review_gate",
            "status": status,
            "next_action": action,
            "review_documents": review_documents,
            "parallel_review_results": [],
            "human_review": {
                "mode": self.config.review_mode,
                "decisions_ready": ready,
                "missing_decision_files": missing,
            },
            "events": append_event(
                state,
                "review_gate",
                f"Review routing selected: {action}.",
                review_mode=self.config.review_mode,
                decisions_ready=ready,
                parallel_documents=len(review_documents),
            ),
        }

    def review_one_paper(self, state: AgentState) -> dict[str, Any]:
        document = state["review_document"]
        slug = document["slug"]
        if self.review_prompt is None:
            self.review_prompt = load_review_prompt(self.config.workspace_root)
        command = [
            self.config.codex_command,
            "exec",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            self.review_prompt(document),
        ]
        self.runner.run_command(
            self._log_name(state, f"07_codex_review_{slug}"),
            command,
            cwd=self.config.workspace_root,
        )
        decision_path = Path(document["expected_decisions"])
        if not decision_path.exists():
            raise FileNotFoundError(
                f"Codex review did not create the expected decision file: {decision_path}"
            )
        return {
            "parallel_review_results": [
                {
                    "slug": slug,
                    "status": "pass",
                    "decision_path": str(decision_path),
                    "review_items": document.get("review_items"),
                }
            ]
        }

    def aggregate_codex_reviews(self, state: AgentState) -> dict[str, Any]:
        expected = {document["slug"] for document in state.get("review_documents", [])}
        results = state.get("parallel_review_results", [])
        completed = {result["slug"] for result in results if result.get("status") == "pass"}
        missing_results = sorted(expected - completed)
        if missing_results:
            raise ValueError(f"Parallel Codex reviews missing results: {missing_results}")

        ready, missing = decisions_ready(self.config.workspace_root / REVIEW_STATUS)
        if not ready:
            raise ValueError(f"Parallel Codex review completed without all decisions: {missing}")
        metrics = dict(state.get("metrics", {}))
        metrics.update(
            {
                "parallel_review_documents": len(results),
                "max_parallel_reviews": self.config.max_parallel_reviews,
            }
        )
        return {
            "stage": "aggregate_codex_reviews",
            "next_action": "materialize",
            "metrics": metrics,
            "events": append_event(
                state,
                "aggregate_codex_reviews",
                "All per-paper Codex reviews completed and passed the decision-file gate.",
                reviewed_documents=len(results),
                max_parallel_reviews=self.config.max_parallel_reviews,
            ),
        }

    def materialize_results(self, state: AgentState) -> dict[str, Any]:
        result = self.runner.run_python_script(
            self._log_name(state, "08_materialize_results"),
            "materialize_codex_child_review_results.py",
        )
        summary = json.loads(result.stdout)
        metrics = dict(state.get("metrics", {}))
        metrics.update(
            {
                "reviewed_children": summary.get("reviewed_children"),
                "retained_label_0": summary.get("final_label_0_retained"),
                "demoted_label_1": summary.get("final_label_1_demoted"),
            }
        )
        return {
            "stage": "materialize_results",
            "status": "running",
            "metrics": metrics,
            "events": append_event(
                state,
                "materialize_results",
                "Validated and materialized Codex semantic decisions.",
                reviewed_children=summary.get("reviewed_children"),
                retained_label_0=summary.get("final_label_0_retained"),
                demoted_label_1=summary.get("final_label_1_demoted"),
            ),
        }

    def optimize_candidate_policy(self, state: AgentState) -> dict[str, Any]:
        learning_root = self.config.resolved_learning_output_dir
        result = optimize_policy(
            candidates_path=self.config.workspace_root / LABELED_CHILDREN,
            feedback_path=self.config.resolved_learning_feedback_path,
            active_policy_path=self.config.resolved_candidate_policy_path,
            output_root=learning_root,
            min_examples=self.config.learning_min_examples,
            min_recall=self.config.learning_min_recall,
            min_table_coverage=self.config.learning_min_table_coverage,
        )
        learning = dict(state.get("learning", {}))
        learning["policy_optimization"] = result
        metrics = dict(state.get("metrics", {}))
        metrics["human_learning_examples"] = result["human_examples"]
        metrics["candidate_policy_status"] = result["status"]
        return {
            "stage": "optimize_candidate_policy",
            "learning": learning,
            "metrics": metrics,
            "events": append_event(
                state,
                "optimize_candidate_policy",
                f"Candidate-policy learning result: {result['status']}.",
                human_examples=result["human_examples"],
                policy_activated=result["policy_activated"],
            ),
        }

    def background_reflection(self, state: AgentState) -> dict[str, Any]:
        optimization = state.get("learning", {}).get("policy_optimization", {})
        reflection = build_reflection(
            feedback_path=self.config.resolved_learning_feedback_path,
            candidate_path=self.config.workspace_root / LABELED_CHILDREN,
            reviewed_path=self.config.workspace_root / REVIEWED_CHILDREN,
            learning_root=self.config.resolved_learning_output_dir,
            optimization=optimization,
            memory_db_path=self.config.resolved_memory_db_path,
            core_memory_path=self.config.resolved_core_memory_path,
            active_memory_pack_path=self.config.resolved_active_memory_pack_path,
            max_active_memory_items=self.config.max_active_memory_items,
        )
        learning = dict(state.get("learning", {}))
        learning["reflection"] = reflection
        metrics = dict(state.get("metrics", {}))
        active_pack = reflection.get("active_memory_pack", {})
        metrics["active_memory_items"] = active_pack.get(
            "selected_items",
            metrics.get("active_memory_items"),
        )
        return {
            "stage": "background_reflection",
            "learning": learning,
            "metrics": metrics,
            "events": append_event(
                state,
                "background_reflection",
                "Recorded human-grounded memory events, SQLite memories, and a post-run reflection.",
                human_feedback=reflection["human_feedback_count"],
                learned_rules=len(reflection["lessons"]),
                active_memory_items=active_pack.get("selected_items"),
            ),
        }

    def stage_skill_update(self, state: AgentState) -> dict[str, Any]:
        proposal = stage_skill_proposal(
            skill_dir=self.config.resolved_selector_skill_dir,
            learning_root=self.config.resolved_learning_output_dir,
            reflection=state.get("learning", {}).get("reflection", {}),
        )
        learning = dict(state.get("learning", {}))
        learning["skill_proposal"] = proposal
        return {
            "stage": "stage_skill_update",
            "learning": learning,
            "events": append_event(
                state,
                "stage_skill_update",
                f"Skill proposal result: {proposal['status']}.",
                proposal_id=proposal.get("proposal_id"),
                auto_apply_forbidden=True,
            ),
        }

    def finalize(self, state: AgentState) -> dict[str, Any]:
        output_dir = self.config.resolved_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        terminal_status = state.get("status", "completed")
        if terminal_status == "running":
            terminal_status = "completed"

        payload = {
            "schema_version": "1.0",
            "agent_version": "0.3.1",
            "job_id": state.get("job_id"),
            "thread_id": state.get("thread_id"),
            "status": terminal_status,
            "review_mode": self.config.review_mode,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "documents": state.get("documents", []),
            "metrics": state.get("metrics", {}),
            "artifacts": state.get("artifacts", {}),
            "human_review": state.get("human_review", {}),
            "learning": state.get("learning", {}),
            "events": state.get("events", []),
        }
        json_path = output_dir / f"{state['job_id']}_run_summary.json"
        md_path = output_dir / f"{state['job_id']}_run_summary.md"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# TabRef Context Agent Run",
            "",
            f"- Job: `{state['job_id']}`",
            f"- Thread: `{state['thread_id']}`",
            f"- Status: `{terminal_status}`",
            f"- Review mode: `{self.config.review_mode}`",
            "",
            "## Metrics",
            "",
        ]
        for key, value in payload["metrics"].items():
            lines.append(f"- {key}: {value}")
        lines.extend(["", "## Events", ""])
        for event in payload["events"]:
            lines.append(f"- `{event['stage']}`: {event['detail']}")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        artifacts = dict(state.get("artifacts", {}))
        artifacts["agent_run_summary_json"] = str(json_path)
        artifacts["agent_run_summary_markdown"] = str(md_path)
        return {
            "stage": "finalize",
            "status": terminal_status,
            "artifacts": artifacts,
            "events": append_event(
                state,
                "finalize",
                f"Run finalized with status {terminal_status}.",
            ),
        }
