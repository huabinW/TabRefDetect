from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


ReviewMode = Literal["prepare", "codex", "existing", "manual"]
LearningMode = Literal["off", "analyze", "propose"]


class AgentConfig(BaseModel):
    workspace_root: Path
    manifest_path: Path = Path("table_tree_batch_manifest.json")
    output_dir: Path = Path("batch_table_text_tree/agent_v1")
    checkpoint_db: Path = Path("batch_table_text_tree/agent_v1/checkpoints.sqlite")
    review_mode: ReviewMode = "prepare"
    codex_command: str = "codex"
    max_parallel_reviews: int = Field(default=2, ge=1, le=8)
    force_codex_review: bool = False
    learning_mode: LearningMode = "off"
    learning_feedback_path: Path = Path(
        "batch_table_text_tree/learning/human_child_feedback.jsonl"
    )
    candidate_policy_path: Path = Path(
        "batch_table_text_tree/learning/candidate_policy/active.json"
    )
    learning_output_dir: Path = Path("batch_table_text_tree/learning")
    selector_skill_dir: Path = Field(
        default_factory=lambda: (
            Path.home() / ".codex/skills/tabref-table-text-child-selector"
        )
    )
    learning_min_examples: int = Field(default=30, ge=1)
    learning_min_recall: float = Field(default=0.98, ge=0.0, le=1.0)
    learning_min_table_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    python_executable: Path | None = None
    schema_version: str = Field(default="1.0", frozen=True)

    @field_validator("workspace_root")
    @classmethod
    def normalize_workspace(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    def resolve_path(self, value: Path) -> Path:
        return value if value.is_absolute() else self.workspace_root / value

    @property
    def resolved_manifest_path(self) -> Path:
        return self.resolve_path(self.manifest_path)

    @property
    def resolved_output_dir(self) -> Path:
        return self.resolve_path(self.output_dir)

    @property
    def resolved_checkpoint_db(self) -> Path:
        return self.resolve_path(self.checkpoint_db)

    @property
    def resolved_learning_feedback_path(self) -> Path:
        return self.resolve_path(self.learning_feedback_path)

    @property
    def resolved_candidate_policy_path(self) -> Path:
        return self.resolve_path(self.candidate_policy_path)

    @property
    def resolved_learning_output_dir(self) -> Path:
        return self.resolve_path(self.learning_output_dir)

    @property
    def resolved_selector_skill_dir(self) -> Path:
        return (
            self.selector_skill_dir.expanduser().resolve()
            if self.selector_skill_dir.is_absolute()
            else self.resolve_path(self.selector_skill_dir)
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "AgentConfig":
        config_path = Path(path).expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        workspace = Path(payload["workspace_root"]).expanduser()
        if not workspace.is_absolute():
            payload["workspace_root"] = str((config_path.parent / workspace).resolve())
        return cls.model_validate(payload)
