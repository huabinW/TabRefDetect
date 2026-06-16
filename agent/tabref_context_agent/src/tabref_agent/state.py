from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    schema_version: str
    job_id: str
    thread_id: str
    review_mode: str
    learning_mode: str
    stage: str
    status: str
    next_action: str
    workspace_root: str
    manifest_path: str
    output_dir: str
    documents: list[str]
    review_documents: list[dict[str, Any]]
    review_document: dict[str, Any]
    parallel_review_results: Annotated[list[dict[str, Any]], operator.add]
    artifacts: dict[str, str]
    metrics: dict[str, Any]
    events: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    human_review: dict[str, Any]
    learning: dict[str, Any]
