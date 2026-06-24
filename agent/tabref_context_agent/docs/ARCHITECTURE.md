# TabRef Context Agent v0.3 Architecture

## System Boundary

The agent orchestrates existing project tools. It does not independently
replace MinerU, invent a PageIndex tree, or treat Codex decisions as human gold.

### Inputs

- `table_tree_batch_manifest.json`
- Existing MinerU outputs
- Reviewed document structure trees
- Current manual parent-paragraph selections
- Optional existing Codex decision files
- Optional human child-level feedback JSONL

### Outputs

- Refreshed paragraph and table-text trees
- Traceability audit reports
- Parent and child candidate datasets
- Codex review packages
- Validated semantic decisions
- Agent run summaries and SQLite checkpoints
- Human-grounded learning memory, reflections, and candidate policies
- Pending Skill proposals that require approval before installation

## Node Responsibilities

1. `validate_workspace`: validates the manifest, required project scripts, and document list.
2. `build_evidence_trees`: calls `build_local_paragraph_table_text_tree_batch.py`.
3. `audit_evidence_trees`: blocks downstream work on a failed traceability check.
4. `materialize_annotations`: converts manual parent selections into the project annotation schema.
5. `build_recall_candidates`: preserves every child under a correct parent and marks `send_to_semantic_review`.
6. `prepare_review_packages`: packages table, parent, child, hashes, and traceability fields for review.
7. `review_gate`: routes execution according to `prepare`, `codex`, `existing`, or `manual`.
8. `review_one_paper`: receives one paper through LangGraph `Send` and invokes local `codex exec`.
9. `aggregate_codex_reviews`: waits for every paper branch and validates decision files.
10. `materialize_results`: validates and writes final semantic decisions.
11. `optimize_candidate_policy`: updates rule weights and thresholds only from human feedback when guardrails pass.
12. `background_reflection`: records memory events and a compact reflection.
13. `stage_skill_update`: creates a pending Skill proposal without modifying the live Skill.
14. `finalize`: writes a compact JSON and Markdown run report.

## State Design

Graph state stores only job identity, paths, aggregate metrics, routing status,
learning summaries, and events. OCR text, table HTML, all candidates, learning
memory, and Skill proposals remain in versioned artifacts. This avoids oversized
checkpoints and keeps every transformation independently auditable.

## Learning Contract

The learning loop has two update surfaces:

- Candidate policy: may update automatically from human feedback after sample,
  recall, and per-table coverage guardrails pass. It controls only
  `send_to_semantic_review`.
- Skill update: never automatic. The agent writes a pending proposal, and a
  human must run `approve-skill`. Approval first saves a full live Skill
  snapshot, then copies only allowlisted files from the proposal.

Codex decisions can be compared against human gold, but they cannot become gold.

## Reliability Decisions

- SQLite checkpoints support thread inspection and manual resume.
- LangGraph `Send` fans out only independent paper-review tasks.
- `max_parallel_reviews` limits simultaneous Codex processes.
- Existing scripts remain the source of deterministic preprocessing behavior.
- Semantic decisions are hash-bound to the exact child text.
- A failed tree audit stops the graph before semantic review.
- Candidate-policy activation is blocked when human feedback is insufficient.
- Live Skill mutation is blocked unless the approval CLI is invoked.

## Current Non-Goals

- PDF upload and MinerU execution
- PageIndex MCP invocation
- Child-level model training
- Automatic citation-error judgment across citing and cited papers
- Distributed workers or production authentication
- Open-ended autonomous tool selection
