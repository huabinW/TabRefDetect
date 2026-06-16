# TabRef Context Agent

`tabref-context-agent` is a LangGraph workflow agent for auditable table-context
identification in the TabRefDetect project.

The agent orchestrates deterministic document-analysis scripts, prepares
table-parent-child review packages, routes semantic review to Codex or existing
decision files, and supports a controlled self-learning loop from human
child-level feedback.

## Key Features

- Explicit LangGraph `StateGraph` workflow.
- SQLite checkpointing for resumable long-running jobs.
- Paper-level parallel review with LangGraph `Send`.
- Deterministic high-recall child candidate preservation.
- Separate `send_to_semantic_review` flag for semantic-review queue control.
- Human-feedback learning for candidate-policy weights and thresholds.
- Pending Skill proposal workflow with explicit human approval.

## What Is Not Included

This release contains reusable agent source code, tests, and a sanitized config
template only. It does not include PDFs, OCR outputs, generated trees, review
packages, model decisions, annotations, checkpoints, or local runtime files.

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .
```

On Linux or macOS, use the corresponding virtual-environment activation path.

## Configure

Copy the example config and adjust paths for your local project layout:

```bash
cp config.example.json config.local.json
```

Important fields:

- `workspace_root`: project workspace containing the document-analysis scripts.
- `manifest_path`: batch manifest for the papers to process.
- `review_mode`: `prepare`, `codex`, `existing`, or `manual`.
- `learning_mode`: `off`, `analyze`, or `propose`.
- `selector_skill_dir`: local Codex Skill directory used for semantic review.

`config.local.json` should remain local and must not contain secrets.

## Commands

Inspect the resolved plan:

```bash
python -m tabref_agent.cli plan --config config.local.json
```

Prepare review packages without calling a model:

```bash
python -m tabref_agent.cli run --config config.local.json --mode prepare
```

Reuse existing decision files:

```bash
python -m tabref_agent.cli run --config config.local.json --mode existing
```

Inspect learning status:

```bash
python -m tabref_agent.cli learning-status --config config.local.json
```

Approve a pending Skill proposal after manual review:

```bash
python -m tabref_agent.cli approve-skill \
  --config config.local.json \
  --proposal-id <proposal-id> \
  --approver <name>
```

## Tests

```bash
python -m pytest -q
```

## Label Convention

- `0`: correct or relevant table-context evidence.
- `1`: incorrect or irrelevant evidence.

Human annotations are the only gold labels. Codex or local model decisions are
treated as provisional labels until evaluated against human feedback.
