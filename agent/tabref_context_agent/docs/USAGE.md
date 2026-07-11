# TabRef Context Agent Usage

## 1. Environment

Create a local virtual environment from the repository root:

```powershell
cd <repo-root>
py -3.10 -m venv .\agent\tabref_context_agent\.venv
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m pip install `
  -e .\agent\tabref_context_agent
```

Important config settings:

```json
{
  "max_parallel_reviews": 2,
  "learning_mode": "off",
  "learning_min_examples": 30,
  "learning_min_recall": 0.98,
  "learning_min_table_coverage": 1.0
}
```

## 2. Inspect the Plan

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli plan `
  --config .\agent\tabref_context_agent\config.local.json `
  --mode prepare
```

## 3. Prepare Without a Model

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli run `
  --config .\agent\tabref_context_agent\config.local.json `
  --mode prepare `
  --thread-id paper-batch-prepare-001
```

Expected terminal state: `awaiting_codex_review`.

## 4. Run Through Codex

Confirm that `codex` is available:

```powershell
Get-Command codex
```

Then run:

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli run `
  --config .\agent\tabref_context_agent\config.local.json `
  --mode codex `
  --thread-id paper-batch-codex-001
```

To regenerate existing decision files, add `--force-codex-review`.

### Preferred Subagent Review

For independent paper packages, let the current desktop conversation coordinate
the Codex subagents. The number of subagents is dynamic and may be smaller than,
equal to, or larger than the paper count depending on package independence,
available concurrency, token budget, and retry state:

1. Run prepare mode or `prepare_popo_codex_precision_review_packages.py`.
2. Read the installed `tabref-table-text-child-selector` Skill.
3. Open `references/subagent-review-prompt.md` from the Skill.
4. Assign every package a unique decision path. Spawn one subagent per package
   when practical, or queue packages through fewer subagents.
5. Fill only `{PACKAGE_JSON_OR_MD}`, `{DECISION_JSON}`, and `{SLUG}` in the
   standard prompt.
6. Track expected, completed, missing, and invalid decision files in the package
   status or coordinator manifest.
7. Run the materializer in existing-decision mode only after every expected
   decision JSON passes validation.

Do not put old decision files, retained counts, or paper-specific conclusions
into a normal review prompt. Use previous outputs only after materialization,
in an explicit comparison/audit pass.

When model usage is constrained, stop after package preparation. On resume,
inspect the package-status or coordinator manifest, assign only missing or
invalid decision files, and skip packages that already have valid decisions.

## 5. Reuse Existing Decisions

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli run `
  --config .\agent\tabref_context_agent\config.local.json `
  --mode existing `
  --thread-id paper-batch-existing-001
```

For the Popo strict review scripts, materialize existing package decisions with:

```powershell
python .\materialize_popo_codex_precision_review_results.py `
  --decision-dir <review-package-dir> `
  --output-dir <result-dir>
```

The materializer validates `review_key`, `child_id`, text hashes, label schema,
role/support consistency, and existing human gold constraints before writing
outputs.

## 5.1 Human Annotation Outputs

The primary file for human annotation is the slim template:

```text
batch_popo_codex_supplementary_v3_human_child_annotation_template.slim.json
```

It keeps only the fields needed for annotation:

- table label, caption, page, bbox, and table body;
- full parent text and parent trace keys;
- child text, offsets, hashes, and code score;
- Codex label/role/support/rationale;
- empty `human_label` and `human_rationale` fields.

Caption fields are table anchors. They should be present for every table, but
they are not child candidates. Full traceability remains in:

```text
all_*_codex_supplementary_v3_reviewed.json
codex_supplementary_v3_retained_children.json
codex_supplementary_v3_demoted_children.json
batch_popo_codex_supplementary_v3_full_audit_template.json
```

## 6. Enable Learning

Learning is controlled by `learning_mode` in the config:

- `off`: no learning.
- `analyze`: evaluate human feedback, update candidate policy if guardrails pass,
  and write memory/reflection artifacts.
- `propose`: also stage a pending Skill proposal if human-supported lessons exist.

Human feedback goes to:

```text
batch_table_text_tree\learning\human_child_feedback.jsonl
```

Each row must include:

```json
{
  "slug": "paper-id",
  "table_label": "Table 1",
  "candidate_id": "table-1-paragraph-0001",
  "child_id": "paragraph-0001-child-001",
  "gold_label": 0
}
```

Inspect learning status:

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli learning-status `
  --config .\agent\tabref_context_agent\config.local.json
```

Approve a pending Skill proposal only after reviewing its files:

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli approve-skill `
  --config .\agent\tabref_context_agent\config.local.json `
  --proposal-id <proposal-id> `
  --approver huabin
```

Reject a proposal:

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli reject-skill `
  --config .\agent\tabref_context_agent\config.local.json `
  --proposal-id <proposal-id> `
  --approver huabin `
  --reason "Needs clearer rule wording"
```

Approval always saves a full live Skill snapshot before applying allowlisted
files. The graph itself cannot auto-approve proposals.

## 7. Manual Interrupt and Resume

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli run `
  --config .\agent\tabref_context_agent\config.local.json `
  --mode manual `
  --thread-id paper-batch-manual-001
```

After decision files are completed:

```powershell
.\agent\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli resume `
  --config .\agent\tabref_context_agent\config.local.json `
  --thread-id paper-batch-manual-001 `
  --approve
```

## 8. Tests

```powershell
cd <repo-root>\agent\tabref_context_agent
.\.venv\Scripts\python.exe -m pytest -q
```

## 9. Optional API Backend

V0.3 intentionally uses Codex CLI for semantic review. A later model gateway can
implement the same decision JSON schema using an OpenAI-compatible API, a local
reranker, or a fine-tuned classifier without changing downstream materialization.
