# Controlled Self-Learning

## Purpose

The self-learning loop reduces semantic-review workload without weakening
evidence recall. It borrows three ideas from Hermes-style agents:

- Memory: preserve corrections and lessons across runs.
- Skill self-update: propose better instructions after repeated human feedback.
- Approval gate: never rewrite operating instructions without human review.

## What Can Learn Automatically

Only the candidate policy can update automatically. The policy controls signal
weights, review threshold, and force-review signals. It affects
`send_to_semantic_review`; it does not remove child blocks from the full
candidate dataset and does not change label semantics.

Automatic activation requires:

- At least `learning_min_examples` human examples.
- At least one positive and one negative human example.
- Recall greater than or equal to `learning_min_recall`.
- Per-table coverage greater than or equal to `learning_min_table_coverage`.

If any condition fails, the agent writes a skipped-learning reflection and keeps
the active policy unchanged.

## What Requires Approval

Any Skill text update requires explicit approval. In `learning_mode = propose`,
the graph can write:

```text
batch_table_text_tree\learning\skill_proposals\pending\<proposal-id>
```

The pending proposal contains a manifest and proposed files. It does not change
the live Skill.

Approval is a separate command:

```powershell
.\tabref_context_agent\.venv\Scripts\python.exe -m tabref_agent.cli approve-skill `
  --config .\tabref_context_agent\config.local.json `
  --proposal-id <proposal-id> `
  --approver <name>
```

Approval checks the live base version, saves a complete Skill snapshot, and
copies only allowlisted files. Rejecting a proposal moves it to the rejected
folder and leaves the live Skill untouched.

## Memory Artifacts

```text
learning\human_child_feedback.jsonl
learning\memory\events.jsonl
learning\memory\lessons.json
learning\reflections\*.json
learning\candidate_policy\active.json
learning\candidate_policy\history\*.json
learning\skill_proposals\pending\*
learning\skill_proposals\approved\*
learning\skill_proposals\rejected\*
learning\skill_history\tabref-table-text-child-selector\*
```

## Feedback Schema

Required fields:

```json
{
  "slug": "paper-id",
  "table_label": "Table 1",
  "candidate_id": "table-1-paragraph-0001",
  "child_id": "paragraph-0001-child-001",
  "gold_label": 0
}
```

Optional fields include `error_category`, `comment`, `annotator`, and
`timestamp`. `gold_label = 0` means relevant. `gold_label = 1` means irrelevant.

## Why Codex Labels Are Not Gold

Codex labels are useful silver labels for bootstrapping and comparison. They
are not treated as training gold because the second stage is exactly the part
being evaluated and improved. Human feedback is the only source that can change
the active candidate policy or support a Skill proposal.

## Safe Workflow

1. Run the pipeline in `off` mode for a stable baseline.
2. Add human feedback JSONL.
3. Switch config to `learning_mode = analyze`.
4. Run `existing` mode to reuse decisions and evaluate learning.
5. Inspect active policy and reflection artifacts.
6. Switch to `learning_mode = propose` when lessons should update the Skill.
7. Review the pending proposal manually.
8. Approve or reject it with the CLI.
9. Re-run tests and the versioned regression corpus with the normal prompt blind
   to previous decisions; keep user-confirmed holdout papers separate.

This keeps automatic learning useful but boxed in: it can tune cheap candidate
rules, while human approval remains the only path to changing the agent's
procedural knowledge.
