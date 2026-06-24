# Output Contract

Each training record must contain:

```text
slug
table
candidate_id
manual_parent_label
parent.parent_paragraph_id
parent.full_parent_text
parent.raw_content_indices
parent.page_indices
parent.parent_node_id
parent.parent_title
child.child_id
child.char_start
child.char_end
child.child_text
child.child_label
child.child_label_source
child.child_review_status
child.child_selection_signals
child.send_to_semantic_review
child.candidate_policy_version
candidate_signals
```

Label semantics:

- `0`: correct/relevant table-text relation.
- `1`: incorrect/irrelevant table-text relation.

Validation requirements:

- `child_text == full_parent_text[char_start:char_end]`.
- Incorrect parents cannot contain children labeled `0`.
- Correct parents must contain at least one child labeled `0`.
- Correct parents must contain at least one child with `send_to_semantic_review = true`.
- Human gold and provisional Skill predictions must remain distinguishable.
- `send_to_semantic_review` controls the review queue only; it does not delete or relabel preserved child evidence.

Codex review fields:

```text
codex_review.codex_label
codex_review.semantic_role
codex_review.citation_support
codex_review.rationale
final_child_label
final_child_label_source
```

For the table-scope precision-v2 stage:

```text
final_child_label_source = codex_table_scope_precision_review_v2
```

For the manually reaudited supplementary-context stage:

```text
final_child_label_source = codex_supplementary_context_reaudit_v3
```

Precision-v2 validation requirements:

- Every strict high-recall child must receive exactly one Codex decision.
- Existing human labels are binding and a Codex decision cannot contradict them.
- Label `1` must use `semantic_role = irrelevant` and `citation_support = none`.
- A table may retain zero Codex label-`0` children.
- The retained human template is a view over all reviewed children; it must not replace or delete the complete reviewed artifact.
- Direct result restatements and promotional efficacy claims are label `1`.
- Result interpretations are label `0` only when they add a necessary limitation, metric behavior, dataset difference, or cross-study qualification not visible in the table.

Learning and approval artifacts:

```text
learning/human_child_feedback.jsonl
learning/memory/events.jsonl
learning/memory/lessons.json
learning/reflections/*.json
learning/candidate_policy/active.json
learning/candidate_policy/history/*.json
learning/skill_proposals/pending/*/manifest.json
learning/skill_proposals/approved/*/manifest.json
learning/skill_proposals/rejected/*/manifest.json
learning/skill_history/tabref-table-text-child-selector/*
```

Human feedback JSONL rows must contain:

```text
slug
table_label
candidate_id
child_id
gold_label
```

Optional feedback fields include `error_category`, `comment`, `annotator`, and
`timestamp`. `gold_label` keeps the project convention: `0` means relevant and
`1` means irrelevant.

Human-readable Markdown requirements:

- Group by paper ID, then canonical table ID.
- List Codex-retained label `0` children first.
- List Codex-demoted label `1` children separately after each table.
- Include the rule score as a review aid; do not describe it as probability or confidence.
- Include the Codex semantic role and rationale, but omit bulky bbox metadata.
