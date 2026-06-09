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
candidate_signals
```

Label semantics:

- `0`: correct/relevant table-text relation.
- `1`: incorrect/irrelevant table-text relation.

Validation requirements:

- `child_text == full_parent_text[char_start:char_end]`.
- Incorrect parents cannot contain children labeled `0`.
- Correct parents must contain at least one child labeled `0`.
- Human gold and provisional Skill predictions must remain distinguishable.

Codex review fields:

```text
codex_review.codex_label
codex_review.semantic_role
codex_review.citation_support
codex_review.rationale
final_child_label
final_child_label_source
```

Human-readable Markdown requirements:

- Group by paper ID, then canonical table ID.
- List Codex-retained label `0` children first.
- List Codex-demoted label `1` children separately after each table.
- Include the rule score as a review aid; do not describe it as probability or confidence.
- Include the Codex semantic role and rationale, but omit bulky bbox metadata.
