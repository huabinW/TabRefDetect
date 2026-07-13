# V2 data contract

## Candidate inventory

Each paper inventory contains:

- auditable table anchors and the matched Popo table nodes;
- Popo text parents plus MinerU text blocks not represented by Popo;
- exact child offsets within each complete parent;
- one copy of every child span;
- stable section/scope ids and non-binding table suggestions;
- source hashes and source-coverage statistics.

`review_eligible_table_ids` always lists every table in the paper. Suggested table ids and scores
are ordering metadata only.

## Subagent decision

Each child decision contains:

- `child_id` and `parent_id`;
- `relevant_tables`, each with table id, label `0`, confidence, evidence role, and rationale;
- an empty list when the child supplements no table;
- `rejection_reason` when no relation is retained.

Reviewers must apply the supplementary-context test. Captions are table anchors and are never
semantic child decisions.

## Human annotation

The compact human template contains one record per retained table-child relation:

- paper id, table id/label/caption/body;
- complete parent and exact child offsets/text;
- subagent confidence and role;
- empty `human_label` and `human_rationale` fields.

The full audit output separately preserves every parent, child, table, suggestion, and decision.
