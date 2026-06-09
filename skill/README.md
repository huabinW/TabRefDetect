# TabRefDetect Codex Skills

This directory contains Codex Skills used by the MinerU + PageIndex table-text
tree workflow.

## Included Skills

- `tabref-table-tree-audit`: audit table counts, positions, traceability, and
  PageIndex section assignments.
- `tabref-table-caption-resolver`: recover and audit table numbers and captions
  from MinerU output.
- `tabref-table-text-child-selector`: generate high-recall child candidates and
  perform Codex semantic precision review.

## Installation

Copy a Skill directory into your Codex Skills directory:

```powershell
Copy-Item -Recurse skill\<skill-name> $env:USERPROFILE\.codex\skills\
```

Run commands from the TabRefDetect project root so bundled wrappers can locate
the corresponding project scripts.

The child-selector wrappers expect the associated project-owned pipeline
scripts to be present in the working project. The Skill repository does not
include datasets, OCR outputs, review packages, or paper-specific annotations.
