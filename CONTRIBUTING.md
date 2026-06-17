# Contributing to TabRefDetect

Thank you for your interest in TabRefDetect. This repository is a research codebase for numerical discrepancy detection and table-context analysis in scientific papers.

## What Contributions Are Welcome

- Bug fixes for reusable scripts in `Code/`.
- Improvements to the MinerU + PageIndex table-text tree pipeline.
- Agent workflow improvements under `agent/tabref_context_agent/`.
- Documentation, examples, and reproducibility improvements.
- New evaluation scripts that operate on user-provided or synthetic data.
- Model training utilities that do not include private datasets, credentials, or generated predictions.

## Data and Privacy Rules

Do not include:

- Full private datasets or unpublished annotations.
- Original PDFs unless they are explicitly redistributable.
- OCR outputs, prompts, model responses, or review packages containing paper-specific content.
- API keys, access tokens, cloud storage credentials, local machine paths, or server addresses.
- Checkpoints, large model weights, cache directories, or experiment logs.

Use small synthetic examples whenever possible. If a real example is necessary, make sure it is publicly redistributable and does not expose sensitive annotations.

## Development Setup

Clone the repository and install dependencies as needed for the component you are working on:

```bash
pip install -r requirements.txt
```

For the LangGraph agent:

```bash
cd agent/tabref_context_agent
pip install -e .
python -m pytest -q
```

Some training scripts require GPU-specific environments. Please document the expected environment in the relevant README when adding new scripts.

## Code Style

- Keep paths configurable through command-line arguments or example config files.
- Preserve traceability fields such as page index, bbox, content index, table label, and assignment reason.
- Avoid silently summarizing or rewriting OCR evidence.
- Keep deterministic preprocessing separate from LLM or model-based judgment.
- Add tests for reusable logic when the change affects selection rules, labels, routing, or output schemas.

## Pull Request Checklist

Before opening a pull request:

- Confirm that no private data or credentials are included.
- Run the relevant tests or explain why they could not be run.
- Update README files when commands, inputs, outputs, or labels change.
- Keep generated outputs out of the repository unless they are tiny, synthetic, and useful as examples.
- Describe whether the change affects label semantics, candidate generation, agent memory, or Skill approval behavior.

## Label Convention

For table-context child selection, this project uses:

- `0`: correct or relevant evidence.
- `1`: incorrect or irrelevant evidence.

Human annotations are the only gold labels. Model or Codex outputs should be treated as provisional unless evaluated against human gold.
