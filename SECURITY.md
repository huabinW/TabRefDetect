# Security Policy

## Supported Scope

This repository contains research code, configuration examples, Codex Skills, and workflow-agent code. Security reports are most relevant when they involve:

- Accidental exposure of API keys, access tokens, server credentials, or cloud storage credentials.
- Data leakage through committed PDFs, OCR outputs, prompts, generated model responses, or private annotations.
- Unsafe default behavior in scripts that could overwrite user files or upload data unexpectedly.
- Dependency or execution issues that create a realistic risk for users running the released code.

## Reporting a Vulnerability

Please do not disclose sensitive issues in a public GitHub issue. Use GitHub's private vulnerability reporting feature when available, or contact the repository owner privately through GitHub.

When reporting, include:

- A clear description of the risk.
- The affected file or command.
- Steps to reproduce, if safe.
- Whether private data, credentials, or non-redistributable PDFs may be involved.

## Data Handling Expectations

Contributors should not commit:

- Private datasets or unpublished annotations.
- Original PDFs unless redistribution is allowed.
- OCR outputs or prompts that reveal paper-specific content.
- Local runtime outputs, checkpoints, logs, or cache files.
- API keys, access tokens, passwords, or server addresses.

The released code should prefer example configs and synthetic data placeholders over machine-specific paths.
