# API Clients

This directory contains minimal API client examples for closed-source model experiments used in TabRefDetect.

## Important Safety Notes

- Replace placeholder values such as `your_api_key` with environment variables or local-only config files.
- Do not commit real API keys, access tokens, OSS credentials, endpoints tied to private infrastructure, prompts containing dataset content, or model responses.
- Keep generated outputs outside the repository unless they are synthetic examples.

## Contents

- `Gemini/`: Gemini API example.
- `GLM/`: GLM API and OSS upload example.
- `Qwen/`: Qwen API and OSS upload example.

These scripts are examples. For reproducible experiments, record model names, dates, prompts, and evaluation settings in a separate local experiment log that does not expose private data.
