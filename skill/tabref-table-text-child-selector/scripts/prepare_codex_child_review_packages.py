"""Run the project-owned Codex review package builder."""

import runpy
from pathlib import Path


SCRIPT = Path.cwd() / "prepare_codex_child_review_packages.py"
if not SCRIPT.exists():
    raise SystemExit(f"Run from the TabRefDetect project root; missing {SCRIPT}")
runpy.run_path(str(SCRIPT), run_name="__main__")
