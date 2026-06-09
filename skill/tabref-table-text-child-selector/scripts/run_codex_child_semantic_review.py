"""Run the project-owned Codex semantic review launcher."""

import runpy
from pathlib import Path


SCRIPT = Path.cwd() / "run_codex_child_semantic_review.py"
if not SCRIPT.exists():
    raise SystemExit(f"Run from the TabRefDetect project root; missing {SCRIPT}")
runpy.run_path(str(SCRIPT), run_name="__main__")
