"""Run the repository-owned Codex review package builder."""

import runpy
import sys
from pathlib import Path


SCRIPT = (
    Path.cwd()
    / "Code"
    / "MinerU_PageIndex_TableTree"
    / "prepare_codex_child_review_packages.py"
)
if not SCRIPT.exists():
    raise SystemExit(f"Run from the TabRefDetect repository root; missing {SCRIPT}")
sys.path.insert(0, str(SCRIPT.parent))
runpy.run_path(str(SCRIPT), run_name="__main__")
