"""Run the repository-owned Codex decision validator and materializer."""

import runpy
import sys
from pathlib import Path


SCRIPT = (
    Path.cwd()
    / "Code"
    / "MinerU_PageIndex_TableTree"
    / "materialize_codex_child_review_results.py"
)
if not SCRIPT.exists():
    raise SystemExit(f"Run from the TabRefDetect repository root; missing {SCRIPT}")
sys.path.insert(0, str(SCRIPT.parent))
runpy.run_path(str(SCRIPT), run_name="__main__")
