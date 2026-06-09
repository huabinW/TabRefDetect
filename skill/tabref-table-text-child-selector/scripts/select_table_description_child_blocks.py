"""Run the repository-owned high-recall child candidate selector."""

import runpy
import sys
from pathlib import Path


SCRIPT = (
    Path.cwd()
    / "Code"
    / "MinerU_PageIndex_TableTree"
    / "select_table_description_child_blocks.py"
)
if not SCRIPT.exists():
    raise SystemExit(f"Run from the TabRefDetect repository root; missing {SCRIPT}")
sys.path.insert(0, str(SCRIPT.parent))
runpy.run_path(str(SCRIPT), run_name="__main__")
