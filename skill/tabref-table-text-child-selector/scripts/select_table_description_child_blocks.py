"""Run the project-owned high-recall child candidate selector."""

import runpy
from pathlib import Path


SCRIPT = Path.cwd() / "select_table_description_child_blocks.py"
if not SCRIPT.exists():
    raise SystemExit(f"Run from the TabRefDetect project root; missing {SCRIPT}")
runpy.run_path(str(SCRIPT), run_name="__main__")
