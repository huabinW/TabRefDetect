from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    started_at_utc: str
    completed_at_utc: str


class ProjectToolRunner:
    """Run existing project scripts without reimplementing their algorithms."""

    def __init__(self, workspace_root: Path, output_dir: Path, python_executable: Path | None = None):
        self.workspace_root = workspace_root
        self.output_dir = output_dir
        self.python_executable = str(python_executable or Path(sys.executable))
        self.log_dir = output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_python_script(
        self,
        name: str,
        script_name: str,
        arguments: list[str] | None = None,
    ) -> CommandResult:
        script_path = self.workspace_root / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Project tool not found: {script_path}")
        command = [self.python_executable, str(script_path), *(arguments or [])]
        return self.run_command(name, command, cwd=self.workspace_root)

    def run_command(
        self,
        name: str,
        command: list[str],
        cwd: Path | None = None,
    ) -> CommandResult:
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(
            command,
            cwd=cwd or self.workspace_root,
            check=False,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        result = CommandResult(
            name=name,
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            started_at_utc=started,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        log_path = self.log_dir / f"{name}.json"
        log_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{name} failed with exit code {result.returncode}. "
                f"See {log_path}"
            )
        return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def append_event(state: dict[str, Any], stage: str, detail: str, **extra: Any) -> list[dict[str, Any]]:
    events = list(state.get("events", []))
    event = {
        "stage": stage,
        "detail": detail,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    event.update(extra)
    events.append(event)
    return events
