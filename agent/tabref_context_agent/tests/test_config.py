from pathlib import Path

from tabref_agent.config import AgentConfig


def test_relative_paths_are_resolved_under_workspace(tmp_path: Path):
    config = AgentConfig(workspace_root=tmp_path)
    assert config.resolved_manifest_path == tmp_path / "table_tree_batch_manifest.json"
    assert config.resolved_output_dir == tmp_path / "batch_table_text_tree" / "agent_v1"


def test_absolute_path_is_preserved(tmp_path: Path):
    absolute = tmp_path / "custom.json"
    config = AgentConfig(workspace_root=tmp_path, manifest_path=absolute)
    assert config.resolved_manifest_path == absolute

