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


def test_memory_paths_are_resolved_under_workspace(tmp_path: Path):
    config = AgentConfig(workspace_root=tmp_path)
    memory_root = tmp_path / "batch_table_text_tree" / "learning" / "memory"
    assert config.resolved_memory_db_path == memory_root / "memory.sqlite"
    assert config.resolved_core_memory_path == memory_root / "core_memory.json"
    assert (
        config.resolved_active_memory_pack_path
        == memory_root / "active_memory_pack.json"
    )


def test_json_workspace_root_is_relative_to_config_file(tmp_path: Path):
    config_dir = tmp_path / "agent"
    workspace = tmp_path / "Code" / "MinerU_PageIndex_TableTree"
    config_dir.mkdir()
    workspace.mkdir(parents=True)
    config_path = config_dir / "config.example.json"
    config_path.write_text(
        '{"workspace_root": "../Code/MinerU_PageIndex_TableTree"}',
        encoding="utf-8",
    )

    config = AgentConfig.from_json(config_path)
    assert config.workspace_root == workspace.resolve()
