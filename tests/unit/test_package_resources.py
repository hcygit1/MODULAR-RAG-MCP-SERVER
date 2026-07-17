"""Installed-package resource contracts."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_packaged_settings_load_without_repository_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings_module = importlib.import_module(
        "modular_rag_mcp_server.core.settings"
    )

    settings = settings_module.load_settings()

    assert settings.vector_store.collection_name == "cslearning"
    assert settings.ingestion.chunk_size > 0


@pytest.mark.parametrize(
    "prompt_name",
    [
        "boundary_repair.txt",
        "chunk_refinement.txt",
        "image_captioning.txt",
        "rerank.txt",
    ],
)
def test_packaged_prompts_are_non_empty(prompt_name: str) -> None:
    resource_module = importlib.import_module(
        "modular_rag_mcp_server.package_resources"
    )

    prompt = resource_module.read_prompt(prompt_name)

    assert prompt.strip()


@pytest.mark.parametrize(
    "name",
    [
        "../settings.yaml",
        "/tmp/settings.yaml",
        "..\\settings.yaml",
        "prompts\\..\\settings.yaml",
        "C:\\temp\\settings.yaml",
        "\\\\server\\share\\settings.yaml",
    ],
)
def test_resource_reader_rejects_path_traversal(name: str) -> None:
    resource_module = importlib.import_module(
        "modular_rag_mcp_server.package_resources"
    )

    with pytest.raises(ValueError, match="资源路径"):
        resource_module.read_resource_text(name)


def test_default_settings_merge_working_directory_local_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.local.yaml").write_text(
        "vector_store:\n  collection_name: local-docs\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    settings_module = importlib.import_module(
        "modular_rag_mcp_server.core.settings"
    )

    settings = settings_module.load_settings()

    assert settings.vector_store.collection_name == "local-docs"
