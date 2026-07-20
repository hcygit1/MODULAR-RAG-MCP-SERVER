"""Tests for packaged operational CLI commands."""

from pathlib import Path

import pytest

from modular_rag_mcp_server.cli_commands.evaluate import _resolve_golden_set
from modular_rag_mcp_server.cli_commands.ingest import collect_pdf_files


def test_collect_pdf_files_returns_nested_files_once(tmp_path: Path) -> None:
    top_level = tmp_path / "top.pdf"
    nested = tmp_path / "nested" / "child.pdf"
    nested.parent.mkdir()
    top_level.write_bytes(b"pdf")
    nested.write_bytes(b"pdf")

    assert collect_pdf_files(tmp_path) == sorted([top_level, nested])


def test_collect_pdf_files_rejects_non_pdf_file(tmp_path: Path) -> None:
    text_file = tmp_path / "document.txt"
    text_file.write_text("text", encoding="utf-8")

    with pytest.raises(ValueError, match="不支持的文件类型"):
        collect_pdf_files(text_file)


def test_resolve_golden_set_falls_back_to_packaged_fixture(tmp_path: Path) -> None:
    missing_source_path = tmp_path / "tests" / "fixtures" / "golden_test_set.json"
    with _resolve_golden_set(str(missing_source_path)) as path:
        assert path.is_file()
        assert path.name == "golden_test_set.json"


def test_pipeline_factory_preserves_falsey_integrity_checker() -> None:
    from modular_rag_mcp_server.ingestion.pipeline_components import (
        resolve_integrity_checker,
    )

    class FalseyChecker:
        def __bool__(self) -> bool:
            return False

    checker = FalseyChecker()
    assert resolve_integrity_checker(checker) is checker
