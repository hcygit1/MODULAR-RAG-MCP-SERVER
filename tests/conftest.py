"""全局 pytest 隔离夹具。"""

from pathlib import Path

import fitz
import pytest

from modular_rag_mcp_server.libs.loader import file_integrity


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> str:
    """Create a tracked-independent PDF fixture for ingestion tests."""
    pdf_path = tmp_path / "sample.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Modular RAG test document. Python retrieval and data science.",
        )
        document.save(pdf_path)
    return str(pdf_path)


@pytest.fixture(autouse=True)
def isolate_file_integrity_database(tmp_path, monkeypatch):
    """避免测试读写运行时的文件完整性历史数据库。"""
    monkeypatch.setattr(
        file_integrity,
        "DEFAULT_DB_PATH",
        tmp_path / "file_integrity.db",
    )
    monkeypatch.setattr(file_integrity, "_default_checker", None)
    yield
    file_integrity._default_checker = None
