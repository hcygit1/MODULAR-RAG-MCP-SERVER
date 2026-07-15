"""全局 pytest 隔离夹具。"""

import pytest

from src.libs.loader import file_integrity


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
