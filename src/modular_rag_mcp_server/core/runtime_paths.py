"""Resolve writable runtime paths without changing the legacy default."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_DIR_ENV = "MODULAR_RAG_DATA_DIR"
_data_dir_override: Optional[Path] = None


@dataclass(frozen=True)
class RuntimePaths:
    """Writable paths shared by CLI, ingestion and observability."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "db" / "rag.sqlite"

    @property
    def log_path(self) -> Path:
        return self.root / "logs" / "traces.jsonl"

    @property
    def integrity_db_path(self) -> Path:
        return self.root / "cache" / "processing" / "file_integrity.db"


def set_data_dir(path: Optional[str]) -> None:
    """Set the process-local CLI override; ``None`` restores env/default lookup."""
    global _data_dir_override
    _data_dir_override = Path(path).expanduser() if path else None


def get_runtime_paths() -> RuntimePaths:
    """Return runtime paths using CLI override, environment, then current directory."""
    root = _data_dir_override
    if root is None:
        env_root = os.environ.get(DATA_DIR_ENV)
        root = Path(env_root).expanduser() if env_root else Path.cwd()
    return RuntimePaths(root=root)


def resolve_runtime_path(value: str, category: str) -> str:
    """Resolve known ``data``/``logs``/``cache`` relative paths under the runtime root."""
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)

    root = get_runtime_paths().root
    if root == Path.cwd():
        return value

    parts = path.parts
    if parts and parts[0] == category:
        return str(root.joinpath(*parts))
    return value
