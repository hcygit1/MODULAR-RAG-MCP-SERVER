"""Read immutable resources bundled with the installed package."""
from __future__ import annotations

from importlib import resources
from pathlib import PurePosixPath, PureWindowsPath

_RESOURCE_PACKAGE = "modular_rag_mcp_server.resources"


def _resource_parts(name: str) -> tuple[str, ...]:
    raw_parts = name.split("/")
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(f"非法资源路径: {name}")
    return path.parts


def read_resource_text(name: str) -> str:
    """Return one UTF-8 package resource without exposing filesystem paths."""
    resource = resources.files(_RESOURCE_PACKAGE)
    for part in _resource_parts(name):
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


def read_prompt(name: str) -> str:
    """Return a bundled prompt by filename."""
    if PurePosixPath(name).name != name:
        raise ValueError(f"非法资源路径: {name}")
    return read_resource_text(f"prompts/{name}")
