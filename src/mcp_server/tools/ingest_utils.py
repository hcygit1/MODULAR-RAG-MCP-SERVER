"""Shared helpers for ingestion MCP tools."""
from __future__ import annotations

from typing import Any


def parse_force(value: Any) -> bool:
    """Parse the force flag from bool or common string values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False
