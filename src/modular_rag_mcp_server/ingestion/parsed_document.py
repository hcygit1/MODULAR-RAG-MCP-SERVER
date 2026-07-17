"""Structured parser output shared by Docling and MinerU adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedBlock:
    """A layout-aware document block before ingestion chunking."""

    id: str
    type: str
    text: str = ""
    html: Optional[str] = None
    image_refs: List[str] = field(default_factory=list)
    page_idx: Optional[int] = None
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """A document represented as ordered blocks plus shared assets."""

    id: str
    blocks: List[ParsedBlock]
    metadata: Dict[str, Any] = field(default_factory=dict)
