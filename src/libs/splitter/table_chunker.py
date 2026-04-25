"""Table-aware chunking utilities."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence

from src.ingestion.models import Chunk
from src.ingestion.parsed_document import ParsedBlock


class _HTMLTableParser(HTMLParser):
    """Best-effort HTML table parser for MinerU table_body output."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._current_row: Optional[List[str]] = None
        self._current_cell: Optional[List[str]] = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: Sequence[tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
            self._in_cell = True

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            stripped = data.strip()
            if stripped:
                self._current_cell.append(stripped)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_cell is not None:
            cell = " ".join(self._current_cell).strip()
            self._current_row = self._current_row or []
            self._current_row.append(cell)
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", text).strip()


def _parse_html_rows(html: str) -> List[List[str]]:
    parser = _HTMLTableParser()
    parser.feed(html or "")
    return parser.rows


def _render_markdown(headers: List[str], rows: List[List[str]]) -> str:
    if not headers and rows:
        headers = [f"col_{idx + 1}" for idx in range(max(len(row) for row in rows))]
    width = len(headers)

    def normalize(row: List[str]) -> List[str]:
        padded = list(row[:width])
        if len(padded) < width:
            padded.extend([""] * (width - len(padded)))
        return [cell.replace("\n", " ").strip() for cell in padded]

    lines = [
        "| " + " | ".join(normalize(headers)) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(normalize(row)) + " |")
    return "\n".join(lines)


class TableChunker:
    """Split table blocks while preserving headers and raw table metadata."""

    def __init__(self, chunk_size: int = 512, max_rows_per_chunk: int = 30) -> None:
        self._chunk_size = max(128, chunk_size)
        self._max_rows_per_chunk = max(1, max_rows_per_chunk)

    def chunk_table(
        self,
        block: ParsedBlock,
        *,
        doc_id: str,
        chunk_index_start: int,
        base_metadata: Dict[str, Any],
        heading_path: List[str],
    ) -> List[Chunk]:
        html = block.html or block.metadata.get("raw_table_html") or ""
        rows = _parse_html_rows(html)
        if not rows:
            return [
                self._make_chunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index_start,
                    text=self._render_fallback_table(block, heading_path),
                    metadata=base_metadata,
                    block=block,
                    row_range=None,
                    headers=[],
                )
            ]

        headers = rows[0]
        body_rows = rows[1:] or rows
        whole_table = self._render_table_text(block, heading_path, headers, body_rows, (1, len(body_rows)))
        if len(whole_table) <= self._chunk_size * 2 and len(body_rows) <= self._max_rows_per_chunk:
            return [
                self._make_chunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index_start,
                    text=whole_table,
                    metadata=base_metadata,
                    block=block,
                    row_range=(1, len(body_rows)),
                    headers=headers,
                )
            ]

        chunks: List[Chunk] = []
        for offset in range(0, len(body_rows), self._max_rows_per_chunk):
            group = body_rows[offset : offset + self._max_rows_per_chunk]
            row_range = (offset + 1, offset + len(group))
            text = self._render_table_text(block, heading_path, headers, group, row_range)
            chunks.append(
                self._make_chunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index_start + len(chunks),
                    text=text,
                    metadata=base_metadata,
                    block=block,
                    row_range=row_range,
                    headers=headers,
                )
            )
        return chunks

    def _render_fallback_table(self, block: ParsedBlock, heading_path: List[str]) -> str:
        parts = self._context_parts(block, heading_path)
        fallback = block.text or _strip_html(block.html or "")
        if fallback:
            parts.append(fallback)
        return "\n\n".join(parts).strip()

    def _render_table_text(
        self,
        block: ParsedBlock,
        heading_path: List[str],
        headers: List[str],
        rows: List[List[str]],
        row_range: tuple[int, int],
    ) -> str:
        parts = self._context_parts(block, heading_path)
        parts.append(f"行范围: {row_range[0]}-{row_range[1]}")
        parts.append(_render_markdown(headers, rows))
        return "\n\n".join(part for part in parts if part).strip()

    def _context_parts(self, block: ParsedBlock, heading_path: List[str]) -> List[str]:
        parts: List[str] = []
        if heading_path:
            parts.append("章节: " + " > ".join(heading_path))
        caption = block.metadata.get("table_caption") or block.metadata.get("caption")
        if caption:
            if isinstance(caption, list):
                caption = " ".join(str(item) for item in caption if item)
            parts.append(f"表格说明: {caption}")
        footnote = block.metadata.get("table_footnote")
        if footnote:
            if isinstance(footnote, list):
                footnote = " ".join(str(item) for item in footnote if item)
            parts.append(f"表格注释: {footnote}")
        return parts

    def _make_chunk(
        self,
        *,
        doc_id: str,
        chunk_index: int,
        text: str,
        metadata: Dict[str, Any],
        block: ParsedBlock,
        row_range: Optional[tuple[int, int]],
        headers: List[str],
    ) -> Chunk:
        chunk_metadata = dict(metadata)
        chunk_metadata.update(block.metadata)
        chunk_metadata.update(
            {
                "chunk_index": chunk_index,
                "source_doc_id": doc_id,
                "block_id": block.id,
                "block_type": "table",
                "page_idx": block.page_idx,
                "bbox": block.bbox,
                "headers": headers,
            }
        )
        if block.html:
            chunk_metadata["raw_table_html"] = block.html
        if row_range:
            chunk_metadata["row_range"] = [row_range[0], row_range[1]]
        if block.image_refs:
            chunk_metadata["image_refs"] = block.image_refs
        return Chunk(
            id=f"{doc_id}_chunk_{chunk_index}",
            text=text,
            metadata={k: v for k, v in chunk_metadata.items() if v is not None},
        )
