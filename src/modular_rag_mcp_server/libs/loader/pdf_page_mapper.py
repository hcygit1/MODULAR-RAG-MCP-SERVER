"""Map PDF pages to ranges in converted Markdown."""

from __future__ import annotations

import re
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for tolerant fingerprint matching."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()


def map_pages(
    fitz_module: Any,
    pdf_path: str,
    markdown_text: str,
    fingerprint_length: int = 60,
) -> dict[int, tuple[int, int]]:
    """Return Markdown line ranges keyed by zero-based PDF page number."""
    page_to_markdown: dict[int, tuple[int, int]] = {}
    lines = markdown_text.split("\n")
    doc = fitz_module.open(pdf_path)
    try:
        fingerprints = {
            page_num: normalize_text(page.get_text()[:fingerprint_length].strip())
            for page_num, page in enumerate(doc)
            if page.get_text() and len(normalize_text(page.get_text()[:fingerprint_length])) >= 10
        }
        for page_num, fingerprint in fingerprints.items():
            search_start = page_to_markdown.get(page_num - 1, (0, 0))[1]
            start_line = _find_fingerprint(lines, fingerprint, search_start)
            if start_line is None:
                continue
            end_line = len(lines)
            for next_page in range(page_num + 1, len(doc)):
                next_fingerprint = fingerprints.get(next_page)
                if not next_fingerprint:
                    continue
                next_line = _find_fingerprint(lines, next_fingerprint, start_line + 1)
                if next_line is not None:
                    end_line = next_line
                    break
            page_to_markdown[page_num] = (start_line, end_line)
    finally:
        doc.close()
    return page_to_markdown


def _find_fingerprint(lines: list[str], fingerprint: str, start: int) -> int | None:
    for line_index in range(start, len(lines)):
        window_end = min(line_index + 40, len(lines))
        window = normalize_text("\n".join(lines[line_index:window_end]))
        if fingerprint in window:
            exact_start = line_index
            for inner in range(line_index, window_end):
                inner_window = normalize_text("\n".join(lines[inner:window_end]))
                if fingerprint in inner_window:
                    exact_start = inner
                else:
                    break
            return exact_start
    return None
