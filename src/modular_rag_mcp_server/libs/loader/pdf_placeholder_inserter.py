"""Insert image placeholders into converted Markdown."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def insert_placeholders(
    fitz_module: Any,
    markdown_text: str,
    page_images: dict,
    pdf_path: str,
    page_to_markdown: dict[int, tuple[int, int]] | None,
    map_pages,
) -> str:
    """Insert placeholders by page position, falling back to document end."""
    if not page_images:
        return markdown_text
    if page_to_markdown is None:
        try:
            page_to_markdown = map_pages(fitz_module, pdf_path, markdown_text)
        except Exception as exc:
            logger.warning("页面映射失败，使用简单策略: %s", exc)
            page_to_markdown = {}
    if not page_to_markdown:
        return append_placeholders(markdown_text, page_images)

    lines = markdown_text.split("\n")
    insertions: list[tuple[int, str]] = []
    doc = fitz_module.open(pdf_path)
    try:
        for page_num in sorted(page_images, reverse=True):
            images = page_images[page_num]
            start_line, end_line = page_to_markdown.get(page_num, (len(lines), len(lines)))
            if page_num not in page_to_markdown:
                insertions.extend((len(lines), f"[IMAGE:{item[0]}]") for item in images)
                continue
            try:
                page_height = doc[page_num].rect.height
            except Exception:
                page_height = 800
            line_count = end_line - start_line
            for image_id, y_position, _, _ in images:
                target = end_line if line_count <= 0 else start_line + int(
                    max(0.0, min(1.0, y_position / page_height if page_height > 0 else 0.5))
                    * line_count
                )
                target = max(start_line, min(target, max(start_line, end_line - 1)))
                insertions.append((target, f"[IMAGE:{image_id}]"))
    finally:
        doc.close()
    for line_index, placeholder in sorted(insertions, reverse=True):
        lines.insert(line_index, placeholder)
    return "\n".join(lines)


def append_placeholders(markdown_text: str, page_images: dict) -> str:
    placeholders = [
        f"[IMAGE:{image_id}]"
        for page_num in sorted(page_images)
        for image_id, _, _, _ in page_images[page_num]
    ]
    return markdown_text + ("\n\n" + "\n".join(placeholders) if placeholders else "")
