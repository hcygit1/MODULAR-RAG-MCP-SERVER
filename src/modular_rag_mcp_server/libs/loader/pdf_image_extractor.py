"""PyMuPDF image extraction for PDF ingestion."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_images(fitz_module: Any, pdf_path: str, doc_id: str) -> tuple[dict[str, bytes], list[dict[str, Any]], dict]:
    """Extract image bytes and positional metadata from a PDF document."""
    image_data: dict[str, bytes] = {}
    metadata: list[dict[str, Any]] = []
    page_images: dict = {}
    doc = fitz_module.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images[page_num] = []
            for image_index, image_info in enumerate(page.get_images(full=True)):
                try:
                    xref = image_info[0]
                    image = doc.extract_image(xref)
                    image_id = f"{doc_id}_page_{page_num}_img_{image_index}"
                    try:
                        rects = page.get_image_rects(xref)
                        rect = rects[0] if rects else None
                    except Exception:
                        rect = None
                    y_position = rect.y0 if rect else image_index * 1000
                    bbox = (
                        {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1}
                        if rect
                        else None
                    )
                    ext = image.get("ext", "png")
                    item = {
                        "image_id": image_id,
                        "page": page_num,
                        "xref": xref,
                        "y_position": y_position,
                        "width": image.get("width", 0),
                        "height": image.get("height", 0),
                        "mime_type": f"image/{ext}",
                        "ext": ext,
                        "bbox": bbox,
                    }
                    image_bytes = image["image"]
                    image_data[image_id] = image_bytes
                    metadata.append(item)
                    page_images[page_num].append((image_id, y_position, image_bytes, item))
                except Exception as exc:
                    logger.warning(
                        "提取第 %s 页第 %s 张图片失败: %s", page_num, image_index, exc
                    )
            page_images[page_num].sort(key=lambda item: item[1])
    finally:
        doc.close()
    return image_data, metadata, page_images
