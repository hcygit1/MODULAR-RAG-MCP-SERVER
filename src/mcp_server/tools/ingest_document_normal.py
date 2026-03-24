"""
ingest_document_normal Tool

支持本地文件入库：
- PDF：使用 PdfLoader（MarkItDown）解析
- Markdown / TXT：直接读取文本，构造 Document 入库
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.ingestion.models import Document

from mcp.types import CallToolResult

from src.libs.loader.file_integrity import FileIntegrityChecker
from src.mcp_server.tools.config_utils import load_mcp_settings
from src.mcp_server.tools.mcp_utils import dict_to_call_tool_result
from src.mcp_server.tools.error_utils import build_error_response

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
_MD_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_IMAGE_MIME: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _parse_force(value: Any) -> bool:
    """解析 force 参数，兼容 bool/字符串。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _parse_markdown(path: Path) -> "Document":
    """
    加载 Markdown 文件，将本地图片引用转换为 [IMAGE: id] 占位符。

    - 本地可读取的图片：替换为占位符，二进制数据写入 document.metadata["image_data"]
    - 外链或找不到的图片：保持 ![alt](src) 原样，不阻断流程
    """
    from src.ingestion.models import Document

    abs_path = os.path.abspath(str(path))
    doc_id = "doc_" + hashlib.sha256(abs_path.encode()).hexdigest()[:16]
    text = path.read_text(encoding="utf-8")
    md_dir = path.parent

    image_data: Dict[str, bytes] = {}
    images_list: List[Dict[str, Any]] = []
    id_counter: Dict[str, int] = {}

    def _replace(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)

        img_path = (md_dir / src).resolve()
        if not img_path.exists():
            return m.group(0)

        try:
            data = img_path.read_bytes()
        except OSError:
            return m.group(0)

        stem = img_path.stem
        count = id_counter.get(stem, 0)
        id_counter[stem] = count + 1
        image_id = stem if count == 0 else f"{stem}_{count}"

        mime = _IMAGE_MIME.get(img_path.suffix.lower(), "image/png")
        image_data[image_id] = data
        images_list.append({
            "image_id": image_id,
            "alt_text": alt,
            "mime_type": mime,
            "source_path": str(img_path),
        })
        return f"[IMAGE: {image_id}]"

    replaced_text = _MD_IMAGE_RE.sub(_replace, text)

    metadata: Dict[str, Any] = {
        "source_path": abs_path,
        "doc_type": "markdown",
        "title": path.stem,
    }
    if image_data:
        metadata["image_data"] = image_data
        metadata["images"] = images_list

    return Document(id=doc_id, text=replaced_text, metadata=metadata)


def _load_document(file_path: str):
    """按文件类型加载并返回 Document。"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from src.libs.loader.pdf_loader import PdfLoader
        return PdfLoader().load(file_path)

    if suffix == ".md":
        return _parse_markdown(path)

    if suffix == ".txt":
        from src.ingestion.models import Document
        abs_path = os.path.abspath(file_path)
        doc_id = "doc_" + hashlib.sha256(abs_path.encode()).hexdigest()[:16]
        return Document(
            id=doc_id,
            text=path.read_text(encoding="utf-8"),
            metadata={
                "source_path": abs_path,
                "doc_type": "text",
                "title": path.stem,
            },
        )

    raise ValueError(
        f"不支持的文件类型: {suffix}，支持：{', '.join(sorted(_SUPPORTED_SUFFIXES))}"
    )


def execute_ingest_document_normal(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 ingest_document_normal 工具。

    Args:
        arguments: file_path（必填）、collection_name（可选）

    Returns:
        MCP tools/call result
    """
    file_path = arguments.get("file_path")
    if not file_path or not str(file_path).strip():
        return build_error_response(
            "INVALID_PARAMS",
            "参数 file_path 不能为空",
            structured_content_base={"chunk_count": 0},
        )

    file_path = str(file_path).strip()
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return build_error_response(
            "RESOURCE_NOT_FOUND",
            f"文件不存在: {file_path}",
            structured_content_base={"chunk_count": 0},
        )

    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        return build_error_response(
            "INVALID_PARAMS",
            f"不支持的文件类型: {path.suffix}，支持：{', '.join(sorted(_SUPPORTED_SUFFIXES))}",
            structured_content_base={"chunk_count": 0},
        )

    collection_name = arguments.get("collection_name")
    if not collection_name or not str(collection_name).strip():
        settings = load_mcp_settings()
        collection_name = settings.vector_store.collection_name
    else:
        collection_name = str(collection_name).strip()
    force = _parse_force(arguments.get("force", False))

    # 与 CLI 行为对齐：默认开启完整性跳过，force=True 时强制重入库。
    checker = FileIntegrityChecker()
    try:
        file_hash = checker.compute_sha256(file_path)
        should_skip = False if force else checker.should_skip(file_hash)
    except Exception as e:
        return build_error_response(
            "INTERNAL_ERROR",
            f"完整性检查失败: {e}",
            structured_content_base={"chunk_count": 0, "skipped": False},
        )

    if should_skip:
        doc_type = {
            ".pdf": "pdf",
            ".md": "markdown",
            ".txt": "text",
        }.get(path.suffix.lower(), "unknown")
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"文件未变更，已跳过入库（{doc_type}）：{file_path}",
                }
            ],
            "structuredContent": {
                "chunk_count": 0,
                "collection_name": collection_name,
                "doc_type": doc_type,
                "skipped": True,
                "force": force,
            },
            "isError": False,
        }

    try:
        settings = load_mcp_settings()
        from src.ingestion.pipeline import IngestionPipeline

        document = _load_document(file_path)

        from src.core.trace.trace_context import TraceContext
        from src.observability.logger import get_trace_collector

        pipeline = IngestionPipeline(settings)
        trace = TraceContext(operation="ingestion")
        try:
            chunk_count = pipeline.process_document(document, collection_name, trace=trace)
            checker.mark_success(file_hash)
            doc_type = document.metadata.get("doc_type", "unknown")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"解析入库成功（{doc_type}），共写入 {chunk_count} 个 chunks 到集合 {collection_name}",
                    }
                ],
                "structuredContent": {
                    "chunk_count": chunk_count,
                    "collection_name": collection_name,
                    "doc_type": doc_type,
                    "skipped": False,
                    "force": force,
                },
                "isError": False,
            }
        finally:
            get_trace_collector().collect(trace)
            pipeline.close()
    except FileNotFoundError as e:
        return build_error_response(
            "RESOURCE_NOT_FOUND",
            str(e),
            structured_content_base={"chunk_count": 0},
        )
    except ValueError as e:
        return build_error_response(
            "INVALID_PARAMS",
            str(e),
            structured_content_base={"chunk_count": 0},
        )
    except RuntimeError as e:
        logger.exception("ingest_document_normal failed: %s", e)
        return build_error_response(
            "INTERNAL_ERROR",
            f"解析入库失败: {e}",
            structured_content_base={"chunk_count": 0},
        )
    except Exception as e:
        logger.exception("ingest_document_normal failed: %s", e)
        return build_error_response(
            "INTERNAL_ERROR",
            f"解析入库失败: {e}",
            structured_content_base={"chunk_count": 0},
        )


def ingest_document_normal(
    file_path: str,
    collection_name: Optional[str] = None,
    force: bool = False,
) -> CallToolResult:
    """
    解析本地文件并入库。支持 PDF、Markdown（.md）、纯文本（.txt）。

    Args:
        file_path: 本地文件路径（.pdf / .md / .txt）
        collection_name: 目标集合名，不传则使用配置的 vector_store.collection_name
        force: 是否强制重入库（True 时忽略完整性跳过）

    Returns:
        CallToolResult 含 content、structuredContent.chunk_count
    """
    args: Dict[str, Any] = {"file_path": file_path}
    if collection_name and str(collection_name).strip():
        args["collection_name"] = str(collection_name).strip()
    if force:
        args["force"] = True
    return dict_to_call_tool_result(execute_ingest_document_normal(args))
