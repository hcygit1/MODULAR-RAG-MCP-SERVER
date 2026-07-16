"""
ingest_document_normal Tool

支持本地文件入库：
- PDF：使用 PdfLoader（MarkItDown）解析
- Markdown / TXT：直接读取文本，构造 Document 入库
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.ingestion.models import Document

from mcp.types import CallToolResult

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    build_processing_profile,
)
from src.mcp_server.services.ingestion_service import (
    IngestionService,
    IngestionToolSpec,
    ParsedIngestion,
    SkipResponseContext,
    SuccessResponseContext,
)
from src.mcp_server.tools.config_utils import load_mcp_settings
from src.mcp_server.tools.mcp_utils import dict_to_call_tool_result, run_sync_callable

_SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
_LOADER_NAME_BY_SUFFIX = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".txt": "text",
}
_MD_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_IMAGE_MIME: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

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

    def _replace(m: re.Match[str]) -> str:
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
        image_id = f"{doc_id}_image_{stem}_{count}"

        mime = _IMAGE_MIME.get(img_path.suffix.lower(), "image/png")
        image_data[image_id] = data
        images_list.append({
            "image_id": image_id,
            "alt_text": alt,
            "mime_type": mime,
            "source_path": str(img_path),
        })
        return f"[IMAGE:{image_id}]"

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


def _parse_normal(file_path: str, _settings: Any) -> ParsedIngestion:
    document = _load_document(file_path)
    return ParsedIngestion(
        document=document,
        metadata={"doc_type": document.metadata.get("doc_type", "unknown")},
    )


def _build_skip_response(context: SkipResponseContext) -> Dict[str, Any]:
    file_path = context.file_path
    doc_type = {
        ".pdf": "pdf",
        ".md": "markdown",
        ".txt": "text",
    }.get(Path(file_path).suffix.lower(), "unknown")
    return {
        "content": [
            {
                "type": "text",
                "text": f"文件未变更，已跳过入库（{doc_type}）：{file_path}",
            }
        ],
        "structuredContent": {
            "chunk_count": 0,
            "collection_name": context.collection_name,
            "doc_type": doc_type,
            "skipped": True,
            "force": context.force,
        },
        "isError": False,
    }


def _build_success_response(context: SuccessResponseContext) -> Dict[str, Any]:
    doc_type = context.metadata["doc_type"]
    chunk_count = context.chunk_count
    collection_name = context.collection_name
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"解析入库成功（{doc_type}），共写入 {chunk_count} 个 chunks "
                    f"到集合 {collection_name}"
                ),
            }
        ],
        "structuredContent": {
            "chunk_count": chunk_count,
            "collection_name": collection_name,
            "doc_type": doc_type,
            "skipped": False,
            "force": context.force,
        },
        "isError": False,
    }


_NORMAL_SPEC = IngestionToolSpec(
    tool_name="ingest_document_normal",
    supported_suffixes=frozenset(_SUPPORTED_SUFFIXES),
    loader_name_resolver=lambda path: _LOADER_NAME_BY_SUFFIX[path.suffix.lower()],
    parser=_parse_normal,
    skip_response_builder=_build_skip_response,
    success_response_builder=_build_success_response,
    runtime_error_prefix="解析入库失败",
    generic_error_prefix="解析入库失败",
)


def execute_ingest_document_normal(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 ingest_document_normal 工具。

    Args:
        arguments: file_path（必填）、collection_name（可选）

    Returns:
        MCP tools/call result
    """
    service = IngestionService(
        settings_loader=load_mcp_settings,
        integrity_checker_factory=FileIntegrityChecker,
        processing_profile_builder=build_processing_profile,
    )
    return service.execute(arguments, _NORMAL_SPEC)


async def ingest_document_normal(
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
    result = await run_sync_callable(execute_ingest_document_normal, args)
    return dict_to_call_tool_result(result)
