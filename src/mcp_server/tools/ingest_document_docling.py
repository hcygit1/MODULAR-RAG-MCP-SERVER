"""ingest_document_docling Tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.types import CallToolResult

from src.libs.loader.file_integrity import FileIntegrityChecker
from src.mcp_server.tools.config_utils import load_mcp_settings
from src.mcp_server.tools.error_utils import build_error_response
from src.mcp_server.tools.ingest_utils import parse_force
from src.mcp_server.tools.mcp_utils import dict_to_call_tool_result

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".html", ".md"}

def execute_ingest_document_docling(arguments: Dict[str, Any]) -> Dict[str, Any]:
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
    settings = load_mcp_settings()
    if not collection_name or not str(collection_name).strip():
        collection_name = settings.vector_store.collection_name
    else:
        collection_name = str(collection_name).strip()
    force = parse_force(arguments.get("force", False))

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
        return {
            "content": [{"type": "text", "text": f"文件未变更，已跳过 Docling 入库：{file_path}"}],
            "structuredContent": {
                "chunk_count": 0,
                "collection_name": collection_name,
                "parser": "docling",
                "skipped": True,
                "force": force,
            },
            "isError": False,
        }

    try:
        from src.core.trace.trace_context import TraceContext
        from src.ingestion.pipeline import IngestionPipeline
        from src.libs.loader.docling_loader import DoclingLoader
        from src.observability.logger import get_trace_collector

        loader = DoclingLoader(chunk_size=settings.ingestion.chunk_size)
        chunks = loader.load_chunks(file_path)
        if not chunks:
            return build_error_response(
                "INTERNAL_ERROR",
                "Docling 解析后未产生任何 chunks",
                structured_content_base={"chunk_count": 0},
            )

        pipeline = IngestionPipeline(settings)
        trace = TraceContext(operation="ingestion")
        try:
            chunk_count = pipeline.process_chunks(chunks, collection_name, trace=trace)
            checker.mark_success(file_hash)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Docling 解析并入库成功，共写入 {chunk_count} 个 chunks 到集合 {collection_name}",
                    }
                ],
                "structuredContent": {
                    "chunk_count": chunk_count,
                    "collection_name": collection_name,
                    "parser": "docling",
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
        logger.exception("ingest_document_docling failed: %s", e)
        return build_error_response(
            "INTERNAL_ERROR",
            f"Docling 解析入库失败: {e}",
            structured_content_base={"chunk_count": 0},
        )
    except Exception as e:
        logger.exception("ingest_document_docling failed: %s", e)
        return build_error_response(
            "INTERNAL_ERROR",
            f"解析入库失败: {e}",
            structured_content_base={"chunk_count": 0},
        )


def ingest_document_docling(
    file_path: str,
    collection_name: Optional[str] = None,
    force: bool = False,
) -> CallToolResult:
    """使用 Docling 结构化解析本地文件并入库。"""
    args: Dict[str, Any] = {"file_path": file_path}
    if collection_name and str(collection_name).strip():
        args["collection_name"] = str(collection_name).strip()
    if force:
        args["force"] = True
    return dict_to_call_tool_result(execute_ingest_document_docling(args))
