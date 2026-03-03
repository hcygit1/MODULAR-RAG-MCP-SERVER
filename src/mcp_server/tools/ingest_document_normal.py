"""
ingest_document_normal Tool

使用本地 PdfLoader（MarkItDown）解析 PDF 后入库。
适用于简单排版的 PDF，解析快、无需云端。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.types import CallToolResult

from src.mcp_server.tools.config_utils import load_mcp_settings
from src.mcp_server.tools.mcp_utils import dict_to_call_tool_result
from src.mcp_server.tools.error_utils import build_error_response

logger = logging.getLogger(__name__)


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

    collection_name = arguments.get("collection_name")
    if not collection_name or not str(collection_name).strip():
        settings = load_mcp_settings()
        collection_name = settings.vector_store.collection_name
    else:
        collection_name = str(collection_name).strip()

    try:
        settings = load_mcp_settings()
        from src.libs.loader.pdf_loader import PdfLoader
        from src.ingestion.pipeline import IngestionPipeline

        loader = PdfLoader()
        document = loader.load(file_path)

        pipeline = IngestionPipeline(settings)
        try:
            chunk_count = pipeline.process_document(document, collection_name)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"普通解析并入库成功，共写入 {chunk_count} 个 chunks 到集合 {collection_name}",
                    }
                ],
                "structuredContent": {
                    "chunk_count": chunk_count,
                    "collection_name": collection_name,
                },
                "isError": False,
            }
        finally:
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
            f"PDF 解析入库失败: {e}",
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
) -> CallToolResult:
    """
    使用本地 PdfLoader 解析 PDF 并入库。适用于简单排版 PDF，无需云端。

    Args:
        file_path: 本地 PDF 文件路径
        collection_name: 目标集合名，不传则使用配置的 vector_store.collection_name

    Returns:
        CallToolResult 含 content、structuredContent.chunk_count
    """
    args: Dict[str, Any] = {"file_path": file_path}
    if collection_name and str(collection_name).strip():
        args["collection_name"] = str(collection_name).strip()
    return dict_to_call_tool_result(execute_ingest_document_normal(args))
