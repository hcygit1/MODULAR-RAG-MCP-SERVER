"""
ingest_document_mineru Tool

使用 MinerU 云端 API 解析 PDF 后入库。
适用于复杂排版的 PDF（多栏、表格多、公式多、扫描件等），解析精度高。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.types import CallToolResult

from src.libs.loader.file_integrity import FileIntegrityChecker
from src.mcp_server.tools.config_utils import load_mcp_settings
from src.mcp_server.tools.mcp_utils import dict_to_call_tool_result
from src.mcp_server.tools.error_utils import build_error_response

logger = logging.getLogger(__name__)


def _parse_force(value: Any) -> bool:
    """解析 force 参数，兼容 bool/字符串。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _get_mineru_token(settings: Any) -> str:
    """优先从配置读取，为空时尝试环境变量"""
    token = getattr(getattr(settings, "mineru", None), "api_token", "") or ""
    if not token or not str(token).strip():
        token = os.environ.get("MINERU_API_TOKEN", "")
    return (token or "").strip()


def execute_ingest_document_mineru(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 ingest_document_mineru 工具。

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
    force = _parse_force(arguments.get("force", False))

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
            "content": [
                {
                    "type": "text",
                    "text": f"文件未变更，已跳过 MinerU 入库：{file_path}",
                }
            ],
            "structuredContent": {
                "chunk_count": 0,
                "collection_name": collection_name,
                "skipped": True,
                "force": force,
            },
            "isError": False,
        }

    try:
        settings = load_mcp_settings()
        token = _get_mineru_token(settings)
        if not token:
            return build_error_response(
                "INVALID_PARAMS",
                "MinerU API Token 未配置，请在 config/settings.yaml 的 mineru.api_token 中填写，或设置环境变量 MINERU_API_TOKEN",
                structured_content_base={"chunk_count": 0},
            )

        mineru_config = settings.mineru
        from src.libs.loader.mineru_cloud_client import MinerUCloudClient
        from src.libs.loader.mineru_result_adapter import to_document
        from src.ingestion.pipeline import IngestionPipeline

        client = MinerUCloudClient(
            api_token=token,
            model_version=mineru_config.model_version,
            poll_interval_seconds=mineru_config.poll_interval_seconds,
            poll_timeout_seconds=mineru_config.poll_timeout_seconds,
        )

        raw = client.upload_and_parse(file_path)
        document = to_document(raw)

        from src.core.trace.trace_context import TraceContext
        from src.observability.logger import get_trace_collector

        pipeline = IngestionPipeline(settings)
        trace = TraceContext(operation="ingestion")
        try:
            chunk_count = pipeline.process_document(document, collection_name, trace=trace)
            checker.mark_success(file_hash)
            return {
                "content": [{"type": "text", "text": f"MinerU 解析并入库成功，共写入 {chunk_count} 个 chunks 到集合 {collection_name}"}],
                "structuredContent": {
                    "chunk_count": chunk_count,
                    "collection_name": collection_name,
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
        logger.exception("ingest_document_mineru failed: %s", e)
        return build_error_response(
            "INTERNAL_ERROR",
            f"MinerU 解析入库失败: {e}",
            structured_content_base={"chunk_count": 0},
        )
    except Exception as e:
        logger.exception("ingest_document_mineru failed: %s", e)
        return build_error_response(
            "INTERNAL_ERROR",
            f"解析入库失败: {e}",
            structured_content_base={"chunk_count": 0},
        )


def ingest_document_mineru(
    file_path: str,
    collection_name: Optional[str] = None,
    force: bool = False,
) -> CallToolResult:
    """
    使用 MinerU 云端解析 PDF 并入库。适用于复杂排版 PDF（表格多、公式多、扫描件等）。

    Args:
        file_path: 本地 PDF 文件路径
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
    return dict_to_call_tool_result(execute_ingest_document_mineru(args))
