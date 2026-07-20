"""
ingest_document_mineru Tool

使用 MinerU 云端 API 解析 PDF 后入库。
适用于复杂排版的 PDF（多栏、表格多、公式多、扫描件等），解析精度高。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from mcp.types import CallToolResult

from modular_rag_mcp_server.ingestion.integrity import (
    FileIntegrityChecker,
    build_processing_profile,
)
from modular_rag_mcp_server.mcp_server.services.ingestion_service import (
    IngestionResponseError,
    IngestionService,
    IngestionToolSpec,
    ParsedIngestion,
    SkipResponseContext,
    SuccessResponseContext,
)
from modular_rag_mcp_server.mcp_server.tools.config_utils import load_mcp_settings
from modular_rag_mcp_server.mcp_server.tools.mcp_utils import (
    dict_to_call_tool_result,
    run_sync_callable,
)


def _get_mineru_token(settings: Any) -> str:
    """优先从配置读取，为空时尝试环境变量"""
    token = getattr(getattr(settings, "mineru", None), "api_token", "") or ""
    if not token or not str(token).strip():
        token = os.environ.get("MINERU_API_TOKEN", "")
    return (token or "").strip()


def _parse_mineru(file_path: str, settings: Any) -> ParsedIngestion:
    token = _get_mineru_token(settings)
    if not token:
        raise IngestionResponseError(
            "INVALID_PARAMS",
            "MinerU API Token 未配置，请在外部配置或 "
            "config/settings.local.yaml 的 mineru.api_token 中填写，"
            "或设置环境变量 MINERU_API_TOKEN",
        )

    mineru_config = settings.mineru
    from modular_rag_mcp_server.libs.loader.mineru_cloud_client import MinerUCloudClient

    client = MinerUCloudClient(
        api_token=token,
        model_version=mineru_config.model_version,
        poll_interval_seconds=mineru_config.poll_interval_seconds,
        poll_timeout_seconds=mineru_config.poll_timeout_seconds,
    )
    raw = client.upload_and_parse(file_path)

    if raw.content_list:
        from modular_rag_mcp_server.libs.loader.mineru_block_parser import to_chunks

        chunks = to_chunks(
            raw,
            chunk_size=settings.ingestion.chunk_size,
        )
        return ParsedIngestion(
            chunks=chunks,
            metadata={"parser_mode": "mineru_content_list"},
        )

    from modular_rag_mcp_server.libs.loader.mineru_result_adapter import to_document

    return ParsedIngestion(
        document=to_document(raw),
        metadata={"parser_mode": "mineru_markdown_fallback"},
    )


def _build_skip_response(context: SkipResponseContext) -> Dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": f"文件未变更，已跳过 MinerU 入库：{context.file_path}",
            }
        ],
        "structuredContent": {
            "chunk_count": 0,
            "collection_name": context.collection_name,
            "skipped": True,
            "force": context.force,
        },
        "isError": False,
    }


def _build_success_response(context: SuccessResponseContext) -> Dict[str, Any]:
    chunk_count = context.chunk_count
    collection_name = context.collection_name
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"MinerU 解析并入库成功，共写入 {chunk_count} 个 chunks "
                    f"到集合 {collection_name}"
                ),
            }
        ],
        "structuredContent": {
            "chunk_count": chunk_count,
            "collection_name": collection_name,
            "parser_mode": context.metadata["parser_mode"],
            "skipped": False,
            "force": context.force,
        },
        "isError": False,
    }


_MINERU_SPEC = IngestionToolSpec(
    tool_name="ingest_document_mineru",
    supported_suffixes=frozenset(),
    loader_name_resolver=lambda _path: "mineru",
    parser=_parse_mineru,
    skip_response_builder=_build_skip_response,
    success_response_builder=_build_success_response,
    runtime_error_prefix="MinerU 解析入库失败",
    generic_error_prefix="解析入库失败",
)


def execute_ingest_document_mineru(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """执行 ingest_document_mineru 工具。"""
    service = IngestionService(
        settings_loader=load_mcp_settings,
        integrity_checker_factory=FileIntegrityChecker,
        processing_profile_builder=build_processing_profile,
    )
    return service.execute(arguments, _MINERU_SPEC)


async def ingest_document_mineru(
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
    result = await run_sync_callable(execute_ingest_document_mineru, args)
    return dict_to_call_tool_result(result)
