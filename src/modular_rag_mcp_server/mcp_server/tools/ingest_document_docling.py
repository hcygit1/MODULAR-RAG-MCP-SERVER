"""ingest_document_docling Tool."""
from __future__ import annotations

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

_SUPPORTED_SUFFIXES = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".html", ".md"}


def _parse_docling(file_path: str, settings: Any) -> ParsedIngestion:
    from modular_rag_mcp_server.libs.loader.docling_loader import DoclingLoader

    loader = DoclingLoader(chunk_size=settings.ingestion.chunk_size)
    chunks = loader.load_chunks(file_path)
    if not chunks:
        raise IngestionResponseError(
            "INTERNAL_ERROR",
            "Docling 解析后未产生任何 chunks",
        )
    return ParsedIngestion(chunks=chunks)


def _build_skip_response(context: SkipResponseContext) -> Dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"文件未变更，已跳过 Docling 入库：{context.file_path}"
                ),
            }
        ],
        "structuredContent": {
            "chunk_count": 0,
            "collection_name": context.collection_name,
            "parser": "docling",
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
                    f"Docling 解析并入库成功，共写入 {chunk_count} 个 chunks "
                    f"到集合 {collection_name}"
                ),
            }
        ],
        "structuredContent": {
            "chunk_count": chunk_count,
            "collection_name": collection_name,
            "parser": "docling",
            "skipped": False,
            "force": context.force,
        },
        "isError": False,
    }


_DOCLING_SPEC = IngestionToolSpec(
    tool_name="ingest_document_docling",
    supported_suffixes=frozenset(_SUPPORTED_SUFFIXES),
    loader_name_resolver=lambda _path: "docling",
    parser=_parse_docling,
    skip_response_builder=_build_skip_response,
    success_response_builder=_build_success_response,
    runtime_error_prefix="Docling 解析入库失败",
    generic_error_prefix="解析入库失败",
)


def execute_ingest_document_docling(arguments: Dict[str, Any]) -> Dict[str, Any]:
    service = IngestionService(
        settings_loader=load_mcp_settings,
        integrity_checker_factory=FileIntegrityChecker,
        processing_profile_builder=build_processing_profile,
    )
    return service.execute(arguments, _DOCLING_SPEC)


async def ingest_document_docling(
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
    result = await run_sync_callable(execute_ingest_document_docling, args)
    return dict_to_call_tool_result(result)
