"""
query_knowledge_hub Tool

调用 RetrievalPipeline 执行检索，返回 Markdown + structured citations。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from mcp.types import CallToolResult

from src.mcp_server.tools.mcp_utils import dict_to_call_tool_result

logger = logging.getLogger(__name__)


# 默认配置路径
_DEFAULT_CONFIG_PATH = "config/settings.yaml"

# 懒加载的 pipeline
_pipeline: Optional[Any] = None
_config_path: str = _DEFAULT_CONFIG_PATH


def _get_pipeline():
    """懒加载并返回 RetrievalPipeline。"""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from src.core.query_engine.dense_retriever import DenseRetriever
        from src.core.query_engine.hybrid_search import HybridSearch
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.reranker import RerankerOrchestrator
        from src.core.query_engine.retrieval_pipeline import RetrievalPipeline
        from src.core.query_engine.sparse_retriever import SparseRetriever
        from src.core.settings import load_settings
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.reranker.reranker_factory import RerankerFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = load_settings(_config_path)
        embedding = EmbeddingFactory.create(settings)
        vector_store = VectorStoreFactory.create(settings)
        reranker_backend = RerankerFactory.create(settings)

        dense = DenseRetriever(embedding=embedding, vector_store=vector_store)
        sparse = SparseRetriever(
            base_path=settings.ingestion.bm25_base_path,
            collection_name=settings.vector_store.collection_name,
        )
        hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)
        reranker = RerankerOrchestrator(backend=reranker_backend)

        _pipeline = RetrievalPipeline(
            query_processor=QueryProcessor(),
            hybrid_search=hybrid,
            reranker=reranker,
            retrieval_config=settings.retrieval,
        )
        return _pipeline
    except Exception as e:
        logger.exception("Failed to create RetrievalPipeline: %s", e)
        raise


def set_pipeline(pipeline: Any) -> None:
    """测试注入用：替换默认 pipeline。"""
    global _pipeline
    _pipeline = pipeline


def set_config_path(path: str) -> None:
    """设置配置文件路径（测试用）。"""
    global _config_path, _pipeline
    _config_path = path
    _pipeline = None


_images_base_path_override: Optional[str] = None


def set_images_base_path(path: Optional[str]) -> None:
    """测试注入：覆盖 images_base_path，None 表示使用配置。"""
    global _images_base_path_override
    _images_base_path_override = path


def execute_query_knowledge_hub(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 query_knowledge_hub 工具。

    Args:
        arguments: 包含 query（必填）、collection_name（可选）、top_k（可选，默认 10）

    Returns:
        MCP tools/call result：{ content, structuredContent, isError }
    """
    from src.core.response.response_builder import build_mcp_content
    from src.mcp_server.tools.error_utils import build_error_response

    query = arguments.get("query")
    if not query or not str(query).strip():
        return build_error_response(
            "INVALID_PARAMS",
            "参数 query 不能为空",
            structured_content_base={"citations": []},
        )

    collection_name = arguments.get("collection_name")
    top_k = arguments.get("top_k", 10)
    if not isinstance(top_k, int) or top_k <= 0:
        top_k = 10

    try:
        pipeline = _get_pipeline()
        results = pipeline.retrieve(
            query=str(query).strip(),
            top_k=top_k,
            collection_name=collection_name,
        )
        if _images_base_path_override is not None:
            images_base = _images_base_path_override
        else:
            from src.core.settings import load_settings

            settings = load_settings(_config_path)
            images_base = getattr(
                getattr(settings, "ingestion", None), "images_base_path", "data/images"
            )
        return build_mcp_content(
            results,
            images_base_path=images_base,
            collection_name=collection_name,
        )
    except FileNotFoundError as e:
        return build_error_response(
            "RESOURCE_NOT_FOUND",
            f"BM25 索引不存在，请先运行 ingest: {e}",
            structured_content_base={"citations": []},
        )
    except ValueError as e:
        return build_error_response(
            "INVALID_PARAMS",
            str(e),
            structured_content_base={"citations": []},
        )
    except Exception as e:
        logger.exception("query_knowledge_hub failed: %s", e)
        return build_error_response(
            "INTERNAL_ERROR",
            f"检索失败: {e}",
            structured_content_base={"citations": []},
        )


def query_knowledge_hub(
    query: str,
    collection_name: str | None = None,
    top_k: int = 10,
) -> CallToolResult:
    """
    在知识库中检索与查询相关的文档片段，返回 Markdown 格式的检索结果和结构化引用（source, page, chunk_id, score）。

    Args:
        query: 检索查询字符串
        collection_name: 集合名称，需与 ingest 时一致（可选）
        top_k: 返回 Top-K 数量，默认 10

    Returns:
        CallToolResult 含 content（Markdown）、structuredContent.citations
    """
    d = execute_query_knowledge_hub({
        "query": query,
        "collection_name": collection_name,
        "top_k": top_k,
    })
    return dict_to_call_tool_result(d)
