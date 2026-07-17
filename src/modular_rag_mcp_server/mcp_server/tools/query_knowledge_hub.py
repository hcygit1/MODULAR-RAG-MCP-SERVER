"""
query_knowledge_hub Tool

调用 RetrievalPipeline 执行检索，返回 Markdown + structured citations。
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from mcp.types import CallToolResult

from modular_rag_mcp_server.mcp_server.tools.config_utils import load_mcp_settings
from modular_rag_mcp_server.mcp_server.tools.config_utils import set_config_path as _set_config_path
from modular_rag_mcp_server.mcp_server.tools.mcp_utils import (
    dict_to_call_tool_result,
    run_sync_callable,
)

logger = logging.getLogger(__name__)

# 懒加载的 pipeline 与缓存的 settings（避免重复 load）
_pipeline: Optional[Any] = None
_cached_settings: Optional[Any] = None
_query_worker_state = threading.local()


def _initialize_query_worker() -> None:
    """标记固定 query executor 的 worker，供同步入口避免递归提交。"""
    _query_worker_state.active = True


_query_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="mcp-query",
    initializer=_initialize_query_worker,
)


def _get_pipeline():
    """懒加载并返回 RetrievalPipeline。"""
    global _pipeline, _cached_settings
    if _pipeline is not None:
        return _pipeline
    try:
        from modular_rag_mcp_server.core.query_engine.dense_retriever import DenseRetriever
        from modular_rag_mcp_server.core.query_engine.hybrid_search import HybridSearch
        from modular_rag_mcp_server.core.query_engine.query_processor import QueryProcessor
        from modular_rag_mcp_server.core.query_engine.reranker import RerankerOrchestrator
        from modular_rag_mcp_server.core.query_engine.retrieval_pipeline import RetrievalPipeline
        from modular_rag_mcp_server.core.query_engine.sparse_retriever import SparseRetriever
        from modular_rag_mcp_server.libs.embedding.embedding_factory import EmbeddingFactory
        from modular_rag_mcp_server.libs.reranker.reranker_factory import RerankerFactory
        from modular_rag_mcp_server.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings = load_mcp_settings()
        _cached_settings = settings
        embedding = EmbeddingFactory.create(settings)
        vector_store = VectorStoreFactory.create(settings)
        reranker_backend = RerankerFactory.create(settings)

        dense = DenseRetriever(embedding=embedding, vector_store=vector_store)
        sqlite_path = getattr(settings.vector_store, "sqlite_path", None)
        sparse = SparseRetriever(
            vector_store=vector_store,
            sqlite_path=sqlite_path,
            collection_name=settings.vector_store.collection_name,
        )
        hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)
        reranker = RerankerOrchestrator(backend=reranker_backend)

        # 若配置启用父聚合，构建 ParentAggregator 并注入
        parent_agg = None
        if getattr(settings.retrieval, "aggregate_by_parent", False):
            from modular_rag_mcp_server.core.query_engine.parent_aggregator import ParentAggregator
            parent_agg = ParentAggregator(
                vector_store=vector_store,
                parent_id_key=getattr(settings.retrieval, "parent_id_metadata_key", "parent_id"),
                score_strategy=getattr(settings.retrieval, "parent_score_strategy", "sum"),
            )

        _pipeline = RetrievalPipeline(
            query_processor=QueryProcessor(),
            hybrid_search=hybrid,
            reranker=reranker,
            retrieval_config=settings.retrieval,
            parent_aggregator=parent_agg,
            collection_name=settings.vector_store.collection_name,
        )
        return _pipeline
    except Exception as e:
        logger.exception("Failed to create RetrievalPipeline: %s", e)
        raise


def set_pipeline(pipeline: Any) -> None:
    """测试注入用：替换默认 pipeline。"""
    global _pipeline, _cached_settings
    _pipeline = pipeline
    _cached_settings = None


def set_config_path(path: str) -> None:
    """设置配置文件路径（测试用）。"""
    global _pipeline, _cached_settings
    _set_config_path(path)
    _pipeline = None
    _cached_settings = None


_sqlite_path_override: Optional[str] = None


def set_sqlite_path(path: Optional[str]) -> None:
    """测试注入：覆盖 sqlite_path，None 表示使用配置。"""
    global _sqlite_path_override
    _sqlite_path_override = path


def _execute_query_knowledge_hub_core(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 query_knowledge_hub 工具。

    Args:
        arguments: 包含 query（必填）、collection_name（可选）、top_k（可选，默认 10）

    Returns:
        MCP tools/call result：{ content, structuredContent, isError }
    """
    from modular_rag_mcp_server.core.response.response_builder import build_mcp_content
    from modular_rag_mcp_server.mcp_server.tools.error_utils import build_error_response

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
        from modular_rag_mcp_server.core.trace.trace_context import TraceContext
        from modular_rag_mcp_server.observability.logger import get_trace_collector

        pipeline = _get_pipeline()
        trace = TraceContext(operation="retrieval")
        try:
            results = pipeline.retrieve(
                query=str(query).strip(),
                top_k=top_k,
                collection_name=collection_name,
                trace=trace,
            )
        finally:
            get_trace_collector().collect(trace)

        settings = _cached_settings
        if settings is None:
            settings = load_mcp_settings()
        effective_collection = (
            str(collection_name).strip()
            if collection_name and str(collection_name).strip()
            else settings.vector_store.collection_name
        )
        vs_backend = getattr(getattr(settings, "vector_store", None), "backend", "")
        sparse_backend = getattr(getattr(settings, "retrieval", None), "sparse_backend", "bm25")
        use_sqlite_images = vs_backend == "sqlite" and sparse_backend == "fts5"
        if _sqlite_path_override is not None:
            sqlite_path = _sqlite_path_override
        elif use_sqlite_images:
            sqlite_path = getattr(settings.vector_store, "sqlite_path", None)
        else:
            sqlite_path = None
        return build_mcp_content(
            results,
            collection_name=effective_collection,
            sqlite_path=sqlite_path,
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


def execute_query_knowledge_hub(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """在固定 query worker 上同步执行检索，保持缓存 pipeline 的线程亲和。"""
    if getattr(_query_worker_state, "active", False):
        return _execute_query_knowledge_hub_core(arguments)
    return _query_executor.submit(
        _execute_query_knowledge_hub_core,
        arguments,
    ).result()


async def query_knowledge_hub(
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
    d = await run_sync_callable(
        execute_query_knowledge_hub,
        {
            "query": query,
            "collection_name": collection_name,
            "top_k": top_k,
        },
        executor=_query_executor,
    )
    return dict_to_call_tool_result(d)
