#!/usr/bin/env python3
"""
评估脚本入口

读取黄金测试集，调用 RetrievalPipeline 执行检索，
通过 EvalRunner 产出 hit_rate / mrr 等指标。

用法:
  python scripts/evaluate.py
  python scripts/evaluate.py --golden-set tests/fixtures/golden_test_set.json
  python scripts/evaluate.py --collection report --top-k 5 --json
  python scripts/evaluate.py --dense-only  # 仅稠密检索（无 FTS5 / RRF），作双路对照基线
  python scripts/evaluate.py --summary-only  # 只打印最终汇总，不打印逐条用例
  python scripts/evaluate.py --e2e  # E2E 模式：L1 检索 + L2 内容评估
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class DenseOnlySearch:
    """
    与 HybridSearch.search 参数兼容，仅执行稠密向量检索（不调用稀疏索引、不做 RRF）。
    供评估脚本与双路召回对照。
    """

    def __init__(self, dense_retriever: Any) -> None:
        self._dense = dense_retriever

    def search(
        self,
        query: Union[str, List[str]],
        top_k: int,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        top_k_final: Optional[int] = None,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[Any]:
        from src.core.trace.trace_context import TraceContext

        k_dense = top_k_dense if top_k_dense is not None else top_k
        k_final = top_k_final if top_k_final is not None else top_k
        if top_k <= 0 or k_dense <= 0 or k_final <= 0:
            raise ValueError(
                f"top_k 必须大于 0，得到: top_k={top_k}, dense={k_dense}, final={k_final}"
            )

        n_retrieve = max(k_dense, k_final, top_k)
        _trace = trace if isinstance(trace, TraceContext) else None
        if _trace:
            with _trace.stage(
                "dense_only_retrieval",
                top_k_retrieve=n_retrieve,
                top_k_final=k_final,
            ):
                results = self._dense.retrieve(
                    query=query,
                    top_k=n_retrieve,
                    filters=filters,
                    trace=trace,
                    collection_name=collection_name,
                )
        else:
            results = self._dense.retrieve(
                query=query,
                top_k=n_retrieve,
                filters=filters,
                trace=trace,
                collection_name=collection_name,
            )

        return results[:k_final]


def _build_pipeline(
    settings: Any,
    collection_override: str | None = None,
    *,
    dense_only: bool = False,
):
    """根据 settings 构建 RetrievalPipeline（可选仅稠密检索）。"""
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.libs.reranker.reranker_factory import RerankerFactory
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.reranker import RerankerOrchestrator
    from src.core.query_engine.retrieval_pipeline import RetrievalPipeline

    embedding = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings)
    reranker_backend = RerankerFactory.create(settings)

    coll = collection_override or settings.vector_store.collection_name

    dense = DenseRetriever(embedding=embedding, vector_store=vector_store)
    if dense_only:
        search_backend: Any = DenseOnlySearch(dense)
    else:
        sqlite_path = getattr(settings.vector_store, "sqlite_path", None)
        sparse = SparseRetriever(
            vector_store=vector_store,
            sqlite_path=sqlite_path,
            collection_name=coll,
        )
        search_backend = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)

    reranker = RerankerOrchestrator(backend=reranker_backend)

    pipeline = RetrievalPipeline(
        query_processor=QueryProcessor(),
        hybrid_search=search_backend,
        reranker=reranker,
        retrieval_config=settings.retrieval,
    )
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于黄金测试集评估 Retrieval 质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/evaluate.py
  python scripts/evaluate.py --golden-set tests/fixtures/golden_test_set.json
  python scripts/evaluate.py --collection report --json
  python scripts/evaluate.py --e2e   # E2E 模式：L1 + L2
  python scripts/evaluate.py --dense-only --json   # 单路稠密基线
  python scripts/evaluate.py --summary-only      # 仅汇总
        """,
    )
    parser.add_argument(
        "--golden-set",
        type=str,
        default=None,
        help="黄金测试集路径（默认使用 settings.yaml 中配置的路径）",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=None,
        help="集合名称覆盖（默认使用测试用例中的 collection 或 settings 中的默认值）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="全局 top_k 覆盖（默认使用每条测试用例自身的 top_k）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出完整报告",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="E2E 模式：执行 retrieve + build_mcp_content，产出 L1 检索 + L2 内容指标",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="仅稠密向量检索（跳过 FTS5 稀疏检索与 RRF），用于与双路召回对比",
    )
    parser.add_argument(
        "--summary-only",
        "-q",
        action="store_true",
        help="仅打印汇总统计（总用例、成功/失败、平均指标等），不输出逐条用例明细",
    )
    args = parser.parse_args()

    # 加载配置
    try:
        from src.core.settings import load_settings

        settings = load_settings(args.config)
    except Exception as e:
        logger.error("配置加载失败: %s", e)
        sys.exit(1)

    golden_set_path = args.golden_set or settings.evaluation.golden_test_set
    if not golden_set_path:
        logger.error("未指定黄金测试集路径，请通过 --golden-set 或 settings.yaml 配置")
        sys.exit(1)

    # 构建 Pipeline
    try:
        pipeline = _build_pipeline(
            settings,
            args.collection,
            dense_only=args.dense_only,
        )
    except Exception as e:
        logger.error("Pipeline 构建失败: %s", e)
        sys.exit(1)

    if args.dense_only:
        logger.info("评估模式: dense-only（无稀疏检索与 RRF）")

    # 构建检索回调 / RAG 回调
    collection_override = args.collection
    top_k_override = args.top_k

    vs_backend = getattr(getattr(settings, "vector_store", None), "backend", "")
    sparse_backend = getattr(getattr(settings, "retrieval", None), "sparse_backend", "bm25")
    use_sqlite_images = vs_backend == "sqlite" and sparse_backend == "fts5"
    sqlite_path = getattr(settings.vector_store, "sqlite_path", None) if use_sqlite_images else None

    def retrieve_func(query: str, top_k: int, collection: str | None) -> list[str]:
        effective_top_k = top_k_override if top_k_override is not None else top_k
        effective_coll = collection_override or collection
        results = pipeline.retrieve(
            query=query,
            top_k=effective_top_k,
            collection_name=effective_coll,
        )
        return [r.id for r in results]

    def rag_func(query: str, top_k: int, collection: str | None):
        """E2E 模式：retrieve + build_mcp_content"""
        effective_top_k = top_k_override if top_k_override is not None else top_k
        effective_coll = collection_override or collection
        results = pipeline.retrieve(
            query=query,
            top_k=effective_top_k,
            collection_name=effective_coll,
        )
        from src.core.response.response_builder import build_mcp_content

        mcp_content = build_mcp_content(
            results,
            collection_name=effective_coll,
            sqlite_path=sqlite_path,
        )
        return results, mcp_content

    # 构建评估器
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory

    try:
        evaluators = EvaluatorFactory.create(settings)
    except (NotImplementedError, ValueError) as e:
        logger.warning("部分评估器不可用 (%s)，回退到 custom 评估器", e)
        evaluators = [EvaluatorFactory.create_custom()]

    # 运行评估
    if args.e2e:
        from src.observability.evaluation.e2e_runner import E2ERunner

        runner = E2ERunner(rag_func=rag_func, l1_evaluators=evaluators)
    else:
        from src.observability.evaluation.eval_runner import EvalRunner

        runner = EvalRunner(retrieve_func=retrieve_func, evaluators=evaluators)

    try:
        report = runner.run(golden_set_path)
    except FileNotFoundError as e:
        logger.error("文件不存在: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("评估执行失败: %s", e)
        sys.exit(1)

    # 输出结果
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_report(report, summary_only=args.summary_only)


def _print_report_summary(report, *, heading: str) -> None:
    """打印汇总：用例统计、总耗时、平均指标。"""
    print()
    print("=" * 60)
    print(f"  {heading}")
    print("=" * 60)
    print()
    print(f"  总用例数:   {report.total_cases}")
    print(f"  成功:       {report.successful_cases}")
    print(f"  失败:       {report.failed_cases}")
    print(f"  总耗时:     {report.total_time_ms:.1f} ms")
    print()

    if report.avg_metrics:
        print("  平均指标:")
        for key, val in sorted(report.avg_metrics.items()):
            print(f"    {key}: {val:.4f}")
        print()
    elif report.successful_cases == 0:
        print("  平均指标:   （无成功用例，未计算）")
        print()


def _print_report(report, *, summary_only: bool = False) -> None:
    """格式化输出评估报告；--summary-only 时只打印汇总。"""
    if summary_only:
        _print_report_summary(report, heading="评估结果汇总")
        print("=" * 60)
        return

    _print_report_summary(report, heading="Evaluation Report")

    print("-" * 60)
    print("  逐条明细:")
    print("-" * 60)

    for i, r in enumerate(report.case_results, 1):
        status = "✅" if r.error is None else "❌"
        print(f"\n  [{i}] {status} {r.query}")
        if r.description:
            print(f"      描述: {r.description}")
        print(f"      期望: {r.golden_chunk_ids}")
        print(f"      实际: {r.retrieved_ids[:5]}{'...' if len(r.retrieved_ids) > 5 else ''}")
        if r.metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in r.metrics.items())
            print(f"      指标: {metrics_str}")
        print(f"      耗时: {r.latency_ms:.1f} ms")
        if r.error:
            print(f"      错误: {r.error}")

    _print_report_summary(report, heading="最终结果（汇总）")
    print("=" * 60)


if __name__ == "__main__":
    main()
