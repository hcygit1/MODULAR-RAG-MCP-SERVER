#!/usr/bin/env python3
"""
评估脚本入口

读取黄金测试集，调用 RetrievalPipeline 执行检索，
通过 EvalRunner 产出 hit_rate / mrr 等指标。

用法:
  python scripts/evaluate.py
  python scripts/evaluate.py --golden-set tests/fixtures/golden_test_set.json
  python scripts/evaluate.py --collection report --top-k 5 --json
"""
import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _build_pipeline(settings, collection_override: str | None = None):
    """根据 settings 构建 RetrievalPipeline"""
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
    sqlite_path = getattr(settings.vector_store, "sqlite_path", None)
    sparse = SparseRetriever(
        vector_store=vector_store,
        sqlite_path=sqlite_path,
        collection_name=coll,
    )
    hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)
    reranker = RerankerOrchestrator(backend=reranker_backend)

    pipeline = RetrievalPipeline(
        query_processor=QueryProcessor(),
        hybrid_search=hybrid,
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
        pipeline = _build_pipeline(settings, args.collection)
    except Exception as e:
        logger.error("Pipeline 构建失败: %s", e)
        sys.exit(1)

    # 构建检索回调
    collection_override = args.collection
    top_k_override = args.top_k

    def retrieve_func(query: str, top_k: int, collection: str | None) -> list[str]:
        effective_top_k = top_k_override if top_k_override is not None else top_k
        effective_coll = collection_override or collection
        results = pipeline.retrieve(
            query=query,
            top_k=effective_top_k,
            collection_name=effective_coll,
        )
        return [r.id for r in results]

    # 构建评估器
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory

    try:
        evaluators = EvaluatorFactory.create(settings)
    except (NotImplementedError, ValueError) as e:
        logger.warning("部分评估器不可用 (%s)，回退到 custom 评估器", e)
        evaluators = [EvaluatorFactory.create_custom()]

    # 运行评估
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
        _print_report(report)


def _print_report(report) -> None:
    """格式化输出评估报告"""
    print()
    print("=" * 60)
    print("  Evaluation Report")
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

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
