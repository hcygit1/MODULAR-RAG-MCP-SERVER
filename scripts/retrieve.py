#!/usr/bin/env python3
"""
检索脚本入口

基于已有 VectorStore 与 BM25 索引执行检索，支持混合检索与 Rerank。
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于已有数据库执行检索（Hybrid + Rerank）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/retrieve.py --query "机器学习是什么" --collection report
  python scripts/retrieve.py -q "RAG 流程" -c report --top-k 5 --json
        """,
    )
    parser.add_argument("--query", "-q", type=str, required=True, help="检索查询字符串")
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=None,
        help="集合名称，需与 ingest 时 --collection 一致（BM25 用）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="返回 Top-K 数量，默认 10",
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
        help="以 JSON 格式输出结果",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        logger.error("top-k 必须大于 0")
        sys.exit(1)

    # 加载配置
    try:
        from src.core.settings import load_settings

        settings = load_settings(args.config)
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        sys.exit(1)

    # 创建组件
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.libs.reranker.reranker_factory import RerankerFactory

    embedding = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings)
    reranker_backend = RerankerFactory.create(settings)

    # 构建 RetrievalPipeline
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.reranker import RerankerOrchestrator
    from src.core.query_engine.retrieval_pipeline import RetrievalPipeline

    dense = DenseRetriever(embedding=embedding, vector_store=vector_store)
    sparse = SparseRetriever(
        base_path=settings.ingestion.bm25_base_path,
        collection_name=args.collection or settings.vector_store.collection_name,
    )
    hybrid = HybridSearch(dense_retriever=dense, sparse_retriever=sparse)
    reranker = RerankerOrchestrator(backend=reranker_backend)

    pipeline = RetrievalPipeline(
        query_processor=QueryProcessor(),
        hybrid_search=hybrid,
        reranker=reranker,
        retrieval_config=settings.retrieval,
    )

    # 执行检索
    try:
        results = pipeline.retrieve(
            query=args.query,
            top_k=args.top_k,
            collection_name=args.collection,
        )
    except FileNotFoundError as e:
        logger.error(f"BM25 索引不存在，请先运行 ingest: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        sys.exit(1)

    # 输出结果
    if args.json:
        out = [
            {
                "id": r.id,
                "score": round(r.score, 4),
                "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                "metadata": {
                    k: v
                    for k, v in (r.metadata or {}).items()
                    if k not in ("sparse_vector", "image_data")
                },
            }
            for r in results
        ]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for i, r in enumerate(results, 1):
            print(f"--- 结果 {i} (score={r.score:.4f}, id={r.id}) ---")
            print(r.text[:500] + ("..." if len(r.text) > 500 else ""))
            print()


if __name__ == "__main__":
    main()
