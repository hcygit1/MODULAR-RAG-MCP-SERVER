"""父聚合专用检索配置测试。"""

from pathlib import Path
from unittest.mock import MagicMock

import yaml

from src.core.query_engine.query_processor import ProcessedQuery
from src.core.query_engine.retrieval_pipeline import RetrievalPipeline
from src.core.settings import RetrievalConfig, load_settings
from src.libs.vector_store.base_vector_store import QueryResult


def test_loads_parent_candidate_limits_from_yaml(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    config["retrieval"]["top_k_dense_parent"] = 80
    config["retrieval"]["top_k_sparse_parent"] = 60
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True),
        encoding="utf-8",
    )

    settings = load_settings(str(config_path))

    assert settings.retrieval.top_k_dense_parent == 80
    assert settings.retrieval.top_k_sparse_parent == 60


def test_parent_mode_uses_dedicated_dense_and_sparse_limits() -> None:
    query_processor = MagicMock()
    query_processor.process.return_value = ProcessedQuery(
        original_query="question",
        keywords=[],
        filters={},
    )
    hybrid_search = MagicMock()
    child = QueryResult(
        id="child-1",
        score=0.9,
        text="child",
        metadata={"parent_id": "parent-1"},
    )
    hybrid_search.search.return_value = [child]
    reranker = MagicMock()
    reranker.rerank_with_fallback.return_value = ([child], False)
    parent_aggregator = MagicMock()
    parent_aggregator.aggregate.return_value = [
        QueryResult("parent-1", 0.9, "parent", {"parent_id": "parent-1"})
    ]
    config = RetrievalConfig(
        sparse_backend="fts5",
        fusion_algorithm="rrf",
        top_k_dense=30,
        top_k_sparse=20,
        top_k_final=10,
        aggregate_by_parent=True,
        parent_aggregate_top_m=50,
        top_k_dense_parent=90,
        top_k_sparse_parent=70,
    )
    pipeline = RetrievalPipeline(
        query_processor=query_processor,
        hybrid_search=hybrid_search,
        reranker=reranker,
        retrieval_config=config,
        parent_aggregator=parent_aggregator,
        collection_name="docs",
    )

    pipeline.retrieve("question", top_k=5)

    hybrid_search.search.assert_called_once()
    kwargs = hybrid_search.search.call_args.kwargs
    assert kwargs["top_k"] == 50
    assert kwargs["top_k_final"] == 50
    assert kwargs["top_k_dense"] == 90
    assert kwargs["top_k_sparse"] == 70
