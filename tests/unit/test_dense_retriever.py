"""
DenseRetriever 单元测试

使用 mock VectorStore 和 Embedding 验证检索逻辑。
"""
from unittest.mock import MagicMock

import pytest

from src.core.query_engine.dense_retriever import DenseRetriever
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.vector_store.base_vector_store import QueryResult
from src.libs.vector_store.fake_vector_store import FakeVectorStore, VectorRecord


class TestDenseRetrieverBasic:
    """基础功能测试"""

    def test_retrieve_returns_results_from_vector_store(self) -> None:
        """VectorStore 返回候选时，DenseRetriever 透传并规范化 score"""
        embedding = FakeEmbedding(dimension=4)
        vector_store = FakeVectorStore()

        # 预置数据
        vector_store.upsert([
            VectorRecord("id1", [1.0, 0.0, 0.0, 0.0], "text1", {}),
            VectorRecord("id2", [0.9, 0.1, 0.0, 0.0], "text2", {}),
        ])

        retriever = DenseRetriever(embedding, vector_store)
        results = retriever.retrieve("query", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, QueryResult) for r in results)
        assert all(0 <= r.score <= 1 for r in results)
        assert all(r.text for r in results)

    def test_retrieve_passes_filters_to_vector_store(self) -> None:
        """filters 正确传递给 VectorStore"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 4]

        mock_store = MagicMock()
        mock_store.query.return_value = [
            QueryResult("id1", 0.8, "text1", {"source": "a.pdf"}),
        ]

        retriever = DenseRetriever(mock_embedding, mock_store)
        retriever.retrieve("query", top_k=5, filters={"source": "a.pdf"})

        mock_store.query.assert_called_once()
        call_kwargs = mock_store.query.call_args[1]
        assert call_kwargs["filters"] == {"source": "a.pdf"}
        assert call_kwargs["top_k"] == 5

    def test_retrieve_normalizes_scores(self) -> None:
        """score 被规范化到 [0, 1]"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.0] * 4]

        # 原始 score 为 0.5, 0.3, 0.1
        mock_store = MagicMock()
        mock_store.query.return_value = [
            QueryResult("id1", 0.5, "t1", {}),
            QueryResult("id2", 0.3, "t2", {}),
            QueryResult("id3", 0.1, "t3", {}),
        ]

        retriever = DenseRetriever(mock_embedding, mock_store, score_normalize=True)
        results = retriever.retrieve("query", top_k=3)

        assert len(results) == 3
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.5)
        assert results[2].score == pytest.approx(0.0)

    def test_retrieve_without_normalize_passes_through_scores(self) -> None:
        """score_normalize=False 时透传原始 score"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.0] * 4]
        mock_store = MagicMock()
        mock_store.query.return_value = [
            QueryResult("id1", 0.87, "t1", {}),
        ]

        retriever = DenseRetriever(mock_embedding, mock_store, score_normalize=False)
        results = retriever.retrieve("query", top_k=1)

        assert len(results) == 1
        assert results[0].score == 0.87


class TestDenseRetrieverEdgeCases:
    """边界情况测试"""

    def test_empty_query_raises(self) -> None:
        """空 query 抛出 ValueError"""
        retriever = DenseRetriever(
            MagicMock(),
            MagicMock(),
        )
        with pytest.raises(ValueError, match="query 不能为空"):
            retriever.retrieve("", top_k=5)
        with pytest.raises(ValueError, match="query 不能为空"):
            retriever.retrieve("   ", top_k=5)

    def test_top_k_zero_raises(self) -> None:
        """top_k <= 0 抛出 ValueError"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 4]
        mock_store = MagicMock()
        mock_store.query.return_value = []

        retriever = DenseRetriever(mock_embedding, mock_store)
        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            retriever.retrieve("q", top_k=0)
        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            retriever.retrieve("q", top_k=-1)

    def test_empty_results_from_store(self) -> None:
        """VectorStore 返回空列表时"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 4]
        mock_store = MagicMock()
        mock_store.query.return_value = []

        retriever = DenseRetriever(mock_embedding, mock_store)
        results = retriever.retrieve("query", top_k=5)

        assert results == []

    def test_query_as_list_converted_to_string(self) -> None:
        """query 为 list 时用空格拼接"""
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 4]
        mock_store = MagicMock()
        mock_store.query.return_value = []

        retriever = DenseRetriever(mock_embedding, mock_store)
        retriever.retrieve(["word1", "word2"], top_k=5)

        mock_embedding.embed.assert_called_once()
        args = mock_embedding.embed.call_args[0][0]
        assert args == ["word1 word2"]


class TestDenseRetrieverIntegration:
    """与 FakeEmbedding + FakeVectorStore 集成测试"""

    def test_roundtrip_with_fake_components(self) -> None:
        """FakeEmbedding + FakeVectorStore 端到端检索"""
        embedding = FakeEmbedding(dimension=16)
        vector_store = FakeVectorStore()

        # 插入测试数据：用 embed 生成向量
        doc_vecs = embedding.embed(["doc about RAG", "doc about MCP", "doc about embedding"])
        vector_store.upsert([
            VectorRecord("c1", doc_vecs[0], "doc about RAG", {}),
            VectorRecord("c2", doc_vecs[1], "doc about MCP", {}),
            VectorRecord("c3", doc_vecs[2], "doc about embedding", {}),
        ])

        retriever = DenseRetriever(embedding, vector_store)
        results = retriever.retrieve("RAG system", top_k=3)

        assert len(results) == 3
        assert all(0 <= r.score <= 1 for r in results)
        ids = {r.id for r in results}
        assert ids == {"c1", "c2", "c3"}
        assert results[0].score >= results[1].score >= results[2].score
