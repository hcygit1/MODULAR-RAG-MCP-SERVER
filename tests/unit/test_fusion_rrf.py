"""
RRF Fusion 单元测试

验证 RRF 融合逻辑：deterministic 输出、k 参数可配置、边界情况。
"""
import pytest

from src.core.query_engine.fusion import fuse_rrf
from src.libs.vector_store.base_vector_store import QueryResult


class TestFusionRRFBasic:
    """基础功能测试"""

    def test_deterministic_output(self) -> None:
        """固定输入输出 deterministic"""
        dense = [
            QueryResult("a", 0.9, "text_a", {}),
            QueryResult("b", 0.8, "text_b", {}),
        ]
        sparse = [
            QueryResult("b", 0.7, "text_b", {}),
            QueryResult("a", 0.6, "text_a", {}),
        ]
        r1 = fuse_rrf(dense, sparse, k=60)
        r2 = fuse_rrf(dense, sparse, k=60)
        assert [x.id for x in r1] == [x.id for x in r2]
        assert all(
            r1[i].score == r2[i].score for i in range(len(r1))
        )

    def test_k_configurable(self) -> None:
        """k 参数可配置"""
        dense = [
            QueryResult("a", 0.9, "ta", {}),
            QueryResult("b", 0.8, "tb", {}),
        ]
        sparse = [
            QueryResult("b", 0.7, "tb", {}),
            QueryResult("a", 0.6, "ta", {}),
        ]
        r_small_k = fuse_rrf(dense, sparse, k=1)
        r_large_k = fuse_rrf(dense, sparse, k=100)
        assert len(r_small_k) == len(r_large_k) == 2

    def test_rrf_scores_from_both_ranks(self) -> None:
        """RRF 分数 = 两路排名贡献之和"""
        dense = [
            QueryResult("a", 0.9, "ta", {}),
            QueryResult("b", 0.8, "tb", {}),
        ]
        sparse = [
            QueryResult("b", 0.7, "tb", {}),
            QueryResult("a", 0.6, "ta", {}),
        ]
        result = fuse_rrf(dense, sparse, k=60)
        # a: dense rank 1, sparse rank 2 -> 1/61 + 1/62
        # b: dense rank 2, sparse rank 1 -> 1/62 + 1/61
        # a 和 b 分数相同，按 id 字典序 a < b，故 a 在前
        assert result[0].id == "a"
        assert result[1].id == "b"
        expected_score = 1.0 / 61 + 1.0 / 62
        assert result[0].score == pytest.approx(expected_score)
        assert result[1].score == pytest.approx(expected_score)

    def test_overlap_ranks_higher_than_single(self) -> None:
        """同时在两路出现的文档 RRF 分数高于仅在一路出现的"""
        dense = [
            QueryResult("both", 0.9, "both_text", {}),
            QueryResult("dense_only", 0.8, "dense_text", {}),
        ]
        sparse = [
            QueryResult("both", 0.7, "both_text", {}),
            QueryResult("sparse_only", 0.6, "sparse_text", {}),
        ]
        result = fuse_rrf(dense, sparse, k=60)
        ids = [r.id for r in result]
        assert ids[0] == "both"  # 两路都有，分数最高
        assert "dense_only" in ids and "sparse_only" in ids


class TestFusionRRFEdgeCases:
    """边界情况测试"""

    def test_empty_dense_uses_sparse_only(self) -> None:
        """dense 为空时只用 sparse"""
        sparse = [
            QueryResult("x", 0.5, "tx", {}),
        ]
        result = fuse_rrf([], sparse, k=60)
        assert len(result) == 1
        assert result[0].id == "x"
        assert result[0].score == pytest.approx(1.0 / 61)
        assert result[0].text == "tx"

    def test_empty_sparse_uses_dense_only(self) -> None:
        """sparse 为空时只用 dense"""
        dense = [
            QueryResult("y", 0.9, "ty", {}),
        ]
        result = fuse_rrf(dense, [], k=60)
        assert len(result) == 1
        assert result[0].id == "y"
        assert result[0].score == pytest.approx(1.0 / 61)
        assert result[0].text == "ty"

    def test_both_empty_returns_empty(self) -> None:
        """两路都空返回空列表"""
        assert fuse_rrf([], [], k=60) == []

    def test_tie_break_by_id(self) -> None:
        """同分时按 id 字典序"""
        dense = [
            QueryResult("z", 0.9, "tz", {}),
            QueryResult("a", 0.8, "ta", {}),
        ]
        sparse = [
            QueryResult("a", 0.9, "ta", {}),
            QueryResult("z", 0.8, "tz", {}),
        ]
        result = fuse_rrf(dense, sparse, k=60)
        # 两文档分数相同，按 id: a < z
        assert result[0].id == "a"
        assert result[1].id == "z"

    def test_preserves_text_and_metadata(self) -> None:
        """保留 text 和 metadata"""
        dense = [
            QueryResult("id1", 0.9, "hello", {"source": "a.pdf"}),
        ]
        sparse = [
            QueryResult("id2", 0.8, "world", {"source": "b.pdf"}),
        ]
        result = fuse_rrf(dense, sparse, k=60)
        by_id = {r.id: r for r in result}
        assert by_id["id1"].text == "hello"
        assert by_id["id1"].metadata == {"source": "a.pdf"}
        assert by_id["id2"].text == "world"
        assert by_id["id2"].metadata == {"source": "b.pdf"}
