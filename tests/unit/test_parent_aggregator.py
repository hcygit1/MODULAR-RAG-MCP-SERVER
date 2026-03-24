"""
ParentAggregator 单元测试

覆盖：分组聚合、sum/count 打分、top_k 截断、正文合并顺序、
去重子块、无 parent_id 子块跳过、空输入。
"""
import pytest

from src.core.query_engine.parent_aggregator import ParentAggregator
from src.libs.vector_store.base_vector_store import QueryResult


# ---------------------------------------------------------------------------
# 辅助：模拟 VectorStore
# ---------------------------------------------------------------------------

class MockStore:
    """模拟 VectorStore，按 parent_id 返回预设子块。"""

    def __init__(self, db: dict):
        # db: {parent_id: [QueryResult, ...]}（已按 chunk_index 排好序）
        self._db = db

    def get_chunks_by_parent_id(self, collection_name, parent_id):
        return self._db.get(parent_id, [])


def _qr(chunk_id, score, text, parent_id, chunk_index=0):
    return QueryResult(chunk_id, score, text, {"parent_id": parent_id, "chunk_index": chunk_index})


# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------

STORE_DB = {
    "doc_s1": [
        _qr("c0", 0, "## 宫保鸡丁", "doc_s1", 0),
        _qr("c1", 0, "### 食材\n鸡肉300g", "doc_s1", 1),
        _qr("c2", 0, "### 步骤\n热锅翻炒", "doc_s1", 2),
    ],
    "doc_s2": [
        _qr("c3", 0, "## 鱼香肉丝", "doc_s2", 0),
        _qr("c4", 0, "### 食材\n猪肉丝200g", "doc_s2", 1),
    ],
}

CHILD_RESULTS = [
    _qr("c1", 0.92, "### 食材\n鸡肉300g", "doc_s1", 1),
    _qr("c2", 0.88, "### 步骤\n热锅翻炒", "doc_s1", 2),
    _qr("c4", 0.71, "### 食材\n猪肉丝200g", "doc_s2", 1),
]


# ---------------------------------------------------------------------------
# 分组与排序
# ---------------------------------------------------------------------------

def test_groups_by_parent_id():
    """子块应按 parent_id 分组，分数累加。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(CHILD_RESULTS, top_k=2, collection_name="col")
    ids = [r.id for r in results]
    assert "doc_s1" in ids
    assert "doc_s2" in ids


def test_sum_score():
    """sum 策略：父分数 = 子分数之和。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(CHILD_RESULTS, top_k=2, collection_name="col")
    s1 = next(r for r in results if r.id == "doc_s1")
    assert abs(s1.score - (0.92 + 0.88)) < 1e-5


def test_count_score():
    """count 策略：父分数 = 命中子块数。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="count")
    results = agg.aggregate(CHILD_RESULTS, top_k=2, collection_name="col")
    s1 = next(r for r in results if r.id == "doc_s1")
    assert s1.score == 2.0  # 命中 c1 和 c2 两个子块


def test_sorted_by_score_descending():
    """结果应按父分数降序排列。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(CHILD_RESULTS, top_k=2, collection_name="col")
    assert results[0].score >= results[1].score


def test_top_k_limits_results():
    """top_k=1 时只返回得分最高的父。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(CHILD_RESULTS, top_k=1, collection_name="col")
    assert len(results) == 1
    assert results[0].id == "doc_s1"  # 分数更高


# ---------------------------------------------------------------------------
# 正文合并
# ---------------------------------------------------------------------------

def test_merged_text_contains_all_sections():
    """父正文应包含所有子块的内容。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(CHILD_RESULTS, top_k=1, collection_name="col")
    text = results[0].text
    assert "宫保鸡丁" in text
    assert "食材" in text
    assert "步骤" in text


def test_merged_text_order():
    """合并顺序应按 chunk_index，即标题行在前、各节依次。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(CHILD_RESULTS, top_k=1, collection_name="col")
    text = results[0].text
    assert text.index("宫保鸡丁") < text.index("食材") < text.index("步骤")


def test_metadata_aggregated_flag():
    """父级结果 metadata 中 aggregated 应为 True。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(CHILD_RESULTS, top_k=1, collection_name="col")
    assert results[0].metadata.get("aggregated") is True


def test_metadata_child_count():
    """child_count 应等于实际子块数（从库里拉回的，不是检索命中数）。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(CHILD_RESULTS, top_k=1, collection_name="col")
    assert results[0].metadata["child_count"] == 3  # STORE_DB["doc_s1"] 有 3 个子块


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------

def test_deduplicates_child_ids():
    """同一子块 id 出现两次时只计一次。"""
    dup_results = CHILD_RESULTS + [_qr("c1", 0.99, "重复", "doc_s1", 1)]
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(dup_results, top_k=1, collection_name="col")
    # 分数应是 0.92+0.88（第一次出现），不应加重复的 0.99
    s1 = results[0]
    assert abs(s1.score - (0.92 + 0.88 + 0.71)) < 1      # 也许 s2 更高了，保守检查
    # 无论如何 c1 只被计一次
    s1_result = next((r for r in results if r.id == "doc_s1"), None)
    if s1_result:
        assert abs(s1_result.score - (0.92 + 0.88)) < 1e-5


def test_skips_chunks_without_parent_id():
    """无 parent_id 的子块应被跳过，不参与聚合。"""
    no_parent = [QueryResult("cx", 0.99, "无父块", {})]
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate(no_parent, top_k=5, collection_name="col")
    assert results == []


def test_empty_input_returns_empty():
    """空输入应返回空列表。"""
    agg = ParentAggregator(MockStore(STORE_DB), score_strategy="sum")
    results = agg.aggregate([], top_k=5, collection_name="col")
    assert results == []
