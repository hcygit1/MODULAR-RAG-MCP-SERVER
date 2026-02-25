"""
RRF (Reciprocal Rank Fusion) 实现

将 dense 与 sparse 两路检索结果按 RRF 算法融合，输出统一排序。
公式: Score = 1/(k + Rank_Dense) + 1/(k + Rank_Sparse)
"""
from typing import List

from src.libs.vector_store.base_vector_store import QueryResult


def fuse_rrf(
    dense_results: List[QueryResult],
    sparse_results: List[QueryResult],
    k: float = 60.0,
) -> List[QueryResult]:
    """
    使用 RRF 融合 dense 与 sparse 排名。

    对每个文档，根据其在两路结果中的排名分别贡献 RRF 分数并累加：
    - 在 dense 中排第 rank_d → 贡献 1/(k + rank_d)
    - 在 sparse 中排第 rank_s → 贡献 1/(k + rank_s)
    若某路中未出现则该路贡献为 0。

    Args:
        dense_results: Dense 检索结果，按 score 降序
        sparse_results: Sparse 检索结果，按 score 降序
        k: RRF 平滑常数，默认 60

    Returns:
        按 RRF 分数降序排列的 QueryResult 列表；同分时按 id 字典序以保证 deterministic
    """
    scores: dict[str, float] = {}
    info: dict[str, QueryResult] = {}

    for rank, r in enumerate(dense_results, start=1):
        rrf = 1.0 / (k + rank)
        scores[r.id] = scores.get(r.id, 0.0) + rrf
        if r.id not in info:
            info[r.id] = r

    for rank, r in enumerate(sparse_results, start=1):
        rrf = 1.0 / (k + rank)
        scores[r.id] = scores.get(r.id, 0.0) + rrf
        if r.id not in info:
            info[r.id] = r

    # 按 RRF 分数降序，同分按 id 升序以保证 deterministic
    sorted_ids = sorted(
        scores.keys(),
        key=lambda x: (-scores[x], x),
    )

    return [
        QueryResult(
            id=doc_id,
            score=scores[doc_id],
            text=info[doc_id].text,
            metadata=info[doc_id].metadata,
        )
        for doc_id in sorted_ids
    ]
