"""
DenseRetriever 实现

调用 Embedding 将 query 向量化，再通过 VectorStore.query 进行稠密向量检索。
透传 VectorStore 返回的候选列表，并对 score 进行规范化。
"""
from typing import Any, Dict, List, Optional, Union

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    QueryResult,
)


class DenseRetriever:
    """
    Dense Retriever

    负责稠密向量检索：将 query 转为 embedding，调用 VectorStore 查询，
    透传候选列表并规范化 score（映射到 [0, 1] 范围）。
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        score_normalize: bool = True,
    ) -> None:
        """
        初始化 DenseRetriever

        Args:
            embedding: Embedding 实例，用于 query 向量化
            vector_store: VectorStore 实例，用于向量相似度查询
            score_normalize: 是否规范化 score 到 [0, 1]，默认 True
        """
        self._embedding = embedding
        self._vector_store = vector_store
        self._score_normalize = score_normalize

    def retrieve(
        self,
        query: Union[str, List[str]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        执行稠密向量检索

        Args:
            query: 查询字符串，或已分词的词列表（将用空格拼接为字符串）
            top_k: 返回的 Top-K 数量
            filters: 元数据过滤条件（可选）
            trace: 追踪上下文（可选）

        Returns:
            List[QueryResult]: 检索结果列表，按 score 降序，score 已规范化

        Raises:
            ValueError: 当 query 为空或 top_k <= 0 时
        """
        query_text = self._to_query_text(query)
        if not query_text or not query_text.strip():
            raise ValueError("query 不能为空")

        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")

        # 1. query 向量化
        vectors = self._embedding.embed([query_text], trace=trace)
        if not vectors:
            return []
        query_vector = vectors[0]

        # 2. 调用 VectorStore 查询
        raw_results = self._vector_store.query(
            vector=query_vector,
            top_k=top_k,
            filters=filters,
            trace=trace,
        )

        # 3. 透传结果，可选规范化 score
        if self._score_normalize and raw_results:
            return self._normalize_scores(raw_results)
        return raw_results

    def _to_query_text(self, query: Union[str, List[str]]) -> str:
        """将 query 转为单个字符串"""
        if isinstance(query, str):
            return query
        if isinstance(query, list):
            return " ".join(str(q) for q in query)
        return str(query)

    def _normalize_scores(self, results: List[QueryResult]) -> List[QueryResult]:
        """
        将 score 规范化到 [0, 1] 范围

        使用 min-max 归一化，保持相对排序不变。
        若所有 score 相同，则设为 1.0。
        """
        if not results:
            return results

        scores = [r.score for r in results]
        s_min = min(scores)
        s_max = max(scores)

        normalized = []
        for r in results:
            if s_max <= s_min:
                score = 1.0
            else:
                score = (r.score - s_min) / (s_max - s_min)
            normalized.append(
                QueryResult(
                    id=r.id,
                    score=min(1.0, max(0.0, score)),
                    text=r.text,
                    metadata=r.metadata,
                )
            )
        return normalized
