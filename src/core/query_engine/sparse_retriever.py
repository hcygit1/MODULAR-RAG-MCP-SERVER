"""
SparseRetriever 实现

从 data/db/bm25/ 载入 BM25 索引，使用关键词进行稀疏检索。
返回与 DenseRetriever 一致的 QueryResult 格式，便于 D4 Fusion 融合。
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.vector_store.base_vector_store import QueryResult


class SparseRetriever:
    """
    Sparse Retriever (BM25)

    从 BM25 倒排索引中检索，输入关键词列表，返回 Top-K 结果。
    结果格式与 DenseRetriever 一致（QueryResult），便于下游 Fusion 融合。
    """

    def __init__(
        self,
        base_path: str = "data/db/bm25",
        collection_name: Optional[str] = None,
    ) -> None:
        """
        初始化 SparseRetriever

        Args:
            base_path: BM25 索引存储的基础路径
            collection_name: 默认集合名称，retrieve 时可覆盖
        """
        self._base_path = Path(base_path)
        self._default_collection = collection_name
        self._indexers: Dict[str, BM25Indexer] = {}

    def retrieve(
        self,
        query: Union[str, List[str]],
        top_k: int,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        执行 BM25 稀疏检索

        Args:
            query: 关键词列表（或空格分隔的字符串）
            top_k: 返回的 Top-K 数量
            collection_name: 集合名称，为 None 时使用默认
            filters: 元数据过滤条件（可选），对结果做 post-filter
            trace: 追踪上下文（可选）

        Returns:
            List[QueryResult]: 检索结果列表，按 score 降序

        Raises:
            ValueError: 当 query 为空、top_k <= 0 或 collection 未指定且无默认时
            FileNotFoundError: 当索引文件不存在时
        """
        keywords = self._to_keywords(query)
        if not keywords:
            return []

        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")

        coll = collection_name or self._default_collection
        if not coll:
            raise ValueError("collection_name 未指定，且无默认集合")

        indexer = self._get_indexer(coll)
        raw_results = indexer.query(terms=keywords, top_k=top_k, trace=trace)

        results = self._to_query_results(indexer, coll, raw_results)
        if filters:
            results = self._apply_filters(results, filters)
        return results[:top_k]

    def _to_keywords(self, query: Union[str, List[str]]) -> List[str]:
        """将 query 转为关键词列表"""
        if isinstance(query, str):
            parts = query.strip().split()
            return [p for p in parts if p]
        if isinstance(query, list):
            return [str(k).strip() for k in query if str(k).strip()]
        return []

    def index_exists(self, collection_name: str) -> bool:
        """检查 collection 的 BM25 索引是否存在。"""
        if not collection_name:
            return False
        index_file = self._base_path / collection_name / "index.json"
        return index_file.exists()

    def _get_indexer(self, collection_name: str) -> BM25Indexer:
        """获取或加载指定集合的 BM25Indexer"""
        if collection_name not in self._indexers:
            indexer = BM25Indexer(base_path=str(self._base_path))
            indexer.load(collection_name)
            self._indexers[collection_name] = indexer
        return self._indexers[collection_name]

    def _to_query_results(
        self,
        indexer: BM25Indexer,
        collection_name: str,
        raw_results: List[tuple],
    ) -> List[QueryResult]:
        """将 (chunk_id, score) 转为 QueryResult，补齐 text 和 metadata"""
        chunk_meta = self._load_chunk_metadata(collection_name)
        results = []
        for chunk_id, score in raw_results:
            info = chunk_meta.get(chunk_id, {})
            text = info.get("text", "")
            metadata = info.get("metadata", {})
            results.append(
                QueryResult(id=chunk_id, score=score, text=text, metadata=metadata)
            )
        return results

    def _load_chunk_metadata(self, collection_name: str) -> Dict[str, Dict[str, Any]]:
        """从索引文件加载 chunk_metadata"""
        index_file = self._base_path / collection_name / "index.json"
        if not index_file.exists():
            return {}
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("chunk_metadata", {})

    def _apply_filters(
        self,
        results: List[QueryResult],
        filters: Dict[str, Any],
    ) -> List[QueryResult]:
        """对结果应用 metadata 过滤"""
        filtered = []
        for r in results:
            if self._matches_filters(r.metadata, filters):
                filtered.append(r)
        return filtered

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查 metadata 是否匹配 filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
