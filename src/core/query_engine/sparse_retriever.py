"""
SparseRetriever 实现

从统一存储 backend 的稀疏索引检索（SQLite FTS5 等）。
接收 VectorStore，调用其 sparse_query 能力；或传入 sqlite_path 使用 FTS5。
"""
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.libs.tokenizer import tokenize_for_search
from src.libs.vector_store.base_vector_store import BaseVectorStore, QueryResult


class SparseRetriever:
    """
    Sparse Retriever（FTS5 / 统一存储）

    从 VectorStore.sparse_query 或 FTS5（sqlite_path）检索，返回 QueryResult。
    """

    def __init__(
        self,
        *,
        vector_store: Optional[BaseVectorStore] = None,
        sqlite_path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        初始化 SparseRetriever

        Args:
            vector_store: 支持 sparse_query 的 VectorStore（如 SQLiteVectorStore write_fts=True）
            sqlite_path: SQLite 路径（当未传 vector_store 时使用，与 VectorStore 共用）
            collection_name: 默认集合名称，retrieve 时可覆盖
        """
        if vector_store is None and not sqlite_path:
            raise ValueError("需提供 vector_store 或 sqlite_path")
        self._vector_store = vector_store
        self._sqlite_path = Path(sqlite_path) if sqlite_path else None
        self._default_collection = collection_name
        self._fts5_indexers: Dict[str, Any] = {}

    def retrieve(
        self,
        query: Union[str, List[str]],
        top_k: int,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[QueryResult]:
        """
        执行稀疏检索

        Args:
            query: 关键词列表或空格分隔字符串
            top_k: 返回的 Top-K 数量
            collection_name: 集合名称，为 None 时使用默认
            filters: 元数据过滤条件（可选）
            trace: 追踪上下文（可选）

        Returns:
            List[QueryResult]: 检索结果列表
        """
        keywords = self._to_keywords(query)
        if not keywords:
            return []

        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，得到: {top_k}")

        coll = collection_name or self._default_collection
        if not coll:
            raise ValueError("collection_name 未指定，且无默认集合")

        results = self._do_retrieve(keywords, coll, top_k, trace)
        if filters:
            results = self._apply_filters(results, filters)
        return results[:top_k]

    def _to_keywords(self, query: Union[str, List[str]]) -> List[str]:
        """将 query 转为关键词列表。"""
        if isinstance(query, list):
            return [str(k).strip() for k in query if str(k).strip()]
        if not isinstance(query, str):
            return []
        return tokenize_for_search(query)

    def _do_retrieve(
        self,
        terms: List[str],
        collection_name: str,
        top_k: int,
        trace: Optional[Any],
    ) -> List[QueryResult]:
        """执行检索：优先使用 vector_store.sparse_query，否则 FTS5。"""
        if self._vector_store is not None and hasattr(
            self._vector_store, "sparse_query"
        ):
            return self._vector_store.sparse_query(
                terms=terms,
                collection_name=collection_name,
                top_k=top_k,
                trace=trace,
            )
        if self._sqlite_path:
            return self._retrieve_via_fts5(terms, collection_name, top_k, trace)
        return []

    def _retrieve_via_fts5(
        self,
        terms: List[str],
        collection_name: str,
        top_k: int,
        trace: Optional[Any],
    ) -> List[QueryResult]:
        """通过 FTS5BM25Indexer 检索。"""
        from src.ingestion.storage.fts5_bm25_indexer import FTS5BM25Indexer

        if collection_name not in self._fts5_indexers:
            idx = FTS5BM25Indexer(sqlite_path=str(self._sqlite_path))
            idx.load(collection_name)
            self._fts5_indexers[collection_name] = idx
        indexer = self._fts5_indexers[collection_name]
        raw = indexer.query(terms=terms, top_k=top_k, trace=trace)
        chunk_meta = self._load_chunk_metadata_from_sqlite(collection_name)
        results = []
        for chunk_id, score in raw:
            info = chunk_meta.get(chunk_id, {})
            results.append(
                QueryResult(
                    id=chunk_id,
                    score=score,
                    text=info.get("text", ""),
                    metadata=info.get("metadata", {}),
                )
            )
        return results

    def _load_chunk_metadata_from_sqlite(
        self, collection_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """从 chunks 表加载 text 和 metadata。"""
        conn = sqlite3.connect(str(self._sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, text, metadata_json FROM chunks WHERE collection_name = ?",
                (collection_name,),
            ).fetchall()
        finally:
            conn.close()
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            meta = {}
            if row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    pass
            out[row["id"]] = {"text": row["text"] or "", "metadata": meta}
        return out

    def index_exists(self, collection_name: str) -> bool:
        """检查 collection 的稀疏索引是否存在。"""
        if not collection_name:
            return False
        if self._vector_store is not None and hasattr(
            self._vector_store, "sparse_index_exists"
        ):
            return self._vector_store.sparse_index_exists(collection_name)
        if self._sqlite_path:
            from src.ingestion.storage.fts5_bm25_indexer import FTS5BM25Indexer

            idx = FTS5BM25Indexer(sqlite_path=str(self._sqlite_path))
            try:
                return idx.index_exists(collection_name)
            finally:
                idx.close()
        return False

    def _apply_filters(
        self,
        results: List[QueryResult],
        filters: Dict[str, Any],
    ) -> List[QueryResult]:
        """对结果应用 metadata 过滤"""
        return [
            r
            for r in results
            if self._matches_filters(r.metadata, filters)
        ]

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查 metadata 是否匹配 filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
