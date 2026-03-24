"""
Vector Upserter 实现

负责将编码后的 chunks 写入向量数据库，并确保幂等性。
使用 Pipeline 生成的 chunk.id 作为 record id，与 BM25 对齐，支持 Hybrid Search Fusion。
将 content_hash 存入 metadata，用于后续修改检测与增量优化。
"""
import hashlib
from typing import List, Dict, Optional, Any

from src.ingestion.models import Chunk
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class VectorUpserter:
    """
    Vector Upserter 实现
    
    负责将 Chunks 及其向量编码结果批量写入向量数据库。
    使用 chunk.id 作为 record id（与 BM25 一致），同一 chunk 多次 upsert 会覆盖更新。
    content_hash 写入 metadata，可用于判断 chunk 是否修改。
    """
    
    def __init__(self, vector_store: BaseVectorStore):
        """
        初始化 VectorUpserter
        
        Args:
            vector_store: BaseVectorStore 实例，用于执行实际的存储操作
        """
        if vector_store is None:
            raise ValueError("vector_store 不能为 None")
        
        self._vector_store = vector_store
    
    def upsert_chunks(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        批量将 Chunks 及其向量写入向量数据库

        Args:
            chunks: Chunk 对象列表
            dense_vectors: 稠密向量列表，每个 Chunk 对应一个向量
            sparse_vectors: 稀疏向量列表（Term Weights），每个 Chunk 对应一个字典
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
            collection_name: 集合名称（可选），为 None 时使用 VectorStore 配置中的默认集合

        Raises:
            ValueError: 当输入参数不匹配或格式不正确时
            RuntimeError: 当存储操作失败时
        """
        if not chunks:
            raise ValueError("chunks 列表不能为空")
        
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 dense_vectors 数量 ({len(dense_vectors)}) 不一致"
            )
        
        if len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 sparse_vectors 数量 ({len(sparse_vectors)}) 不一致"
            )
        
        records = self._build_records(chunks, dense_vectors, sparse_vectors)
        
        # 批量写入向量数据库。SQLite+write_fts 时传入 chunks 以同期写入 images
        store = self._vector_store
        if getattr(store, "_write_fts", False) and hasattr(store, "upsert"):
            store.upsert(
                records,
                trace=trace,
                collection_name=collection_name,
                chunks_for_images=chunks,
            )
        else:
            store.upsert(records, trace=trace, collection_name=collection_name)

    def replace_document_chunks(
        self,
        source_doc_id: str,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> int:
        """
        文档级替换写入：先删旧块，再写入新块。

        Args:
            source_doc_id: 文档 id（metadata.source_doc_id）
            chunks: 新 Chunk 对象列表
            dense_vectors: 稠密向量列表
            sparse_vectors: 稀疏向量列表
            trace: 追踪上下文（可选）
            collection_name: 集合名称（可选）

        Returns:
            int: 删除的旧 chunk 数量
        """
        doc_id = (source_doc_id or "").strip()
        if not doc_id:
            raise ValueError("source_doc_id 不能为空")

        # 先删除旧块，避免文档改短时残留旧 chunk。
        deleted_count = self._vector_store.delete_by_source_doc_id(
            collection_name=collection_name or self._vector_store.get_collection_name(),
            source_doc_id=doc_id,
            trace=trace,
        )
        self.upsert_chunks(
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            trace=trace,
            collection_name=collection_name,
        )
        return deleted_count

    def delete_chunks(
        self,
        chunk_ids: List[str],
        trace: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> int:
        """
        按 chunk_id 删除已写入的向量记录。用于 ingest 失败时回滚。

        Args:
            chunk_ids: 要删除的 chunk id 列表
            trace: 追踪上下文（可选）
            collection_name: 集合名称（可选）

        Returns:
            int: 实际删除的记录数量
        """
        if not chunk_ids:
            return 0
        return self._vector_store.delete(
            ids=chunk_ids,
            trace=trace,
            collection_name=collection_name,
        )

    def _compute_content_hash(self, text: str) -> str:
        """
        计算文本内容的哈希值
        
        Args:
            text: 文本内容
        
        Returns:
            str: SHA256 哈希值的十六进制字符串
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _build_records(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
    ) -> List[VectorRecord]:
        """将 Chunk 和向量组装为 VectorRecord 列表。"""
        records: List[VectorRecord] = []
        for i, chunk in enumerate(chunks):
            if not chunk.id or not str(chunk.id).strip():
                raise ValueError(
                    f"Chunk 缺少有效 id，无法与 BM25 对齐。index={i}"
                )
            chunk_id = chunk.id
            dense_vector = dense_vectors[i]
            sparse_vector = sparse_vectors[i]

            content_hash = self._compute_content_hash(chunk.text)
            enriched_metadata = chunk.metadata.copy()
            enriched_metadata["sparse_vector"] = sparse_vector
            enriched_metadata["content_hash"] = content_hash

            records.append(
                VectorRecord(
                    id=chunk_id,
                    vector=dense_vector,
                    text=chunk.text,
                    metadata=enriched_metadata,
                )
            )
        return records
    
    def get_vector_store(self) -> BaseVectorStore:
        """
        获取当前使用的 VectorStore 实例
        
        Returns:
            BaseVectorStore: VectorStore 实例
        """
        return self._vector_store
