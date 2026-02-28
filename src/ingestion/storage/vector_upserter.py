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
        
        # 使用 chunk.id 作为 record id（与 BM25 对齐），组装 VectorRecord
        records = []
        for i, chunk in enumerate(chunks):
            if not chunk.id or not str(chunk.id).strip():
                raise ValueError(
                    f"Chunk 缺少有效 id，无法与 BM25 对齐。index={i}"
                )
            chunk_id = chunk.id
            dense_vector = dense_vectors[i]
            sparse_vector = sparse_vectors[i]
            
            # 计算 content_hash 并存入 metadata（用于修改检测）
            content_hash = self._compute_content_hash(chunk.text)
            
            enriched_metadata = chunk.metadata.copy()
            enriched_metadata["sparse_vector"] = sparse_vector  # 将稀疏向量存入 metadata
            enriched_metadata["content_hash"] = content_hash  # 用于后续修改检测与增量优化
            
            record = VectorRecord(
                id=chunk_id,
                vector=dense_vector,
                text=chunk.text,
                metadata=enriched_metadata
            )
            records.append(record)
        
        # 批量写入向量数据库
        self._vector_store.upsert(records, trace=trace, collection_name=collection_name)

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
    
    def get_vector_store(self) -> BaseVectorStore:
        """
        获取当前使用的 VectorStore 实例
        
        Returns:
            BaseVectorStore: VectorStore 实例
        """
        return self._vector_store
