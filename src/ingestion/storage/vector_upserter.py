"""
Vector Upserter 实现

负责将编码后的 chunks 写入向量数据库，并确保幂等性。
生成稳定的 chunk_id（hash(source_path + section_path + content_hash)）。
"""
import hashlib
from typing import List, Dict, Optional, Any

from src.ingestion.models import Chunk
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class VectorUpserter:
    """
    Vector Upserter 实现
    
    负责将 Chunks 及其向量编码结果批量写入向量数据库。
    确保幂等性：同一 chunk 两次 upsert 产生相同 id；内容变更 id 变更。
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
        trace: Optional[Any] = None
    ) -> None:
        """
        批量将 Chunks 及其向量写入向量数据库
        
        Args:
            chunks: Chunk 对象列表
            dense_vectors: 稠密向量列表，每个 Chunk 对应一个向量
            sparse_vectors: 稀疏向量列表（Term Weights），每个 Chunk 对应一个字典
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
        
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
        
        # 为每个 chunk 生成稳定的 chunk_id 并组装 VectorRecord
        records = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk)
            dense_vector = dense_vectors[i]
            sparse_vector = sparse_vectors[i]
            
            # 组装完整的 metadata（包含稀疏向量信息）
            enriched_metadata = chunk.metadata.copy()
            enriched_metadata["sparse_vector"] = sparse_vector  # 将稀疏向量存入 metadata
            
            # 创建 VectorRecord
            record = VectorRecord(
                id=chunk_id,
                vector=dense_vector,
                text=chunk.text,
                metadata=enriched_metadata
            )
            records.append(record)
        
        # 批量写入向量数据库
        self._vector_store.upsert(records, trace=trace)
    
    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """
        为 Chunk 生成稳定的 chunk_id
        
        算法：hash(source_path + section_path + content_hash)
        
        Args:
            chunk: Chunk 对象
        
        Returns:
            str: 稳定的 chunk_id（十六进制字符串）
        
        Raises:
            ValueError: 当 chunk.metadata 中缺少必需的字段时
        """
        # 从 metadata 中提取 source_path
        source_path = chunk.metadata.get("source_path", "")
        if not source_path:
            # 如果 metadata 中没有 source_path，尝试从其他字段获取或使用默认值
            source_path = chunk.metadata.get("source", "")
            if not source_path:
                raise ValueError(
                    f"Chunk metadata 中缺少 'source_path' 或 'source' 字段，"
                    f"无法生成稳定的 chunk_id。Chunk id: {chunk.id}"
                )
        
        # 从 metadata 中提取 section_path（可选）
        section_path = chunk.metadata.get("section_path", "")
        if not section_path:
            # 如果没有 section_path，使用 chunk_index 作为替代
            chunk_index = chunk.metadata.get("chunk_index")
            if chunk_index is not None:
                section_path = f"chunk_{chunk_index}"
            else:
                section_path = ""
        
        # 计算 content_hash（chunk.text 的 SHA256 哈希）
        content_hash = self._compute_content_hash(chunk.text)
        
        # 组合并计算最终哈希
        # 使用分隔符确保不同字段不会混淆
        combined = f"{source_path}|{section_path}|{content_hash}"
        chunk_id = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        
        return chunk_id
    
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
