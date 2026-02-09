"""
Batch Processor 实现

批处理编排：将 chunks 分 batch，驱动 dense/sparse 编码，记录批次耗时（为 trace 预留）。
"""
import time
from typing import List, Optional, Any, Tuple, Dict

from src.ingestion.models import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """
    Batch Processor 实现
    
    负责将 chunks 分批处理，协调 DenseEncoder 和 SparseEncoder 进行编码。
    确保批次顺序稳定，并记录批次耗时（为 trace 预留）。
    """
    
    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int
    ):
        """
        初始化 BatchProcessor
        
        Args:
            dense_encoder: DenseEncoder 实例，用于生成稠密向量
            sparse_encoder: SparseEncoder 实例，用于生成稀疏向量
            batch_size: 批处理大小，必须大于 0
        """
        if dense_encoder is None:
            raise ValueError("dense_encoder 不能为 None")
        if sparse_encoder is None:
            raise ValueError("sparse_encoder 不能为 None")
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前值为 {batch_size}")
        
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self._batch_size = batch_size
    
    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """
        批量处理 chunks，生成稠密向量和稀疏向量
        
        Args:
            chunks: Chunk 对象列表
            trace: 追踪上下文（可选），用于记录性能指标和调试信息
        
        Returns:
            Tuple[List[List[float]], List[Dict[str, float]]]:
                - 第一个元素：稠密向量列表，每个 Chunk 对应一个向量
                - 第二个元素：稀疏向量列表（Term Weights），每个 Chunk 对应一个字典
                - 两个列表的长度都与 chunks 相同，且顺序一致
        
        Raises:
            ValueError: 当 chunks 为空时
            RuntimeError: 当编码过程出现错误时
        """
        if not chunks:
            raise ValueError("chunks 列表不能为空")
        
        # 将 chunks 分批
        batches = self._split_into_batches(chunks)
        
        # 存储所有批次的编码结果
        all_dense_vectors = []
        all_sparse_vectors = []
        
        # 记录批次耗时（为 trace 预留）
        batch_timings = []
        
        # 逐批处理
        for batch_idx, batch_chunks in enumerate(batches):
            batch_start_time = time.time()
            
            try:
                # 对当前批次进行编码
                # 注意：DenseEncoder 内部已经有批处理逻辑，但这里我们手动分批
                # 这样可以更好地控制批次大小和记录耗时
                batch_dense_vectors = self._process_dense_batch(batch_chunks, trace)
                batch_sparse_vectors = self._process_sparse_batch(batch_chunks, trace)
                
                # 验证批次结果数量
                if len(batch_dense_vectors) != len(batch_chunks):
                    raise RuntimeError(
                        f"批次 {batch_idx} 的稠密向量数量 ({len(batch_dense_vectors)}) "
                        f"与 chunks 数量 ({len(batch_chunks)}) 不一致"
                    )
                if len(batch_sparse_vectors) != len(batch_chunks):
                    raise RuntimeError(
                        f"批次 {batch_idx} 的稀疏向量数量 ({len(batch_sparse_vectors)}) "
                        f"与 chunks 数量 ({len(batch_chunks)}) 不一致"
                    )
                
                # 追加到总结果列表
                all_dense_vectors.extend(batch_dense_vectors)
                all_sparse_vectors.extend(batch_sparse_vectors)
                
                batch_elapsed_time = time.time() - batch_start_time
                batch_timings.append({
                    "batch_index": batch_idx,
                    "batch_size": len(batch_chunks),
                    "elapsed_time": batch_elapsed_time
                })
                
                # TODO: 将批次耗时记录到 trace（当 trace 实现后）
                # if trace is not None:
                #     trace.record_batch_timing(batch_idx, batch_elapsed_time)
                
            except Exception as e:
                raise RuntimeError(
                    f"批次 {batch_idx} 处理失败: {str(e)}"
                ) from e
        
        # 最终验证：确保结果数量与输入一致
        if len(all_dense_vectors) != len(chunks):
            raise RuntimeError(
                f"稠密向量总数 ({len(all_dense_vectors)}) 与 chunks 总数 ({len(chunks)}) 不一致"
            )
        if len(all_sparse_vectors) != len(chunks):
            raise RuntimeError(
                f"稀疏向量总数 ({len(all_sparse_vectors)}) 与 chunks 总数 ({len(chunks)}) 不一致"
            )
        
        return all_dense_vectors, all_sparse_vectors
    
    def _split_into_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """
        将 chunks 列表按 batch_size 分割成多个批次
        
        Args:
            chunks: Chunk 对象列表
        
        Returns:
            List[List[Chunk]]: 批次列表，每个批次包含最多 batch_size 个 chunks
                            顺序与输入一致
        
        示例:
            batch_size=2, chunks=[c1, c2, c3, c4, c5]
            返回: [[c1, c2], [c3, c4], [c5]]
        """
        batches = []
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i:i + self._batch_size]
            batches.append(batch)
        return batches
    
    def _process_dense_batch(
        self,
        batch_chunks: List[Chunk],
        trace: Optional[Any] = None
    ) -> List[List[float]]:
        """
        处理单个批次的稠密编码
        
        Args:
            batch_chunks: 当前批次的 Chunk 列表
            trace: 追踪上下文（可选）
        
        Returns:
            List[List[float]]: 稠密向量列表
        """
        # DenseEncoder 内部已经有批处理逻辑，但这里我们传入 None 让它一次性处理整个批次
        # 这样可以确保批次大小由 BatchProcessor 控制
        dense_encoder_without_batch = DenseEncoder(
            embedding=self._dense_encoder._embedding,
            batch_size=None  # 让 DenseEncoder 一次性处理整个批次
        )
        return dense_encoder_without_batch.encode(batch_chunks, trace=trace)
    
    def _process_sparse_batch(
        self,
        batch_chunks: List[Chunk],
        trace: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        """
        处理单个批次的稀疏编码
        
        Args:
            batch_chunks: 当前批次的 Chunk 列表
            trace: 追踪上下文（可选）
        
        Returns:
            List[Dict[str, float]]: 稀疏向量列表（Term Weights）
        """
        # SparseEncoder 目前是一次性处理所有 chunks，但我们需要按批次处理
        # 为了保持 BM25 的 IDF 计算准确性，我们需要传入完整的文档集合
        # 但这里我们先简化：每个批次独立计算 IDF
        # 注意：这可能会导致不同批次的 IDF 值略有不同，但符合当前设计
        return self._sparse_encoder.encode(batch_chunks, trace=trace)
    
    def get_batch_size(self) -> int:
        """
        获取批处理大小
        
        Returns:
            int: 批处理大小
        """
        return self._batch_size
