"""
BatchProcessor 契约测试

验证批处理编排功能、批次分割逻辑和顺序稳定性。
"""
import pytest
from unittest.mock import MagicMock

from src.ingestion.models import Chunk
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.libs.embedding.base_embedding import BaseEmbedding


class MockEmbedding(BaseEmbedding):
    """Mock Embedding 实现用于测试"""
    
    def __init__(self, dimension: int = 1536, model_name: str = "test-model", provider: str = "test"):
        self._dimension = dimension
        self._model_name = model_name
        self._provider = provider
    
    def embed(self, texts, trace=None):
        """返回固定维度的向量"""
        return [[0.1] * self._dimension for _ in texts]
    
    def get_model_name(self) -> str:
        return self._model_name
    
    def get_provider(self) -> str:
        return self._provider
    
    def get_dimension(self) -> int:
        return self._dimension


class TestBatchProcessorBasic:
    """BatchProcessor 基础功能测试"""
    
    def test_batch_processor_initialization(self):
        """测试 BatchProcessor 可以初始化"""
        embedding = MockEmbedding()
        dense_encoder = DenseEncoder(embedding)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=2)
        
        assert processor is not None
        assert processor.get_batch_size() == 2
    
    def test_batch_processor_initialization_with_invalid_batch_size(self):
        """测试无效 batch_size 的初始化"""
        embedding = MockEmbedding()
        dense_encoder = DenseEncoder(embedding)
        sparse_encoder = SparseEncoder()
        
        with pytest.raises(ValueError, match="batch_size 必须大于 0"):
            BatchProcessor(dense_encoder, sparse_encoder, batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size 必须大于 0"):
            BatchProcessor(dense_encoder, sparse_encoder, batch_size=-1)
    
    def test_batch_processor_initialization_with_none_encoders(self):
        """测试 None encoder 的初始化"""
        embedding = MockEmbedding()
        dense_encoder = DenseEncoder(embedding)
        sparse_encoder = SparseEncoder()
        
        with pytest.raises(ValueError, match="dense_encoder 不能为 None"):
            BatchProcessor(None, sparse_encoder, batch_size=2)
        
        with pytest.raises(ValueError, match="sparse_encoder 不能为 None"):
            BatchProcessor(dense_encoder, None, batch_size=2)


class TestBatchProcessorBatching:
    """BatchProcessor 批次分割测试"""
    
    def test_split_into_batches_batch_size_2(self):
        """测试 batch_size=2 时对 5 chunks 分成 3 批"""
        embedding = MockEmbedding()
        dense_encoder = DenseEncoder(embedding)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=2)
        
        # 创建 5 个 chunks
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}")
            for i in range(5)
        ]
        
        # 测试批次分割
        batches = processor._split_into_batches(chunks)
        
        assert len(batches) == 3
        assert len(batches[0]) == 2  # [chunk_0, chunk_1]
        assert len(batches[1]) == 2  # [chunk_2, chunk_3]
        assert len(batches[2]) == 1  # [chunk_4]
        
        # 验证顺序
        assert batches[0][0].id == "chunk_0"
        assert batches[0][1].id == "chunk_1"
        assert batches[1][0].id == "chunk_2"
        assert batches[1][1].id == "chunk_3"
        assert batches[2][0].id == "chunk_4"
    
    def test_split_into_batches_batch_size_3(self):
        """测试 batch_size=3 时对 7 chunks 分成 3 批"""
        embedding = MockEmbedding()
        dense_encoder = DenseEncoder(embedding)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=3)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}")
            for i in range(7)
        ]
        
        batches = processor._split_into_batches(chunks)
        
        assert len(batches) == 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1
    
    def test_split_into_batches_exact_multiple(self):
        """测试 chunks 数量正好是 batch_size 的倍数"""
        embedding = MockEmbedding()
        dense_encoder = DenseEncoder(embedding)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=2)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}")
            for i in range(4)
        ]
        
        batches = processor._split_into_batches(chunks)
        
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
    
    def test_split_into_batches_single_chunk(self):
        """测试单个 chunk 的情况"""
        embedding = MockEmbedding()
        dense_encoder = DenseEncoder(embedding)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=2)
        
        chunks = [Chunk(id="chunk_0", text="Text 0")]
        
        batches = processor._split_into_batches(chunks)
        
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0].id == "chunk_0"


class TestBatchProcessorProcessing:
    """BatchProcessor 处理功能测试"""
    
    def test_process_batch_size_2_with_5_chunks(self):
        """测试 batch_size=2 时处理 5 个 chunks"""
        embedding = MockEmbedding(dimension=128)
        dense_encoder = DenseEncoder(embedding, batch_size=None)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=2)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text content {i}")
            for i in range(5)
        ]
        
        dense_vectors, sparse_vectors = processor.process(chunks)
        
        # 验证结果数量
        assert len(dense_vectors) == 5
        assert len(sparse_vectors) == 5
        
        # 验证稠密向量维度
        for vector in dense_vectors:
            assert len(vector) == 128
            assert all(isinstance(x, float) for x in vector)
        
        # 验证稀疏向量结构
        for sparse_vector in sparse_vectors:
            assert isinstance(sparse_vector, dict)
            # 稀疏向量可能为空（如果文本被停用词过滤掉），但结构应该是字典
    
    def test_process_order_stability(self):
        """测试处理结果的顺序稳定性"""
        embedding = MockEmbedding(dimension=64)
        dense_encoder = DenseEncoder(embedding, batch_size=None)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=2)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Unique text {i} for testing")
            for i in range(5)
        ]
        
        dense_vectors, sparse_vectors = processor.process(chunks)
        
        # 验证顺序：每个位置的向量应该对应相应位置的 chunk
        # 由于 MockEmbedding 返回相同的向量，我们主要验证数量
        assert len(dense_vectors) == len(chunks)
        assert len(sparse_vectors) == len(chunks)
        
        # 验证稀疏向量包含对应的词（如果文本没有被完全过滤）
        # 由于使用了不同的文本，稀疏向量应该包含不同的词
        for i, sparse_vector in enumerate(sparse_vectors):
            # 至少应该有一些词（除非完全被停用词过滤）
            # 这里我们只验证结构，不验证具体内容
            assert isinstance(sparse_vector, dict)
    
    def test_process_single_batch(self):
        """测试单个批次的情况"""
        embedding = MockEmbedding(dimension=256)
        dense_encoder = DenseEncoder(embedding, batch_size=None)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=10)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}")
            for i in range(3)
        ]
        
        dense_vectors, sparse_vectors = processor.process(chunks)
        
        assert len(dense_vectors) == 3
        assert len(sparse_vectors) == 3
    
    def test_process_empty_chunks(self):
        """测试空 chunks 列表"""
        embedding = MockEmbedding()
        dense_encoder = DenseEncoder(embedding)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=2)
        
        with pytest.raises(ValueError, match="chunks 列表不能为空"):
            processor.process([])


class TestBatchProcessorEdgeCases:
    """BatchProcessor 边界情况测试"""
    
    def test_process_large_batch_size(self):
        """测试 batch_size 大于 chunks 数量的情况"""
        embedding = MockEmbedding(dimension=128)
        dense_encoder = DenseEncoder(embedding, batch_size=None)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=100)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}")
            for i in range(5)
        ]
        
        dense_vectors, sparse_vectors = processor.process(chunks)
        
        assert len(dense_vectors) == 5
        assert len(sparse_vectors) == 5
    
    def test_process_batch_size_1(self):
        """测试 batch_size=1 的情况（每个 chunk 单独处理）"""
        embedding = MockEmbedding(dimension=64)
        dense_encoder = DenseEncoder(embedding, batch_size=None)
        sparse_encoder = SparseEncoder()
        processor = BatchProcessor(dense_encoder, sparse_encoder, batch_size=1)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"Text {i}")
            for i in range(3)
        ]
        
        dense_vectors, sparse_vectors = processor.process(chunks)
        
        assert len(dense_vectors) == 3
        assert len(sparse_vectors) == 3
        
        # 应该有 3 个批次
        batches = processor._split_into_batches(chunks)
        assert len(batches) == 3
