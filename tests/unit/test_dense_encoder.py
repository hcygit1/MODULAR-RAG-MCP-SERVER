"""
DenseEncoder 契约测试

验证批量编码功能、向量数量一致性和维度一致性。
"""
import pytest
from unittest.mock import MagicMock

from src.ingestion.models import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
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


class TestDenseEncoderBasic:
    """DenseEncoder 基础功能测试"""
    
    def test_dense_encoder_initialization(self):
        """测试 DenseEncoder 可以初始化"""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding)
        
        assert encoder is not None
        assert encoder.get_dimension() == 1536
        assert encoder.get_model_name() == "test-model"
        assert encoder.get_provider() == "test"
    
    def test_dense_encoder_initialization_with_batch_size(self):
        """测试带 batch_size 的初始化"""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding, batch_size=10)
        
        assert encoder is not None
        assert encoder._batch_size == 10
    
    def test_dense_encoder_initialization_with_none_embedding(self):
        """测试 embedding 为 None 时抛出错误"""
        with pytest.raises(ValueError, match="不能为 None"):
            DenseEncoder(None)
    
    def test_encode_single_chunk(self):
        """测试编码单个 Chunk"""
        embedding = MockEmbedding(dimension=128)
        encoder = DenseEncoder(embedding)
        
        chunk = Chunk(
            id="chunk_1",
            text="这是测试文本。",
            metadata={}
        )
        
        vectors = encoder.encode([chunk])
        
        assert len(vectors) == 1
        assert len(vectors[0]) == 128
        assert all(isinstance(v, float) for v in vectors[0])
    
    def test_encode_multiple_chunks(self):
        """测试编码多个 Chunks（核心验收标准）"""
        embedding = MockEmbedding(dimension=256)
        encoder = DenseEncoder(embedding)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"文本内容 {i}", metadata={})
            for i in range(5)
        ]
        
        vectors = encoder.encode(chunks)
        
        # 验证向量数量与 chunks 数量一致（核心验收标准）
        assert len(vectors) == len(chunks)
        assert len(vectors) == 5
        
        # 验证所有向量维度一致（核心验收标准）
        expected_dim = 256
        for vector in vectors:
            assert len(vector) == expected_dim


class TestDenseEncoderBatchProcessing:
    """DenseEncoder 批处理测试"""
    
    def test_batch_processing_with_batch_size(self):
        """测试批处理功能"""
        embedding = MockEmbedding(dimension=512)
        
        # 创建 mock，记录调用次数
        call_count = []
        original_embed = embedding.embed
        
        def tracked_embed(texts, trace=None):
            call_count.append(len(texts))
            return original_embed(texts, trace)
        
        embedding.embed = tracked_embed
        
        encoder = DenseEncoder(embedding, batch_size=3)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"文本 {i}", metadata={})
            for i in range(10)
        ]
        
        vectors = encoder.encode(chunks)
        
        # 验证批处理：10 个 chunks，batch_size=3，应该分 4 批（3+3+3+1）
        assert len(call_count) == 4
        assert call_count[0] == 3
        assert call_count[1] == 3
        assert call_count[2] == 3
        assert call_count[3] == 1
        
        # 验证输出数量
        assert len(vectors) == 10
    
    def test_batch_processing_without_batch_size(self):
        """测试不使用批处理（一次性处理）"""
        embedding = MockEmbedding(dimension=384)
        
        call_count = []
        original_embed = embedding.embed
        
        def tracked_embed(texts, trace=None):
            call_count.append(len(texts))
            return original_embed(texts, trace)
        
        embedding.embed = tracked_embed
        
        encoder = DenseEncoder(embedding)  # batch_size=None
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"文本 {i}", metadata={})
            for i in range(5)
        ]
        
        vectors = encoder.encode(chunks)
        
        # 应该一次性处理所有
        assert len(call_count) == 1
        assert call_count[0] == 5
        assert len(vectors) == 5


class TestDenseEncoderValidation:
    """DenseEncoder 验证测试"""
    
    def test_encode_empty_chunks(self):
        """测试空 chunks 列表抛出错误"""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding)
        
        with pytest.raises(ValueError, match="不能为空"):
            encoder.encode([])
    
    def test_encode_vectors_count_mismatch(self):
        """测试向量数量不一致时抛出错误"""
        embedding = MockEmbedding()
        
        # Mock embed 返回错误数量的向量
        def bad_embed(texts, trace=None):
            # 返回比输入少一个的向量
            return [[0.1] * 1536 for _ in texts[:-1]]
        
        embedding.embed = bad_embed
        
        encoder = DenseEncoder(embedding)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"文本 {i}", metadata={})
            for i in range(3)
        ]
        
        with pytest.raises(RuntimeError, match="向量数量.*不一致"):
            encoder.encode(chunks)
    
    def test_encode_dimension_mismatch(self):
        """测试向量维度不一致时抛出错误"""
        embedding = MockEmbedding(dimension=1536)
        
        # Mock embed 返回不一致维度的向量
        def bad_embed(texts, trace=None):
            vectors = []
            for i, _ in enumerate(texts):
                # 第一个向量维度正确，后面的维度错误
                dim = 1536 if i == 0 else 1024
                vectors.append([0.1] * dim)
            return vectors
        
        embedding.embed = bad_embed
        
        encoder = DenseEncoder(embedding)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"文本 {i}", metadata={})
            for i in range(3)
        ]
        
        with pytest.raises(RuntimeError, match="向量维度不一致"):
            encoder.encode(chunks)
    
    def test_encode_preserves_chunk_order(self):
        """测试保持 Chunk 顺序"""
        embedding = MockEmbedding(dimension=128)
        
        # Mock embed 返回带标识的向量（第一个元素是索引）
        def indexed_embed(texts, trace=None):
            vectors = []
            for idx, _ in enumerate(texts):
                vector = [float(idx)] + [0.1] * (128 - 1)  # 第一个元素是索引
                vectors.append(vector)
            return vectors
        
        embedding.embed = indexed_embed
        
        encoder = DenseEncoder(embedding)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"文本 {i}", metadata={})
            for i in range(5)
        ]
        
        vectors = encoder.encode(chunks)
        
        # 验证顺序：每个向量的第一个元素应该等于索引
        for idx, vector in enumerate(vectors):
            assert vector[0] == float(idx)


class TestDenseEncoderEdgeCases:
    """DenseEncoder 边界情况测试"""
    
    def test_encode_with_empty_text(self):
        """测试包含空文本的 Chunk（应该由 Chunk 模型验证，这里测试编码器行为）"""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding)
        
        # Chunk 模型不允许空文本，所以这里测试的是如果传入空文本列表的情况
        # 实际上这种情况不应该发生，因为 Chunk 模型会验证
        
        # 测试 embedding 返回空向量列表的情况
        def empty_embed(texts, trace=None):
            return []
        
        embedding.embed = empty_embed
        
        chunk = Chunk(
            id="chunk_1",
            text="正常文本",
            metadata={}
        )
        
        with pytest.raises(RuntimeError, match="向量数量.*不一致"):
            encoder.encode([chunk])
    
    def test_encode_large_batch(self):
        """测试大批量编码"""
        embedding = MockEmbedding(dimension=768)
        encoder = DenseEncoder(embedding, batch_size=100)
        
        chunks = [
            Chunk(id=f"chunk_{i}", text=f"文本内容 {i}", metadata={})
            for i in range(250)
        ]
        
        vectors = encoder.encode(chunks)
        
        assert len(vectors) == 250
        assert all(len(v) == 768 for v in vectors)
    
    def test_get_dimension(self):
        """测试获取维度"""
        embedding = MockEmbedding(dimension=1024)
        encoder = DenseEncoder(embedding)
        
        assert encoder.get_dimension() == 1024
    
    def test_get_model_name(self):
        """测试获取模型名称"""
        embedding = MockEmbedding(model_name="custom-model")
        encoder = DenseEncoder(embedding)
        
        assert encoder.get_model_name() == "custom-model"
    
    def test_get_provider(self):
        """测试获取 provider"""
        embedding = MockEmbedding(provider="custom-provider")
        encoder = DenseEncoder(embedding)
        
        assert encoder.get_provider() == "custom-provider"
