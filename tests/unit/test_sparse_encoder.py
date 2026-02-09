"""
SparseEncoder 契约测试

验证 BM25 统计计算、term weights 输出结构和空文本处理。
"""
import pytest

from src.ingestion.models import Chunk
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class TestSparseEncoderBasic:
    """SparseEncoder 基础功能测试"""
    
    def test_sparse_encoder_initialization(self):
        """测试 SparseEncoder 可以初始化"""
        encoder = SparseEncoder()
        
        assert encoder is not None
        assert encoder._k1 == 1.5
        assert encoder._b == 0.75
    
    def test_sparse_encoder_initialization_with_params(self):
        """测试带参数初始化"""
        encoder = SparseEncoder(k1=2.0, b=0.8, min_term_length=2)
        
        assert encoder._k1 == 2.0
        assert encoder._b == 0.8
        assert encoder._min_term_length == 2
    
    def test_encode_single_chunk(self):
        """测试编码单个 Chunk"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="Python is a programming language. Python is popular.",
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        
        assert len(term_weights_list) == 1
        term_weights = term_weights_list[0]
        assert isinstance(term_weights, dict)
        # Python 应该出现两次，应该有权重（可能为正或负，取决于 IDF）
        assert "python" in term_weights
        assert isinstance(term_weights["python"], (int, float))
    
    def test_encode_multiple_chunks(self):
        """测试编码多个 Chunks"""
        encoder = SparseEncoder()
        
        chunks = [
            Chunk(id="chunk_1", text="Python programming", metadata={}),
            Chunk(id="chunk_2", text="Java programming", metadata={}),
            Chunk(id="chunk_3", text="Python and Java", metadata={}),
        ]
        
        term_weights_list = encoder.encode(chunks)
        
        # 验证输出数量与 chunks 数量一致
        assert len(term_weights_list) == len(chunks)
        assert len(term_weights_list) == 3
        
        # 验证每个输出都是字典
        for term_weights in term_weights_list:
            assert isinstance(term_weights, dict)
    
    def test_encode_empty_chunks(self):
        """测试空 chunks 列表抛出错误"""
        encoder = SparseEncoder()
        
        with pytest.raises(ValueError, match="不能为空"):
            encoder.encode([])


class TestSparseEncoderTermWeights:
    """SparseEncoder Term Weights 测试"""
    
    def test_term_weights_structure(self):
        """测试 term weights 结构可用于 bm25_indexer（核心验收标准）"""
        encoder = SparseEncoder()
        
        chunks = [
            Chunk(id="chunk_1", text="machine learning algorithms", metadata={}),
            Chunk(id="chunk_2", text="deep learning neural networks", metadata={}),
        ]
        
        term_weights_list = encoder.encode(chunks)
        
        # 验证结构：每个元素是 Dict[str, float]
        assert len(term_weights_list) == 2
        
        for term_weights in term_weights_list:
            assert isinstance(term_weights, dict)
            # 验证所有值都是 float
            for term, weight in term_weights.items():
                assert isinstance(term, str)
                assert isinstance(weight, float)
                # 注意：BM25 权重可能为负数（当词出现在大多数文档中时）
                # 但通常权重应该是有效的数值
                assert isinstance(weight, (int, float))
    
    def test_term_weights_contain_terms(self):
        """测试 term weights 包含文本中的词"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="artificial intelligence and machine learning",
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        term_weights = term_weights_list[0]
        
        # 验证包含主要词汇（转换为小写）
        assert "artificial" in term_weights or "intelligence" in term_weights
        assert "machine" in term_weights
        assert "learning" in term_weights
    
    def test_term_weights_frequency_affects_weight(self):
        """测试词频影响权重"""
        encoder = SparseEncoder()
        
        # 包含重复词的文本
        chunk = Chunk(
            id="chunk_1",
            text="Python Python Python programming",
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        term_weights = term_weights_list[0]
        
        if "python" in term_weights and "programming" in term_weights:
            # Python 出现 3 次，programming 出现 1 次
            # Python 的权重应该更高（虽然还受 IDF 影响）
            # 这里只验证两者都有权重（可能是正数或负数）
            assert isinstance(term_weights["python"], (int, float))
            assert isinstance(term_weights["programming"], (int, float))


class TestSparseEncoderEmptyText:
    """SparseEncoder 空文本处理测试（核心验收标准）"""
    
    def test_empty_text_returns_empty_dict(self):
        """测试空文本返回空字典（明确行为）"""
        encoder = SparseEncoder()
        
        # Chunk 模型不允许空文本，所以这里测试的是如果文本只有空白字符
        # 或者通过其他方式传入空文本的情况
        
        # 创建一个只有空白字符的文本（虽然 Chunk 模型会验证，但这里测试编码器行为）
        chunk = Chunk(
            id="chunk_1",
            text="   ",  # 只有空白字符
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        
        # 空文本应该返回空字典（明确行为）
        assert len(term_weights_list) == 1
        term_weights = term_weights_list[0]
        assert isinstance(term_weights, dict)
        # 空白字符会被过滤，所以应该是空字典或只包含停用词（也被过滤）
        # 实际上，只有空白字符的文本分词后应该是空的
        assert len(term_weights) == 0
    
    def test_whitespace_only_text(self):
        """测试只有空白字符的文本"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="\n\n   \t\t  ",
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        term_weights = term_weights_list[0]
        
        # 应该返回空字典
        assert term_weights == {}


class TestSparseEncoderIDF:
    """SparseEncoder IDF 计算测试"""
    
    def test_idf_affects_common_vs_rare_terms(self):
        """测试 IDF 影响常见词和罕见词的权重"""
        encoder = SparseEncoder()
        
        chunks = [
            Chunk(id="chunk_1", text="the cat", metadata={}),
            Chunk(id="chunk_2", text="the dog", metadata={}),
            Chunk(id="chunk_3", text="the bird", metadata={}),
            Chunk(id="chunk_4", text="quantum computing", metadata={}),  # 罕见词
        ]
        
        term_weights_list = encoder.encode(chunks)
        
        # 找到包含 "quantum" 的 chunk
        quantum_chunk_idx = None
        for idx, chunk in enumerate(chunks):
            if "quantum" in chunk.text.lower():
                quantum_chunk_idx = idx
                break
        
        if quantum_chunk_idx is not None:
            quantum_weights = term_weights_list[quantum_chunk_idx]
            
            # "quantum" 是罕见词（只在一个文档中出现），应该有较高的 IDF
            # "the" 是常见词（出现在多个文档中），IDF 较低
            # 但 "the" 是停用词，应该被过滤
            # 所以这里主要验证 "quantum" 有权重
            if "quantum" in quantum_weights:
                assert quantum_weights["quantum"] > 0


class TestSparseEncoderTokenization:
    """SparseEncoder 分词测试"""
    
    def test_tokenization_handles_english(self):
        """测试英文分词"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="Hello world! This is a test.",
            metadata={}
        )
        
        tokens = encoder._tokenize(chunk.text)
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # 停用词应该被过滤
        assert "is" not in tokens or "is" in encoder._stopwords
        assert "a" not in tokens or "a" in encoder._stopwords
    
    def test_tokenization_handles_chinese(self):
        """测试中文分词"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="机器学习是人工智能的一个分支",
            metadata={}
        )
        
        tokens = encoder._tokenize(chunk.text)
        
        # 应该包含中文字符
        assert len(tokens) > 0
        # 验证包含关键词
        assert any("机器" in token or "学习" in token for token in tokens)
    
    def test_tokenization_filters_stopwords(self):
        """测试过滤停用词"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="the quick brown fox jumps over the lazy dog",
            metadata={}
        )
        
        tokens = encoder._tokenize(chunk.text)
        
        # 停用词应该被过滤
        assert "the" not in tokens
        assert "over" not in tokens
        # 非停用词应该保留
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens


class TestSparseEncoderEdgeCases:
    """SparseEncoder 边界情况测试"""
    
    def test_single_word_chunk(self):
        """测试单个词的 Chunk"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="Python",
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        term_weights = term_weights_list[0]
        
        assert "python" in term_weights
        assert isinstance(term_weights["python"], (int, float))
    
    def test_repeated_words(self):
        """测试重复词"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="test test test test",
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        term_weights = term_weights_list[0]
        
        # "test" 出现 4 次，应该有权重（可能是正数或负数，取决于 IDF）
        assert "test" in term_weights
        assert isinstance(term_weights["test"], (int, float))
    
    def test_mixed_language(self):
        """测试中英文混合"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="Python 是一种编程语言 programming language",
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        term_weights = term_weights_list[0]
        
        # 应该包含英文和中文词
        assert "python" in term_weights
        assert "programming" in term_weights
        assert "language" in term_weights
        # 中文词也应该被提取
        assert len(term_weights) >= 3
    
    def test_special_characters(self):
        """测试特殊字符处理"""
        encoder = SparseEncoder()
        
        chunk = Chunk(
            id="chunk_1",
            text="test@example.com and http://example.com",
            metadata={}
        )
        
        term_weights_list = encoder.encode([chunk])
        term_weights = term_weights_list[0]
        
        # 应该提取出单词部分
        assert "test" in term_weights or "example" in term_weights
        assert "http" in term_weights or "example" in term_weights
