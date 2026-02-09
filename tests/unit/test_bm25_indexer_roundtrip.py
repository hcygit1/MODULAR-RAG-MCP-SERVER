"""
BM25Indexer Roundtrip 测试

验证索引构建、保存、加载和查询的完整流程：
- build 后能 save
- load 后能 query
- 对同一语料查询返回稳定 top ids
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.ingestion.models import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.embedding.sparse_encoder import SparseEncoder


class TestBM25IndexerRoundtrip:
    """BM25Indexer Roundtrip 测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_chunks(self):
        """创建测试用的 chunks"""
        return [
            Chunk(
                id="chunk_1",
                text="Python is a programming language",
                metadata={"source_path": "/path/to/doc1.pdf", "chunk_index": 0}
            ),
            Chunk(
                id="chunk_2",
                text="Machine learning is a subset of artificial intelligence",
                metadata={"source_path": "/path/to/doc1.pdf", "chunk_index": 1}
            ),
            Chunk(
                id="chunk_3",
                text="Python is widely used for data science and machine learning",
                metadata={"source_path": "/path/to/doc2.pdf", "chunk_index": 0}
            ),
        ]
    
    @pytest.fixture
    def sample_sparse_vectors(self, sample_chunks):
        """生成测试用的稀疏向量"""
        encoder = SparseEncoder()
        return encoder.encode(sample_chunks)
    
    def test_build_and_save(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试构建和保存索引"""
        indexer = BM25Indexer(base_path=temp_dir)
        
        # 构建索引
        indexer.build(sample_chunks, sample_sparse_vectors, collection_name="test_collection")
        
        # 验证索引已构建
        assert indexer.get_total_chunks() == 3
        assert indexer.get_total_terms() > 0
        
        # 保存索引
        indexer.save()
        
        # 验证文件已创建
        index_path = indexer.get_index_path()
        assert index_path is not None
        assert index_path.exists()
    
    def test_load_and_query(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试加载索引和查询"""
        # 先构建并保存
        indexer1 = BM25Indexer(base_path=temp_dir)
        indexer1.build(sample_chunks, sample_sparse_vectors, collection_name="test_collection")
        indexer1.save()
        
        # 创建新的 indexer 并加载
        indexer2 = BM25Indexer(base_path=temp_dir)
        indexer2.load("test_collection")
        
        # 验证索引已加载
        assert indexer2.get_total_chunks() == 3
        assert indexer2.get_total_terms() > 0
        
        # 查询
        results = indexer2.query(["python"], top_k=3)
        
        # 验证查询结果
        assert len(results) > 0
        assert len(results) <= 3
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)
    
    def test_query_stable_top_ids(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试对同一语料查询返回稳定 top ids"""
        # 构建并保存索引
        indexer = BM25Indexer(base_path=temp_dir)
        indexer.build(sample_chunks, sample_sparse_vectors, collection_name="test_collection")
        indexer.save()
        
        # 第一次查询
        results1 = indexer.query(["python", "machine"], top_k=3)
        chunk_ids1 = [chunk_id for chunk_id, _ in results1]
        scores1 = [score for _, score in results1]
        
        # 第二次查询（应该返回相同结果）
        results2 = indexer.query(["python", "machine"], top_k=3)
        chunk_ids2 = [chunk_id for chunk_id, _ in results2]
        scores2 = [score for _, score in results2]
        
        # 验证结果稳定
        assert chunk_ids1 == chunk_ids2, "查询结果应该稳定"
        assert scores1 == scores2, "分数应该稳定"
    
    def test_roundtrip_stable_results(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试完整的 roundtrip：build -> save -> load -> query"""
        # 构建并保存
        indexer1 = BM25Indexer(base_path=temp_dir)
        indexer1.build(sample_chunks, sample_sparse_vectors, collection_name="test_collection")
        indexer1.save()
        
        # 第一次查询（内存中）
        results1 = indexer1.query(["python"], top_k=2)
        chunk_ids1 = [chunk_id for chunk_id, _ in results1]
        
        # 加载并查询
        indexer2 = BM25Indexer(base_path=temp_dir)
        indexer2.load("test_collection")
        results2 = indexer2.query(["python"], top_k=2)
        chunk_ids2 = [chunk_id for chunk_id, _ in results2]
        
        # 验证结果一致
        assert chunk_ids1 == chunk_ids2, "roundtrip 后查询结果应该一致"
    
    def test_query_multiple_terms(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试多词查询"""
        indexer = BM25Indexer(base_path=temp_dir)
        indexer.build(sample_chunks, sample_sparse_vectors, collection_name="test_collection")
        
        # 查询多个词
        results = indexer.query(["python", "learning"], top_k=3)
        
        # 验证结果按分数降序排列
        assert len(results) > 0
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "结果应该按分数降序排列"
    
    def test_query_empty_terms(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试空查询词列表"""
        indexer = BM25Indexer(base_path=temp_dir)
        indexer.build(sample_chunks, sample_sparse_vectors, collection_name="test_collection")
        
        results = indexer.query([], top_k=3)
        assert len(results) == 0
    
    def test_query_nonexistent_terms(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试查询不存在的词"""
        indexer = BM25Indexer(base_path=temp_dir)
        indexer.build(sample_chunks, sample_sparse_vectors, collection_name="test_collection")
        
        results = indexer.query(["nonexistentword12345"], top_k=3)
        assert len(results) == 0


class TestBM25IndexerBasic:
    """BM25Indexer 基础功能测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_chunks(self):
        """创建测试用的 chunks"""
        return [
            Chunk(
                id="chunk_1",
                text="Python is a programming language",
                metadata={"source_path": "/path/to/doc1.pdf", "chunk_index": 0}
            ),
        ]
    
    @pytest.fixture
    def sample_sparse_vectors(self, sample_chunks):
        """生成测试用的稀疏向量"""
        encoder = SparseEncoder()
        return encoder.encode(sample_chunks)
    
    def test_bm25_indexer_initialization(self):
        """测试 BM25Indexer 可以初始化"""
        indexer = BM25Indexer()
        assert indexer is not None
        assert indexer.get_collection_name() is None
    
    def test_bm25_indexer_initialization_with_custom_path(self):
        """测试使用自定义路径初始化"""
        temp_dir = tempfile.mkdtemp()
        try:
            indexer = BM25Indexer(base_path=temp_dir)
            assert indexer._base_path == Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_build_empty_chunks(self):
        """测试构建空 chunks 列表"""
        indexer = BM25Indexer()
        with pytest.raises(ValueError, match="chunks 列表不能为空"):
            indexer.build([], [], "test_collection")
    
    def test_build_length_mismatch(self):
        """测试 chunks 和 sparse_vectors 数量不匹配"""
        indexer = BM25Indexer()
        chunks = [Chunk(id="c1", text="test", metadata={})]
        sparse_vectors = [{}, {}]  # 数量不匹配
        
        with pytest.raises(ValueError, match="数量.*不一致"):
            indexer.build(chunks, sparse_vectors, "test_collection")
    
    def test_build_empty_collection_name(self):
        """测试空集合名称"""
        indexer = BM25Indexer()
        chunks = [Chunk(id="c1", text="test", metadata={})]
        sparse_vectors = [{}]
        
        with pytest.raises(ValueError, match="collection_name 不能为空"):
            indexer.build(chunks, sparse_vectors, "")
    
    def test_save_without_build(self):
        """测试未构建就保存"""
        indexer = BM25Indexer()
        with pytest.raises(RuntimeError, match="索引未构建"):
            indexer.save()
    
    def test_load_nonexistent_collection(self, temp_dir):
        """测试加载不存在的集合"""
        indexer = BM25Indexer(base_path=temp_dir)
        with pytest.raises(FileNotFoundError):
            indexer.load("nonexistent_collection")
    
    def test_query_without_load(self):
        """测试未加载就查询"""
        indexer = BM25Indexer()
        with pytest.raises(RuntimeError, match="索引未加载"):
            indexer.query(["test"])
    
    def test_query_invalid_top_k(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试无效的 top_k"""
        indexer = BM25Indexer(base_path=temp_dir)
        indexer.build(sample_chunks, sample_sparse_vectors, "test_collection")
        
        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            indexer.query(["test"], top_k=0)
        
        with pytest.raises(ValueError, match="top_k 必须大于 0"):
            indexer.query(["test"], top_k=-1)
    
    def test_get_index_path_without_collection(self):
        """测试未设置集合名称时获取索引路径"""
        indexer = BM25Indexer()
        assert indexer.get_index_path() is None
    
    def test_get_index_path_with_collection(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试获取索引路径"""
        indexer = BM25Indexer(base_path=temp_dir)
        indexer.build(sample_chunks, sample_sparse_vectors, "test_collection")
        
        index_path = indexer.get_index_path()
        assert index_path is not None
        assert index_path.name == "index.json"
        assert "test_collection" in str(index_path)
    
    def test_get_index_path_without_collection(self):
        """测试未设置集合名称时获取索引路径"""
        indexer = BM25Indexer()
        assert indexer.get_index_path() is None
    
    def test_get_index_path_with_collection(self, temp_dir, sample_chunks, sample_sparse_vectors):
        """测试获取索引路径"""
        indexer = BM25Indexer(base_path=temp_dir)
        indexer.build(sample_chunks, sample_sparse_vectors, "test_collection")
        
        index_path = indexer.get_index_path()
        assert index_path is not None
        assert index_path.name == "index.json"
        assert "test_collection" in str(index_path)
