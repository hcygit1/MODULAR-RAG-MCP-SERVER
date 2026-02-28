"""
VectorUpserter 幂等性测试

验证 chunk_id 与 BM25 对齐的稳定性：
- 使用 chunk.id 作为 record id
- 同一 chunk 两次 upsert 产生相同 id（覆盖）
- 同一 chunk.id 内容变更时覆盖更新
- content_hash 存入 metadata
"""
import pytest
from unittest.mock import MagicMock

from src.ingestion.models import Chunk
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class MockVectorStore(BaseVectorStore):
    """Mock VectorStore 实现用于测试"""
    
    def __init__(self):
        self._records = {}  # id -> VectorRecord
        self._upsert_calls = []  # 记录所有 upsert 调用
    
    def upsert(self, records, trace=None, collection_name=None):
        """存储记录"""
        self._upsert_calls.append(records)
        for record in records:
            self._records[record.id] = record

    def query(self, vector, top_k, filters=None, trace=None, collection_name=None):
        """查询（测试中不使用）"""
        return []
    
    def get_backend(self) -> str:
        return "mock"
    
    def get_collection_name(self) -> str:
        return "test_collection"
    
    def get_record(self, chunk_id: str) -> VectorRecord:
        """获取存储的记录（用于测试）"""
        return self._records.get(chunk_id)
    
    def get_upsert_calls(self) -> list:
        """获取所有 upsert 调用（用于测试）"""
        return self._upsert_calls


class TestVectorUpserterIdempotency:
    """VectorUpserter 幂等性测试"""
    
    def test_same_chunk_twice_produces_same_id(self):
        """测试同一 chunk 两次 upsert 产生相同 id"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        # 创建相同的 chunk
        chunk = Chunk(
            id="temp_id",
            text="This is a test chunk",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "section_path": "ch1/s1",
                "chunk_index": 0
            }
        )
        
        dense_vector = [0.1] * 128
        sparse_vector = {"test": 1.5, "chunk": 2.0}
        
        # 第一次 upsert
        upserter.upsert_chunks([chunk], [dense_vector], [sparse_vector])
        first_call = mock_store.get_upsert_calls()[0]
        first_chunk_id = first_call[0].id
        
        # 第二次 upsert（相同内容）
        upserter.upsert_chunks([chunk], [dense_vector], [sparse_vector])
        second_call = mock_store.get_upsert_calls()[1]
        second_chunk_id = second_call[0].id
        
        # 验证两次产生的 chunk_id 相同
        assert first_chunk_id == second_chunk_id, \
            f"同一 chunk 两次 upsert 应该产生相同的 id，但得到 {first_chunk_id} 和 {second_chunk_id}"
    
    def test_content_change_overwrites_same_id(self):
        """测试同一 chunk（相同 id）内容变更时，upsert 覆盖而非产生新 id"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        # 模拟 Pipeline 格式：同一逻辑 chunk，内容被 LLM 修改
        chunk1 = Chunk(
            id="doc_abc123_chunk_0",
            text="Original text",
            metadata={"source_path": "/path/to/doc.pdf", "chunk_index": 0}
        )
        chunk2 = Chunk(
            id="doc_abc123_chunk_0",
            text="Modified text",
            metadata={"source_path": "/path/to/doc.pdf", "chunk_index": 0}
        )
        
        dense_vector = [0.1] * 128
        sparse_vector = {"test": 1.5}
        
        upserter.upsert_chunks([chunk1], [dense_vector], [sparse_vector])
        first_id = mock_store.get_upsert_calls()[0][0].id
        
        upserter.upsert_chunks([chunk2], [dense_vector], [sparse_vector])
        second_id = mock_store.get_upsert_calls()[1][0].id
        
        # 同一 chunk.id → 相同 record id（覆盖）
        assert first_id == second_id == "doc_abc123_chunk_0"
        # 最终存储的是新内容
        record = mock_store.get_record(second_id)
        assert record.text == "Modified text"
        assert record.metadata.get("content_hash")
    
    def test_different_chunks_produce_different_id(self):
        """测试不同 chunk（不同 id）产生不同 record id"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        chunk1 = Chunk(
            id="doc_aaa_chunk_0",
            text="Same text",
            metadata={"source_path": "/path/to/doc1.pdf", "chunk_index": 0}
        )
        chunk2 = Chunk(
            id="doc_bbb_chunk_0",
            text="Same text",
            metadata={"source_path": "/path/to/doc2.pdf", "chunk_index": 0}
        )
        
        dense_vector = [0.1] * 128
        sparse_vector = {"test": 1.5}
        
        upserter.upsert_chunks([chunk1], [dense_vector], [sparse_vector])
        first_chunk_id = mock_store.get_upsert_calls()[0][0].id
        
        upserter.upsert_chunks([chunk2], [dense_vector], [sparse_vector])
        second_chunk_id = mock_store.get_upsert_calls()[1][0].id
        
        assert first_chunk_id != second_chunk_id
    
    def test_different_chunk_indices_produce_different_id(self):
        """测试同一文档不同 chunk 产生不同 record id"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        chunk1 = Chunk(
            id="doc_abc_chunk_0",
            text="Same text",
            metadata={"source_path": "/path/to/doc.pdf", "chunk_index": 0}
        )
        chunk2 = Chunk(
            id="doc_abc_chunk_1",
            text="Same text",
            metadata={"source_path": "/path/to/doc.pdf", "chunk_index": 1}
        )
        
        dense_vector = [0.1] * 128
        sparse_vector = {"test": 1.5}
        
        upserter.upsert_chunks([chunk1], [dense_vector], [sparse_vector])
        first_chunk_id = mock_store.get_upsert_calls()[0][0].id
        
        upserter.upsert_chunks([chunk2], [dense_vector], [sparse_vector])
        second_chunk_id = mock_store.get_upsert_calls()[1][0].id
        
        assert first_chunk_id != second_chunk_id


class TestVectorUpserterBasic:
    """VectorUpserter 基础功能测试"""
    
    def test_vector_upserter_initialization(self):
        """测试 VectorUpserter 可以初始化"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        assert upserter is not None
        assert upserter.get_vector_store() == mock_store
    
    def test_vector_upserter_initialization_with_none_store(self):
        """测试 None vector_store 的初始化"""
        with pytest.raises(ValueError, match="vector_store 不能为 None"):
            VectorUpserter(None)
    
    def test_upsert_chunks_basic(self):
        """测试基本的 upsert 功能"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        chunk = Chunk(
            id="temp_id",
            text="Test chunk",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "section_path": "ch1/s1"
            }
        )
        
        dense_vector = [0.1] * 128
        sparse_vector = {"test": 1.5, "chunk": 2.0}
        
        upserter.upsert_chunks([chunk], [dense_vector], [sparse_vector])
        
        # 验证调用了 upsert
        assert len(mock_store.get_upsert_calls()) == 1
        records = mock_store.get_upsert_calls()[0]
        assert len(records) == 1
        
        # 验证记录内容
        record = records[0]
        assert record.text == "Test chunk"
        assert record.vector == dense_vector
        assert record.metadata["sparse_vector"] == sparse_vector
        assert "source_path" in record.metadata
        assert "content_hash" in record.metadata
    
    def test_content_hash_stored_in_metadata(self):
        """测试 content_hash 写入 metadata"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        chunk = Chunk(
            id="doc_xyz_chunk_0",
            text="Unique content here",
            metadata={"source_path": "/path/doc.pdf"}
        )
        
        upserter.upsert_chunks([chunk], [[0.1] * 128], [{}])
        record = mock_store.get_upsert_calls()[0][0]
        
        assert "content_hash" in record.metadata
        assert len(record.metadata["content_hash"]) == 64  # SHA256 hex
    
    def test_upsert_chunks_empty_list(self):
        """测试空 chunks 列表"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        with pytest.raises(ValueError, match="chunks 列表不能为空"):
            upserter.upsert_chunks([], [], [])
    
    def test_upsert_chunks_length_mismatch(self):
        """测试 chunks 和向量数量不匹配"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        chunk = Chunk(
            id="temp_id",
            text="Test",
            metadata={"source_path": "/path/to/doc.pdf"}
        )
        
        # chunks 和 dense_vectors 数量不匹配
        with pytest.raises(ValueError, match="chunks 数量.*与 dense_vectors 数量.*不一致"):
            upserter.upsert_chunks([chunk], [], [{}])
        
        # chunks 和 sparse_vectors 数量不匹配
        with pytest.raises(ValueError, match="chunks 数量.*与 sparse_vectors 数量.*不一致"):
            upserter.upsert_chunks([chunk], [[0.1] * 128], [])
    
    def test_chunk_without_id_raises(self):
        """测试 chunk.id 仅含空白时抛出错误"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        # Chunk 模型允许 id 为空格，但 VectorUpserter 应拒绝
        chunk = Chunk(id="   ", text="Test", metadata={"source_path": "/path/doc.pdf"})
        dense_vector = [0.1] * 128
        sparse_vector = {}
        
        with pytest.raises(ValueError, match="缺少有效 id"):
            upserter.upsert_chunks([chunk], [dense_vector], [sparse_vector])
    
    def test_upsert_chunk_with_minimal_metadata(self):
        """测试 metadata 仅有 chunk_index 时可成功 upsert"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        chunk = Chunk(
            id="doc_xyz_chunk_5",
            text="Test",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "chunk_index": 5
            }
        )
        
        dense_vector = [0.1] * 128
        sparse_vector = {}
        
        upserter.upsert_chunks([chunk], [dense_vector], [sparse_vector])
        
        assert len(mock_store.get_upsert_calls()) == 1
        records = mock_store.get_upsert_calls()[0]
        assert len(records) == 1
        assert records[0].id == "doc_xyz_chunk_5"
    
    def test_upsert_multiple_chunks(self):
        """测试批量 upsert 多个 chunks"""
        mock_store = MockVectorStore()
        upserter = VectorUpserter(mock_store)
        
        chunks = [
            Chunk(
                id=f"temp_id_{i}",
                text=f"Chunk {i}",
                metadata={
                    "source_path": "/path/to/doc.pdf",
                    "section_path": f"ch1/s{i}",
                    "chunk_index": i
                }
            )
            for i in range(3)
        ]
        
        dense_vectors = [[0.1] * 128 for _ in range(3)]
        sparse_vectors = [{"term": 1.0} for _ in range(3)]
        
        upserter.upsert_chunks(chunks, dense_vectors, sparse_vectors)
        
        # 验证所有 chunks 都被处理
        assert len(mock_store.get_upsert_calls()) == 1
        records = mock_store.get_upsert_calls()[0]
        assert len(records) == 3
        
        # 验证每个记录都有唯一的 chunk_id
        chunk_ids = [record.id for record in records]
        assert len(set(chunk_ids)) == 3, "每个 chunk 应该有唯一的 id"
