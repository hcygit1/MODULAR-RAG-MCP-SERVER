"""
Ingestion 数据模型单元测试

验证 Document 和 Chunk 的数据结构、序列化和字段稳定性。
"""
import json
import pytest

from src.ingestion.models import Document, Chunk


class TestDocument:
    """Document 数据模型测试"""
    
    def test_document_creation(self):
        """测试 Document 创建"""
        doc = Document(
            id="doc1",
            text="This is a test document.",
            metadata={"source_path": "test.pdf", "doc_type": "pdf"}
        )
        
        assert doc.id == "doc1"
        assert doc.text == "This is a test document."
        assert doc.metadata == {"source_path": "test.pdf", "doc_type": "pdf"}
    
    def test_document_empty_id(self):
        """测试 Document id 为空时抛出错误"""
        with pytest.raises(ValueError, match="id 不能为空"):
            Document(id="", text="test")
    
    def test_document_empty_text(self):
        """测试 Document text 为空时抛出错误"""
        with pytest.raises(ValueError, match="text 不能为空"):
            Document(id="doc1", text="")
    
    def test_document_default_metadata(self):
        """测试 Document 默认 metadata 为空字典"""
        doc = Document(id="doc1", text="test")
        assert doc.metadata == {}
    
    def test_document_to_dict(self):
        """测试 Document 序列化为字典"""
        doc = Document(
            id="doc1",
            text="Test text",
            metadata={"key": "value"}
        )
        
        result = doc.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == "doc1"
        assert result["text"] == "Test text"
        assert result["metadata"] == {"key": "value"}
    
    def test_document_to_json(self):
        """测试 Document 序列化为 JSON"""
        doc = Document(
            id="doc1",
            text="Test text",
            metadata={"key": "value"}
        )
        
        json_str = doc.to_json()
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["id"] == "doc1"
        assert data["text"] == "Test text"
        assert data["metadata"] == {"key": "value"}
    
    def test_document_from_dict(self):
        """测试从字典创建 Document"""
        data = {
            "id": "doc1",
            "text": "Test text",
            "metadata": {"key": "value"}
        }
        
        doc = Document.from_dict(data)
        
        assert doc.id == "doc1"
        assert doc.text == "Test text"
        assert doc.metadata == {"key": "value"}
    
    def test_document_from_dict_missing_id(self):
        """测试从字典创建 Document 缺少 id 时抛出错误"""
        data = {"text": "test"}
        
        with pytest.raises(ValueError, match="缺少 'id' 字段"):
            Document.from_dict(data)
    
    def test_document_from_dict_missing_text(self):
        """测试从字典创建 Document 缺少 text 时抛出错误"""
        data = {"id": "doc1"}
        
        with pytest.raises(ValueError, match="缺少 'text' 字段"):
            Document.from_dict(data)
    
    def test_document_from_json(self):
        """测试从 JSON 字符串创建 Document"""
        json_str = '{"id": "doc1", "text": "Test text", "metadata": {"key": "value"}}'
        
        doc = Document.from_json(json_str)
        
        assert doc.id == "doc1"
        assert doc.text == "Test text"
        assert doc.metadata == {"key": "value"}
    
    def test_document_equality(self):
        """测试 Document 相等性比较"""
        doc1 = Document(id="doc1", text="test", metadata={"key": "value"})
        doc2 = Document(id="doc1", text="test", metadata={"key": "value"})
        doc3 = Document(id="doc2", text="test", metadata={"key": "value"})
        
        assert doc1 == doc2
        assert doc1 != doc3
    
    def test_document_repr(self):
        """测试 Document 字符串表示"""
        doc = Document(id="doc1", text="test", metadata={"key": "value"})
        
        repr_str = repr(doc)
        
        assert "Document" in repr_str
        assert "doc1" in repr_str
        assert "text_length" in repr_str


class TestChunk:
    """Chunk 数据模型测试"""
    
    def test_chunk_creation(self):
        """测试 Chunk 创建"""
        chunk = Chunk(
            id="chunk1",
            text="This is a test chunk.",
            metadata={"source_path": "test.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=20
        )
        
        assert chunk.id == "chunk1"
        assert chunk.text == "This is a test chunk."
        assert chunk.metadata == {"source_path": "test.pdf", "chunk_index": 0}
        assert chunk.start_offset == 0
        assert chunk.end_offset == 20
    
    def test_chunk_empty_id(self):
        """测试 Chunk id 为空时抛出错误"""
        with pytest.raises(ValueError, match="id 不能为空"):
            Chunk(id="", text="test")
    
    def test_chunk_empty_text(self):
        """测试 Chunk text 为空时抛出错误"""
        with pytest.raises(ValueError, match="text 不能为空"):
            Chunk(id="chunk1", text="")
    
    def test_chunk_default_metadata(self):
        """测试 Chunk 默认 metadata 为空字典"""
        chunk = Chunk(id="chunk1", text="test")
        assert chunk.metadata == {}
    
    def test_chunk_optional_offsets(self):
        """测试 Chunk 的 offset 字段是可选的"""
        chunk = Chunk(id="chunk1", text="test")
        
        assert chunk.start_offset is None
        assert chunk.end_offset is None
    
    def test_chunk_to_dict(self):
        """测试 Chunk 序列化为字典"""
        chunk = Chunk(
            id="chunk1",
            text="Test text",
            metadata={"key": "value"},
            start_offset=0,
            end_offset=9
        )
        
        result = chunk.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == "chunk1"
        assert result["text"] == "Test text"
        assert result["metadata"] == {"key": "value"}
        assert result["start_offset"] == 0
        assert result["end_offset"] == 9
    
    def test_chunk_to_dict_without_offsets(self):
        """测试 Chunk 序列化时不包含 None 的 offset"""
        chunk = Chunk(id="chunk1", text="test")
        
        result = chunk.to_dict()
        
        assert "start_offset" not in result
        assert "end_offset" not in result
    
    def test_chunk_to_json(self):
        """测试 Chunk 序列化为 JSON"""
        chunk = Chunk(
            id="chunk1",
            text="Test text",
            metadata={"key": "value"},
            start_offset=0,
            end_offset=9
        )
        
        json_str = chunk.to_json()
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["id"] == "chunk1"
        assert data["text"] == "Test text"
        assert data["metadata"] == {"key": "value"}
        assert data["start_offset"] == 0
        assert data["end_offset"] == 9
    
    def test_chunk_from_dict(self):
        """测试从字典创建 Chunk"""
        data = {
            "id": "chunk1",
            "text": "Test text",
            "metadata": {"key": "value"},
            "start_offset": 0,
            "end_offset": 9
        }
        
        chunk = Chunk.from_dict(data)
        
        assert chunk.id == "chunk1"
        assert chunk.text == "Test text"
        assert chunk.metadata == {"key": "value"}
        assert chunk.start_offset == 0
        assert chunk.end_offset == 9
    
    def test_chunk_from_dict_without_offsets(self):
        """测试从字典创建 Chunk（不包含 offset）"""
        data = {
            "id": "chunk1",
            "text": "Test text",
            "metadata": {"key": "value"}
        }
        
        chunk = Chunk.from_dict(data)
        
        assert chunk.id == "chunk1"
        assert chunk.text == "Test text"
        assert chunk.start_offset is None
        assert chunk.end_offset is None
    
    def test_chunk_from_dict_missing_id(self):
        """测试从字典创建 Chunk 缺少 id 时抛出错误"""
        data = {"text": "test"}
        
        with pytest.raises(ValueError, match="缺少 'id' 字段"):
            Chunk.from_dict(data)
    
    def test_chunk_from_dict_missing_text(self):
        """测试从字典创建 Chunk 缺少 text 时抛出错误"""
        data = {"id": "chunk1"}
        
        with pytest.raises(ValueError, match="缺少 'text' 字段"):
            Chunk.from_dict(data)
    
    def test_chunk_from_json(self):
        """测试从 JSON 字符串创建 Chunk"""
        json_str = '{"id": "chunk1", "text": "Test text", "metadata": {"key": "value"}, "start_offset": 0, "end_offset": 9}'
        
        chunk = Chunk.from_json(json_str)
        
        assert chunk.id == "chunk1"
        assert chunk.text == "Test text"
        assert chunk.metadata == {"key": "value"}
        assert chunk.start_offset == 0
        assert chunk.end_offset == 9
    
    def test_chunk_equality(self):
        """测试 Chunk 相等性比较"""
        chunk1 = Chunk(id="chunk1", text="test", start_offset=0, end_offset=4)
        chunk2 = Chunk(id="chunk1", text="test", start_offset=0, end_offset=4)
        chunk3 = Chunk(id="chunk2", text="test", start_offset=0, end_offset=4)
        
        assert chunk1 == chunk2
        assert chunk1 != chunk3
    
    def test_chunk_repr(self):
        """测试 Chunk 字符串表示"""
        chunk = Chunk(id="chunk1", text="test", start_offset=0, end_offset=4)
        
        repr_str = repr(chunk)
        
        assert "Chunk" in repr_str
        assert "chunk1" in repr_str
        assert "text_length" in repr_str
        assert "offset" in repr_str


class TestModelSerialization:
    """模型序列化稳定性测试"""
    
    def test_document_serialization_roundtrip(self):
        """测试 Document 序列化往返（dict -> Document -> dict）"""
        original = Document(
            id="doc1",
            text="Test document",
            metadata={"source_path": "test.pdf", "page": 1}
        )
        
        # Document -> dict -> Document
        data = original.to_dict()
        restored = Document.from_dict(data)
        
        assert restored == original
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata
    
    def test_document_json_roundtrip(self):
        """测试 Document JSON 序列化往返"""
        original = Document(
            id="doc1",
            text="Test document",
            metadata={"source_path": "test.pdf"}
        )
        
        # Document -> JSON -> Document
        json_str = original.to_json()
        restored = Document.from_json(json_str)
        
        assert restored == original
    
    def test_chunk_serialization_roundtrip(self):
        """测试 Chunk 序列化往返（dict -> Chunk -> dict）"""
        original = Chunk(
            id="chunk1",
            text="Test chunk",
            metadata={"chunk_index": 0},
            start_offset=0,
            end_offset=10
        )
        
        # Chunk -> dict -> Chunk
        data = original.to_dict()
        restored = Chunk.from_dict(data)
        
        assert restored == original
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata
        assert restored.start_offset == original.start_offset
        assert restored.end_offset == original.end_offset
    
    def test_chunk_json_roundtrip(self):
        """测试 Chunk JSON 序列化往返"""
        original = Chunk(
            id="chunk1",
            text="Test chunk",
            metadata={"chunk_index": 0},
            start_offset=0,
            end_offset=10
        )
        
        # Chunk -> JSON -> Chunk
        json_str = original.to_json()
        restored = Chunk.from_json(json_str)
        
        assert restored == original
    
    def test_document_field_stability(self):
        """测试 Document 字段稳定性（确保字段不会意外变化）"""
        doc = Document(id="doc1", text="test", metadata={"key": "value"})
        data = doc.to_dict()
        
        # 验证必需字段存在
        assert "id" in data
        assert "text" in data
        assert "metadata" in data
        
        # 验证字段类型
        assert isinstance(data["id"], str)
        assert isinstance(data["text"], str)
        assert isinstance(data["metadata"], dict)
    
    def test_chunk_field_stability(self):
        """测试 Chunk 字段稳定性（确保字段不会意外变化）"""
        chunk = Chunk(id="chunk1", text="test", start_offset=0, end_offset=4)
        data = chunk.to_dict()
        
        # 验证必需字段存在
        assert "id" in data
        assert "text" in data
        assert "metadata" in data
        assert "start_offset" in data
        assert "end_offset" in data
        
        # 验证字段类型
        assert isinstance(data["id"], str)
        assert isinstance(data["text"], str)
        assert isinstance(data["metadata"], dict)
        assert isinstance(data["start_offset"], int)
        assert isinstance(data["end_offset"], int)
