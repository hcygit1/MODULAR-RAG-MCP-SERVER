"""
Qdrant VectorStore 单元测试

测试 QdrantStore 的 upsert、query 及复杂 metadata（list、dict）支持。
"""
import tempfile
import shutil
from pathlib import Path

import pytest

from src.libs.vector_store.qdrant_store import QdrantStore, QDRANT_AVAILABLE
from src.libs.vector_store.base_vector_store import VectorRecord, QueryResult
from src.core.settings import VectorStoreConfig


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client 未安装")
class TestQdrantStoreBasic:
    """QdrantStore 基础功能测试"""
    
    @pytest.fixture
    def temp_dir(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)
    
    @pytest.fixture
    def store(self, temp_dir):
        """使用 embedded 模式创建 QdrantStore，无需额外服务"""
        config = VectorStoreConfig(
            backend="qdrant",
            persist_path=str(Path(temp_dir) / "qdrant_data"),
            collection_name="test_unit",
            qdrant_url="",  # 空字符串启用 embedded 模式
        )
        return QdrantStore(config)
    
    def test_upsert_and_query(self, store):
        """测试 upsert 与 query"""
        records = [
            VectorRecord(
                id="doc1_chunk_0",
                vector=[0.1] * 4,
                text="Hello world",
                metadata={"source_path": "/a.pdf", "chunk_index": 0},
            ),
            VectorRecord(
                id="doc1_chunk_1",
                vector=[0.2] * 4,
                text="Second chunk",
                metadata={"source_path": "/a.pdf", "chunk_index": 1},
            ),
        ]
        store.upsert(records)
        
        results = store.query(vector=[0.15] * 4, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)
        assert results[0].score >= results[1].score
    
    def test_metadata_with_list(self, store):
        """测试 list 类型 metadata（tags、image_refs）"""
        records = [
            VectorRecord(
                id="chunk_with_images",
                vector=[0.5] * 4,
                text="Content with images",
                metadata={
                    "source_path": "/doc.pdf",
                    "tags": ["架构", "api", "设计"],
                    "image_refs": ["doc_abc_page_0_img_0", "doc_abc_page_1_img_0"],
                },
            ),
        ]
        store.upsert(records)
        
        results = store.query(vector=[0.5] * 4, top_k=1)
        assert len(results) == 1
        meta = results[0].metadata
        assert "tags" in meta
        assert meta["tags"] == ["架构", "api", "设计"]
        assert "image_refs" in meta
        assert meta["image_refs"] == ["doc_abc_page_0_img_0", "doc_abc_page_1_img_0"]
    
    def test_metadata_with_dict(self, store):
        """测试 dict 类型 metadata"""
        records = [
            VectorRecord(
                id="chunk_with_dict",
                vector=[0.3] * 4,
                text="Content",
                metadata={
                    "source_path": "/x.pdf",
                    "image_captions": {
                        "img_1": "图片描述1",
                        "img_2": "图片描述2",
                    },
                },
            ),
        ]
        store.upsert(records)
        
        results = store.query(vector=[0.3] * 4, top_k=1)
        assert len(results) == 1
        meta = results[0].metadata
        assert "image_captions" in meta
        assert meta["image_captions"] == {"img_1": "图片描述1", "img_2": "图片描述2"}
    
    def test_get_backend_and_collection(self, store):
        """测试 get_backend 和 get_collection_name"""
        assert store.get_backend() == "qdrant"
        assert store.get_collection_name() == "test_unit"
