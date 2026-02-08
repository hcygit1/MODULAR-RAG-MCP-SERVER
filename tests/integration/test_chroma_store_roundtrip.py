"""
Chroma Store 集成测试

测试 Chroma Store 的完整 roundtrip：upsert → query。
标记为 integration 测试，允许跳过。
"""
import pytest
import os
import shutil
import tempfile

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.libs.vector_store.chroma_store import ChromaStore
from src.libs.vector_store.base_vector_store import VectorRecord, QueryResult

# 检查 ChromaDB 是否可用
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB 未安装")
@pytest.mark.integration
def test_factory_creates_chroma_store():
    """测试工厂可以创建 Chroma Store"""
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        vector_store_config = VectorStoreConfig(
            backend="chroma",
            persist_path=temp_dir,
            collection_name="test_collection"
        )
        settings = _create_test_settings(vector_store_config)
        
        store = VectorStoreFactory.create(settings)
        
        assert isinstance(store, ChromaStore)
        assert store.get_backend() == "chroma"
        assert store.get_collection_name() == "test_collection"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB 未安装")
@pytest.mark.integration
def test_chroma_store_upsert_and_query_roundtrip():
    """测试 Chroma Store 的完整 roundtrip：upsert → query"""
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        store = ChromaStore(VectorStoreConfig(
            backend="chroma",
            persist_path=temp_dir,
            collection_name="test_roundtrip"
        ))
        
        # 准备测试数据
        records = [
            VectorRecord(
                id="doc1_chunk1",
                vector=[0.1, 0.2, 0.3] * 512,  # 1536 维
                text="This is the first document chunk.",
                metadata={"source": "doc1.pdf", "page": 1}
            ),
            VectorRecord(
                id="doc1_chunk2",
                vector=[0.4, 0.5, 0.6] * 512,  # 1536 维
                text="This is the second document chunk.",
                metadata={"source": "doc1.pdf", "page": 2}
            ),
            VectorRecord(
                id="doc2_chunk1",
                vector=[0.7, 0.8, 0.9] * 512,  # 1536 维
                text="This is a different document chunk.",
                metadata={"source": "doc2.pdf", "page": 1}
            )
        ]
        
        # 1. Upsert 记录
        store.upsert(records)
        
        # 2. Query 查询（使用第一个记录的向量）
        query_vector = records[0].vector
        results = store.query(query_vector, top_k=2)
        
        # 验证结果
        assert len(results) == 2
        assert results[0].id == "doc1_chunk1"  # 应该是最相似的（相同向量）
        assert results[0].score > 0.9  # 相似度应该很高
        assert results[0].text == "This is the first document chunk."
        assert results[0].metadata["source"] == "doc1.pdf"
        
        # 3. 验证结果按相似度降序排列
        assert results[0].score >= results[1].score
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB 未安装")
@pytest.mark.integration
def test_chroma_store_persistence():
    """测试 Chroma Store 持久化功能"""
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 第一次创建并插入数据
        store1 = ChromaStore(VectorStoreConfig(
            backend="chroma",
            persist_path=temp_dir,
            collection_name="persistent_collection"
        ))
        
        records = [
            VectorRecord(
                id="persistent_1",
                vector=[0.1] * 1536,
                text="Persistent document 1",
                metadata={"test": "persistence"}
            )
        ]
        store1.upsert(records)
        
        # 关闭第一个 store（模拟重启）
        del store1
        
        # 第二次创建（应该能读取之前的数据）
        store2 = ChromaStore(VectorStoreConfig(
            backend="chroma",
            persist_path=temp_dir,
            collection_name="persistent_collection"
        ))
        
        # 查询之前插入的数据
        query_vector = [0.1] * 1536
        results = store2.query(query_vector, top_k=1)
        
        assert len(results) == 1
        assert results[0].id == "persistent_1"
        assert results[0].text == "Persistent document 1"
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB 未安装")
@pytest.mark.integration
def test_chroma_store_idempotent_upsert():
    """测试 Chroma Store 幂等 upsert（相同 ID 更新）"""
    temp_dir = tempfile.mkdtemp()
    try:
        store = ChromaStore(VectorStoreConfig(
            backend="chroma",
            persist_path=temp_dir,
            collection_name="idempotent_test"
        ))
        
        # 第一次插入
        record1 = VectorRecord(
            id="same_id",
            vector=[0.1] * 1536,
            text="Original text",
            metadata={"version": 1}
        )
        store.upsert([record1])
        
        # 第二次 upsert（相同 ID，不同内容）
        record2 = VectorRecord(
            id="same_id",
            vector=[0.2] * 1536,  # 不同的向量
            text="Updated text",  # 不同的文本
            metadata={"version": 2}  # 不同的元数据
        )
        store.upsert([record2])
        
        # 查询应该返回更新后的内容
        query_vector = [0.2] * 1536
        results = store.query(query_vector, top_k=1)
        
        assert len(results) == 1
        assert results[0].id == "same_id"
        assert results[0].text == "Updated text"  # 应该是更新后的文本
        assert results[0].metadata["version"] == 2  # 应该是更新后的元数据
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB 未安装")
@pytest.mark.integration
def test_chroma_store_query_with_filters():
    """测试 Chroma Store 带过滤条件的查询"""
    temp_dir = tempfile.mkdtemp()
    try:
        store = ChromaStore(VectorStoreConfig(
            backend="chroma",
            persist_path=temp_dir,
            collection_name="filter_test"
        ))
        
        # 插入不同 source 的记录
        records = [
            VectorRecord(
                id="doc1_1",
                vector=[0.1] * 1536,
                text="Document 1 chunk 1",
                metadata={"source": "doc1.pdf"}
            ),
            VectorRecord(
                id="doc1_2",
                vector=[0.2] * 1536,
                text="Document 1 chunk 2",
                metadata={"source": "doc1.pdf"}
            ),
            VectorRecord(
                id="doc2_1",
                vector=[0.3] * 1536,
                text="Document 2 chunk 1",
                metadata={"source": "doc2.pdf"}
            )
        ]
        store.upsert(records)
        
        # 查询时使用过滤器（只查询 doc1.pdf）
        query_vector = [0.1] * 1536
        results = store.query(
            query_vector,
            top_k=10,
            filters={"source": "doc1.pdf"}
        )
        
        # 验证所有结果都来自 doc1.pdf
        assert len(results) == 2
        for result in results:
            assert result.metadata["source"] == "doc1.pdf"
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB 未安装")
@pytest.mark.integration
def test_chroma_store_empty_collection_query():
    """测试 Chroma Store 查询空集合"""
    temp_dir = tempfile.mkdtemp()
    try:
        store = ChromaStore(VectorStoreConfig(
            backend="chroma",
            persist_path=temp_dir,
            collection_name="empty_collection"
        ))
        
        # 查询空集合
        query_vector = [0.1] * 1536
        results = store.query(query_vector, top_k=5)
        
        # 应该返回空列表
        assert len(results) == 0
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB 未安装")
@pytest.mark.integration
def test_chroma_store_top_k_limit():
    """测试 Chroma Store top_k 限制"""
    temp_dir = tempfile.mkdtemp()
    try:
        store = ChromaStore(VectorStoreConfig(
            backend="chroma",
            persist_path=temp_dir,
            collection_name="topk_test"
        ))
        
        # 插入多条记录
        records = [
            VectorRecord(
                id=f"record_{i}",
                vector=[float(i) / 100.0] * 1536,
                text=f"Text {i}",
                metadata={"index": i}
            )
            for i in range(10)
        ]
        store.upsert(records)
        
        # 查询，限制 top_k=3
        query_vector = [0.0] * 1536
        results = store.query(query_vector, top_k=3)
        
        # 应该只返回 3 条记录
        assert len(results) == 3
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chroma_store_not_installed():
    """测试 ChromaDB 未安装时的错误处理"""
    if CHROMADB_AVAILABLE:
        pytest.skip("ChromaDB 已安装，跳过此测试")
    
    # 临时模拟 ChromaDB 未安装
    import src.libs.vector_store.chroma_store as cs_module
    original_available = cs_module.CHROMADB_AVAILABLE
    cs_module.CHROMADB_AVAILABLE = False
    
    try:
        with pytest.raises(RuntimeError, match="ChromaDB 未安装"):
            ChromaStore(VectorStoreConfig(
                backend="chroma",
                persist_path="./test",
                collection_name="test"
            ))
    finally:
        cs_module.CHROMADB_AVAILABLE = original_available


def _create_test_settings(vector_store_config: VectorStoreConfig) -> Settings:
    """创建测试用的 Settings 对象"""
    return Settings(
        llm=LLMConfig(provider="fake", model="fake-model"),
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vector_store=vector_store_config,
        retrieval=RetrievalConfig(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10
        ),
        rerank=RerankConfig(backend="none", model="", top_m=30, timeout_seconds=5),
        evaluation=EvaluationConfig(
            backends=["custom"], golden_test_set="./tests/fixtures/golden_test_set.json"
        ),
        observability=ObservabilityConfig(
            enabled=True,
            logging=LoggingConfig(log_file="./logs/traces.jsonl", log_level="INFO"),
            detail_level="standard",
            dashboard=DashboardConfig(enabled=True, port=8501)
        ),
        ingestion=IngestionConfig(
            chunk_size=512,
            chunk_overlap=50,
            enable_llm_refinement=False,
            enable_metadata_enrichment=True,
            enable_image_captioning=True,
            batch_size=32
        )
    )
