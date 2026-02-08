"""
VectorStore 契约测试

测试 VectorStore 接口的输入输出 shape 和契约约束。
"""
import pytest

from src.core.settings import Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig
from src.core.settings import VectorStoreConfig, RetrievalConfig, RerankConfig
from src.core.settings import EvaluationConfig, ObservabilityConfig, IngestionConfig
from src.core.settings import LoggingConfig, DashboardConfig
from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult
)
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.libs.vector_store.fake_vector_store import FakeVectorStore


def test_factory_creates_fake_vector_store():
    """测试工厂可以创建 Fake VectorStore"""
    fake_store = VectorStoreFactory.create_fake(
        backend="test", collection_name="test_collection"
    )
    
    assert isinstance(fake_store, BaseVectorStore)
    assert fake_store.get_backend() == "test"
    assert fake_store.get_collection_name() == "test_collection"


def test_vector_record_structure():
    """测试 VectorRecord 数据结构"""
    record = VectorRecord(
        id="test_id",
        vector=[0.1, 0.2, 0.3],
        text="Test text",
        metadata={"source": "test.pdf"}
    )
    
    assert record.id == "test_id"
    assert record.vector == [0.1, 0.2, 0.3]
    assert record.text == "Test text"
    assert record.metadata == {"source": "test.pdf"}
    
    # 测试 to_dict 方法
    record_dict = record.to_dict()
    assert record_dict["id"] == "test_id"
    assert record_dict["vector"] == [0.1, 0.2, 0.3]
    assert record_dict["text"] == "Test text"
    assert record_dict["metadata"] == {"source": "test.pdf"}


def test_query_result_structure():
    """测试 QueryResult 数据结构"""
    result = QueryResult(
        id="test_id",
        score=0.95,
        text="Test text",
        metadata={"source": "test.pdf"}
    )
    
    assert result.id == "test_id"
    assert result.score == 0.95
    assert result.text == "Test text"
    assert result.metadata == {"source": "test.pdf"}
    
    # 测试 to_dict 方法
    result_dict = result.to_dict()
    assert result_dict["id"] == "test_id"
    assert result_dict["score"] == 0.95
    assert result_dict["text"] == "Test text"
    assert result_dict["metadata"] == {"source": "test.pdf"}


def test_fake_vector_store_upsert():
    """测试 Fake VectorStore 的 upsert 方法"""
    store = FakeVectorStore()
    
    records = [
        VectorRecord(
            id="id1",
            vector=[1.0, 0.0, 0.0],
            text="Text 1",
            metadata={"source": "doc1.pdf"}
        ),
        VectorRecord(
            id="id2",
            vector=[0.0, 1.0, 0.0],
            text="Text 2",
            metadata={"source": "doc2.pdf"}
        )
    ]
    
    store.upsert(records)
    
    # 验证记录已存储
    assert len(store._records) == 2
    assert "id1" in store._records
    assert "id2" in store._records


def test_fake_vector_store_upsert_idempotent():
    """测试 Fake VectorStore 的幂等性（相同 ID 覆盖）"""
    store = FakeVectorStore()
    
    record1 = VectorRecord(
        id="id1",
        vector=[1.0, 0.0],
        text="Original text",
        metadata={"version": 1}
    )
    
    record2 = VectorRecord(
        id="id1",
        vector=[0.0, 1.0],
        text="Updated text",
        metadata={"version": 2}
    )
    
    store.upsert([record1])
    assert store._records["id1"].text == "Original text"
    
    store.upsert([record2])
    assert store._records["id1"].text == "Updated text"
    assert len(store._records) == 1  # 仍然是 1 条记录


def test_fake_vector_store_upsert_validation():
    """测试 Fake VectorStore upsert 的输入验证"""
    store = FakeVectorStore()
    
    # 空记录列表
    with pytest.raises(ValueError, match="记录列表不能为空"):
        store.upsert([])
    
    # 无效的记录类型
    with pytest.raises(ValueError, match="记录必须是 VectorRecord 类型"):
        store.upsert([{"id": "test"}])
    
    # 缺少 ID
    with pytest.raises(ValueError, match="记录 ID 不能为空"):
        store.upsert([VectorRecord(id="", vector=[1.0], text="test")])
    
    # 缺少向量
    with pytest.raises(ValueError, match="记录向量不能为空"):
        store.upsert([VectorRecord(id="id1", vector=[], text="test")])
    
    # 缺少文本
    with pytest.raises(ValueError, match="记录文本不能为空"):
        store.upsert([VectorRecord(id="id1", vector=[1.0], text="")])


def test_fake_vector_store_query():
    """测试 Fake VectorStore 的 query 方法"""
    store = FakeVectorStore()
    
    # 插入测试数据
    records = [
        VectorRecord(
            id="id1",
            vector=[1.0, 0.0, 0.0],
            text="Text about apples",
            metadata={"source": "doc1.pdf"}
        ),
        VectorRecord(
            id="id2",
            vector=[0.0, 1.0, 0.0],
            text="Text about bananas",
            metadata={"source": "doc2.pdf"}
        ),
        VectorRecord(
            id="id3",
            vector=[0.0, 0.0, 1.0],
            text="Text about oranges",
            metadata={"source": "doc3.pdf"}
        )
    ]
    store.upsert(records)
    
    # 查询（使用与 id1 相同的向量）
    query_vector = [1.0, 0.0, 0.0]
    results = store.query(query_vector, top_k=2)
    
    assert len(results) == 2
    assert isinstance(results[0], QueryResult)
    assert results[0].id == "id1"  # 应该最相似
    assert results[0].score == 1.0  # 完全匹配


def test_fake_vector_store_query_with_filters():
    """测试 Fake VectorStore 的带过滤条件的查询"""
    store = FakeVectorStore()
    
    records = [
        VectorRecord(
            id="id1",
            vector=[1.0, 0.0],
            text="Text 1",
            metadata={"source": "doc1.pdf", "category": "A"}
        ),
        VectorRecord(
            id="id2",
            vector=[0.0, 1.0],
            text="Text 2",
            metadata={"source": "doc2.pdf", "category": "B"}
        ),
        VectorRecord(
            id="id3",
            vector=[0.5, 0.5],
            text="Text 3",
            metadata={"source": "doc1.pdf", "category": "A"}
        )
    ]
    store.upsert(records)
    
    # 查询并过滤
    query_vector = [1.0, 0.0]
    results = store.query(
        query_vector,
        top_k=10,
        filters={"source": "doc1.pdf"}
    )
    
    # 应该只返回 doc1.pdf 的记录
    assert len(results) == 2
    assert all(r.metadata["source"] == "doc1.pdf" for r in results)


def test_fake_vector_store_query_validation():
    """测试 Fake VectorStore query 的输入验证"""
    store = FakeVectorStore()
    
    # 空向量
    with pytest.raises(ValueError, match="查询向量不能为空"):
        store.query([], top_k=5)
    
    # top_k <= 0
    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        store.query([1.0, 0.0], top_k=0)
    
    with pytest.raises(ValueError, match="top_k 必须大于 0"):
        store.query([1.0, 0.0], top_k=-1)


def test_fake_vector_store_query_top_k_limit():
    """测试 Fake VectorStore query 的 top_k 限制"""
    store = FakeVectorStore()
    
    # 插入多条记录
    records = [
        VectorRecord(
            id=f"id{i}",
            vector=[float(i), 0.0],
            text=f"Text {i}",
            metadata={}
        )
        for i in range(10)
    ]
    store.upsert(records)
    
    # 查询 top_k=3
    query_vector = [1.0, 0.0]
    results = store.query(query_vector, top_k=3)
    
    assert len(results) <= 3
    assert all(isinstance(r, QueryResult) for r in results)


def test_base_vector_store_interface():
    """测试 BaseVectorStore 接口定义"""
    # 验证 BaseVectorStore 是抽象类
    with pytest.raises(TypeError):
        BaseVectorStore()  # 不能直接实例化抽象类
    
    # 验证 FakeVectorStore 实现了所有抽象方法
    fake_store = FakeVectorStore()
    assert hasattr(fake_store, "upsert")
    assert hasattr(fake_store, "query")
    assert hasattr(fake_store, "get_backend")
    assert hasattr(fake_store, "get_collection_name")
    
    # 验证方法可以调用
    fake_store.upsert([VectorRecord(id="test", vector=[1.0], text="test")])
    results = fake_store.query([1.0], top_k=1)
    assert isinstance(results, list)
    assert fake_store.get_backend() == "fake"
    assert fake_store.get_collection_name() == "test_collection"


def test_fake_vector_store_trace_parameter():
    """测试 Fake VectorStore 接受 trace 参数（虽然不使用）"""
    store = FakeVectorStore()
    
    record = VectorRecord(id="test", vector=[1.0], text="test")
    
    # trace 参数是可选的，应该可以传入 None
    store.upsert([record], trace=None)
    
    # 或者不传入（使用默认值）
    results1 = store.query([1.0], top_k=1, trace=None)
    results2 = store.query([1.0], top_k=1)
    
    # 结果应该相同
    assert len(results1) == len(results2)


def test_factory_unsupported_backend():
    """测试不支持的 backend 抛出 ValueError"""
    vector_store_config = VectorStoreConfig(
        backend="unsupported",
        persist_path="./data/db/chroma",
        collection_name="test"
    )
    
    settings = _create_test_settings(vector_store_config)
    
    with pytest.raises(ValueError, match="不支持的 VectorStore backend"):
        VectorStoreFactory.create(settings)


def test_factory_not_implemented_backends():
    """测试尚未实现的 backend 抛出 NotImplementedError"""
    # 测试 chroma
    vector_store_config_chroma = VectorStoreConfig(
        backend="chroma",
        persist_path="./data/db/chroma",
        collection_name="test"
    )
    settings_chroma = _create_test_settings(vector_store_config_chroma)
    
    with pytest.raises(NotImplementedError, match="Chroma VectorStore 实现将在 B7.6"):
        VectorStoreFactory.create(settings_chroma)
    
    # 测试 qdrant
    vector_store_config_qdrant = VectorStoreConfig(
        backend="qdrant",
        persist_path="./data/db/qdrant",
        collection_name="test"
    )
    settings_qdrant = _create_test_settings(vector_store_config_qdrant)
    
    with pytest.raises(NotImplementedError, match="Qdrant VectorStore 实现尚未完成"):
        VectorStoreFactory.create(settings_qdrant)
    
    # 测试 pinecone
    vector_store_config_pinecone = VectorStoreConfig(
        backend="pinecone",
        persist_path="",
        collection_name="test"
    )
    settings_pinecone = _create_test_settings(vector_store_config_pinecone)
    
    with pytest.raises(NotImplementedError, match="Pinecone VectorStore 实现尚未完成"):
        VectorStoreFactory.create(settings_pinecone)


def test_factory_backend_case_insensitive():
    """测试 backend 名称大小写不敏感"""
    vector_store_config_upper = VectorStoreConfig(
        backend="CHROMA",
        persist_path="./data/db/chroma",
        collection_name="test"
    )
    settings_upper = _create_test_settings(vector_store_config_upper)
    
    # 应该能识别（虽然会抛出 NotImplementedError，但错误信息应该正确）
    with pytest.raises(NotImplementedError, match="Chroma VectorStore"):
        VectorStoreFactory.create(settings_upper)


def _create_test_settings(vector_store_config: VectorStoreConfig) -> Settings:
    """创建测试用的 Settings 对象"""
    return Settings(
        llm=LLMConfig(provider="azure", model="gpt-4o"),
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
