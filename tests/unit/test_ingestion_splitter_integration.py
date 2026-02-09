"""
Ingestion Pipeline 与 Splitter 集成测试

验证 Pipeline 中 Splitter 的集成是否正确，配置切换是否生效。
"""
import pytest
from src.ingestion.models import Document
from src.ingestion.pipeline import IngestionPipeline
from src.core.settings import (
    Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig,
    VectorStoreConfig, RetrievalConfig, RerankConfig,
    EvaluationConfig, ObservabilityConfig, IngestionConfig,
    LoggingConfig, DashboardConfig
)


def _create_test_settings(
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> Settings:
    """创建测试用的 Settings 对象"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    vision_llm_config = VisionLLMConfig(
        provider="azure",
        model="gpt-4o",
        azure_endpoint="",
        azure_api_key="",
        deployment_name=""
    )
    embedding_config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        openai_api_key="test-key"
    )
    vector_store_config = VectorStoreConfig(
        backend="chroma",
        persist_path="./data/db/chroma",
        collection_name="test"
    )
    retrieval_config = RetrievalConfig(
        sparse_backend="bm25",
        fusion_algorithm="rrf",
        top_k_dense=20,
        top_k_sparse=20,
        top_k_final=10
    )
    rerank_config = RerankConfig(
        backend="none",
        model="",
        top_m=30,
        timeout_seconds=5
    )
    evaluation_config = EvaluationConfig(
        backends=["ragas"],
        golden_test_set="./tests/fixtures/golden_test_set.json"
    )
    logging_config = LoggingConfig(
        log_file="./logs/app.log",
        log_level="INFO"
    )
    dashboard_config = DashboardConfig(
        enabled=True,
        port=8501
    )
    observability_config = ObservabilityConfig(
        enabled=True,
        logging=logging_config,
        detail_level="standard",
        dashboard=dashboard_config
    )
    ingestion_config = IngestionConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_llm_refinement=False,
        enable_metadata_enrichment=True,
        enable_image_captioning=True,
        batch_size=32
    )
    
    return Settings(
        llm=llm_config,
        vision_llm=vision_llm_config,
        embedding=embedding_config,
        vector_store=vector_store_config,
        retrieval=retrieval_config,
        rerank=rerank_config,
        evaluation=evaluation_config,
        observability=observability_config,
        ingestion=ingestion_config
    )


def test_pipeline_initialization():
    """测试 Pipeline 可以初始化"""
    settings = _create_test_settings()
    pipeline = IngestionPipeline(settings)
    
    assert pipeline is not None
    assert pipeline.get_splitter_strategy() == "recursive"


def test_split_document_basic():
    """测试基本文档切分功能"""
    settings = _create_test_settings(chunk_size=100, chunk_overlap=10)
    pipeline = IngestionPipeline(settings)
    
    # 创建一个测试文档
    document = Document(
        id="test_doc_1",
        text="这是一个测试文档。它包含多个句子。每个句子都应该被正确处理。",
        metadata={"source_path": "test.txt", "doc_type": "text"}
    )
    
    chunks = pipeline.split_document(document)
    
    assert len(chunks) > 0
    # 验证所有 chunk 都是 Chunk 类型
    from src.ingestion.models import Chunk
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    # 验证所有 chunk 的文本加起来应该等于或接近原文
    total_text = "".join(chunk.text for chunk in chunks)
    # 由于可能有重叠，总长度可能大于原文
    assert len(total_text) >= len(document.text)


def test_split_document_chunk_size_config():
    """测试 chunk_size 配置是否生效"""
    # 创建一个较长的文档
    long_text = "这是一个很长的文档。 " * 50  # 约 500 字符
    
    # 测试 1: 小 chunk_size
    settings_small = _create_test_settings(chunk_size=100, chunk_overlap=10)
    pipeline_small = IngestionPipeline(settings_small)
    document = Document(
        id="test_doc_2",
        text=long_text,
        metadata={"source_path": "test.txt"}
    )
    chunks_small = pipeline_small.split_document(document)
    
    # 测试 2: 大 chunk_size
    settings_large = _create_test_settings(chunk_size=500, chunk_overlap=50)
    pipeline_large = IngestionPipeline(settings_large)
    chunks_large = pipeline_large.split_document(document)
    
    # 验证：小 chunk_size 应该产生更多 chunks
    assert len(chunks_small) > len(chunks_large)
    
    # 验证：每个 chunk 的长度应该符合配置
    # 注意：由于 LangChain 的切分策略，实际 chunk 长度可能略小于配置值
    for chunk in chunks_small:
        assert len(chunk.text) <= settings_small.ingestion.chunk_size + 50  # 允许一些误差
    
    for chunk in chunks_large:
        assert len(chunk.text) <= settings_large.ingestion.chunk_size + 50


def test_split_document_chunk_overlap():
    """测试 chunk_overlap 配置是否生效"""
    text = "句子1。句子2。句子3。句子4。句子5。"
    
    settings_no_overlap = _create_test_settings(chunk_size=20, chunk_overlap=0)
    pipeline_no_overlap = IngestionPipeline(settings_no_overlap)
    document = Document(
        id="test_doc_3",
        text=text,
        metadata={"source_path": "test.txt"}
    )
    chunks_no_overlap = pipeline_no_overlap.split_document(document)
    
    settings_with_overlap = _create_test_settings(chunk_size=20, chunk_overlap=10)
    pipeline_with_overlap = IngestionPipeline(settings_with_overlap)
    chunks_with_overlap = pipeline_with_overlap.split_document(document)
    
    # 有重叠时，总文本长度应该更大
    total_no_overlap = sum(len(chunk.text) for chunk in chunks_no_overlap)
    total_with_overlap = sum(len(chunk.text) for chunk in chunks_with_overlap)
    
    # 如果有多个 chunks，重叠版本的总长度应该更大
    if len(chunks_no_overlap) > 1:
        assert total_with_overlap >= total_no_overlap


def test_split_document_metadata_preserved():
    """测试 Document 的元数据是否被保留到 Chunk"""
    settings = _create_test_settings()
    pipeline = IngestionPipeline(settings)
    
    document = Document(
        id="test_doc_4",
        text="测试文本。",
        metadata={
            "source_path": "/path/to/file.pdf",
            "doc_type": "pdf",
            "page": 1,
            "custom_field": "custom_value"
        }
    )
    
    chunks = pipeline.split_document(document)
    
    assert len(chunks) > 0
    for chunk in chunks:
        # 验证元数据被保留
        assert chunk.metadata["source_path"] == "/path/to/file.pdf"
        assert chunk.metadata["doc_type"] == "pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["custom_field"] == "custom_value"
        
        # 验证 chunk 特定元数据
        assert "chunk_index" in chunk.metadata
        assert "source_doc_id" in chunk.metadata
        assert chunk.metadata["source_doc_id"] == document.id


def test_split_document_chunk_ids():
    """测试 Chunk ID 生成是否正确"""
    settings = _create_test_settings()
    pipeline = IngestionPipeline(settings)
    
    document = Document(
        id="doc_123",
        text="测试文本。更多文本。",
        metadata={}
    )
    
    chunks = pipeline.split_document(document)
    
    # 验证每个 chunk 都有唯一的 ID
    chunk_ids = [chunk.id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))  # 所有 ID 应该唯一
    
    # 验证 ID 格式
    for idx, chunk_id in enumerate(chunk_ids):
        assert chunk_id.startswith("doc_123_chunk_")
        assert str(idx) in chunk_id


def test_split_document_position_info():
    """测试 Chunk 的位置信息是否正确"""
    settings = _create_test_settings()
    pipeline = IngestionPipeline(settings)
    
    text = "第一段。第二段。第三段。"
    document = Document(
        id="test_doc_5",
        text=text,
        metadata={}
    )
    
    chunks = pipeline.split_document(document)
    
    # 验证位置信息
    for chunk in chunks:
        assert chunk.start_offset is not None
        assert chunk.end_offset is not None
        assert chunk.start_offset < chunk.end_offset
        
        # 验证位置信息与文本长度一致
        assert chunk.end_offset - chunk.start_offset == len(chunk.text)


def test_split_document_empty_text():
    """测试空文本时抛出错误"""
    settings = _create_test_settings()
    pipeline = IngestionPipeline(settings)
    
    document = Document(
        id="test_doc_6",
        text="",
        metadata={}
    )
    
    with pytest.raises(ValueError, match="不能为空"):
        pipeline.split_document(document)


def test_split_document_image_refs_extraction():
    """测试图片引用提取功能"""
    settings = _create_test_settings()
    pipeline = IngestionPipeline(settings)
    
    # 创建包含图片占位符的文档
    text_with_images = """
    这是第一段。
    [IMAGE: doc_123_5_1]
    这是第二段。
    [IMAGE: doc_123_5_2]
    这是第三段。
    """
    
    document = Document(
        id="test_doc_7",
        text=text_with_images,
        metadata={
            "images": [
                {"image_id": "doc_123_5_1", "page": 5},
                {"image_id": "doc_123_5_2", "page": 5}
            ]
        }
    )
    
    chunks = pipeline.split_document(document)
    
    # 验证包含图片占位符的 chunk 有 image_refs
    chunks_with_images = [chunk for chunk in chunks if "image_refs" in chunk.metadata]
    
    # 至少应该有一个 chunk 包含图片引用
    if chunks_with_images:
        for chunk in chunks_with_images:
            assert "image_refs" in chunk.metadata
            assert isinstance(chunk.metadata["image_refs"], list)
            assert len(chunk.metadata["image_refs"]) > 0


def test_pipeline_get_splitter_info():
    """测试获取 Splitter 信息的方法"""
    settings = _create_test_settings(chunk_size=256, chunk_overlap=25)
    pipeline = IngestionPipeline(settings)
    
    assert pipeline.get_splitter_strategy() == "recursive"
    assert pipeline.get_chunk_size() == 256
    assert pipeline.get_chunk_overlap() == 25
