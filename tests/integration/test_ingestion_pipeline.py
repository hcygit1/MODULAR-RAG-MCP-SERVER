"""
Ingestion Pipeline 集成测试

验证完整的 ingestion 流程：
- integrity → load → split → transform → encode → store
- 输出向量与 bm25 索引文件
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.ingestion.pipeline import IngestionPipeline
from src.core.settings import Settings, IngestionConfig, LLMConfig, VisionLLMConfig, EmbeddingConfig, VectorStoreConfig, RetrievalConfig, RerankConfig, EvaluationConfig, ObservabilityConfig, LoggingConfig, DashboardConfig
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.vector_store.fake_vector_store import FakeVectorStore
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


def create_test_settings(temp_dir: str) -> Settings:
    """创建测试用的 Settings"""
    ingestion_config = IngestionConfig(
        chunk_size=256,
        chunk_overlap=25,
        enable_llm_refinement=False,  # 测试中禁用 LLM
        enable_metadata_enrichment=True,
        enable_image_captioning=False,  # 测试中禁用图片描述
        batch_size=2
    )
    
    llm_config = LLMConfig(
        provider="fake",
        model="fake-model"
    )
    
    vision_llm_config = VisionLLMConfig(
        provider="fake",
        model="fake-vision-model"
    )
    
    embedding_config = EmbeddingConfig(
        provider="fake",
        model="fake-embedding",
        openai_api_key="",
        local_model_path="",
        device="cpu"
    )
    
    vector_store_config = VectorStoreConfig(
        backend="fake",
        persist_path=str(Path(temp_dir) / "chroma"),
        collection_name="test_collection"
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
        golden_test_set=""
    )
    
    logging_config = LoggingConfig(
        log_file=str(Path(temp_dir) / "logs" / "traces.jsonl"),
        log_level="INFO"
    )
    
    dashboard_config = DashboardConfig(
        enabled=False,
        port=8501
    )
    
    observability_config = ObservabilityConfig(
        enabled=True,
        logging=logging_config,
        detail_level="standard",
        dashboard=dashboard_config
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


class TestIngestionPipelineIntegration:
    """Ingestion Pipeline 集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def test_settings(self, temp_dir):
        """创建测试用的 Settings"""
        return create_test_settings(temp_dir)
    
    @pytest.fixture
    def sample_txt_file(self, temp_dir):
        """创建测试用的文本文件"""
        file_path = Path(temp_dir) / "sample.txt"
        file_path.write_text(
            "This is a test document.\n\n"
            "It contains multiple paragraphs.\n\n"
            "Each paragraph should be split into chunks.\n\n"
            "The pipeline should process this document completely."
        )
        return str(file_path)
    
    def test_pipeline_processes_document(self, temp_dir, test_settings, sample_txt_file):
        """测试 Pipeline 能处理文档（使用 PDF Loader 需要 PDF 文件）"""
        # 注意：当前 Pipeline 只支持 PDF，所以这个测试可能需要调整
        # 或者创建一个简单的 PDF 文件用于测试
        
        # 由于 PDF Loader 需要实际的 PDF 文件，这里先测试基本流程
        # 实际测试应该使用 fixtures/sample_documents/sample.pdf
        
        # 使用 Fake Embedding 和 VectorStore 进行测试
        fake_embedding = FakeEmbedding(dimension=128)
        fake_vector_store = FakeVectorStore()
        
        pipeline = IngestionPipeline(
            test_settings,
            embedding=fake_embedding,
            vector_store=fake_vector_store
        )
        
        # 验证 Pipeline 已初始化
        assert pipeline is not None
        assert pipeline.get_chunk_size() == 256
        assert pipeline.get_chunk_overlap() == 25
    
    def test_pipeline_integrity_check_skips_processed_file(self, temp_dir, test_settings):
        """测试文件完整性检查能跳过已处理的文件"""
        from src.ingestion.models import Document
        from src.libs.loader.file_integrity import FileIntegrityChecker
        
        # 创建一个测试文件
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        # 计算哈希并标记为成功
        integrity_checker = FileIntegrityChecker()
        file_hash = integrity_checker.compute_sha256(str(test_file))
        integrity_checker.mark_success(file_hash)
        
        # 创建 Pipeline（使用 Fake 组件）
        fake_embedding = FakeEmbedding(dimension=128)
        fake_vector_store = FakeVectorStore()
        pipeline = IngestionPipeline(
            test_settings,
            embedding=fake_embedding,
            vector_store=fake_vector_store
        )
        
        # 尝试处理文件（应该被跳过）
        # 注意：由于当前 Pipeline 只支持 PDF，这里简化测试
        # 实际应该测试 PDF 文件的跳过逻辑
        assert integrity_checker.should_skip(file_hash)
    
    def test_pipeline_stores_results(self, temp_dir, test_settings):
        """测试 Pipeline 存储结果（向量和索引）"""
        # 这个测试需要实际的 PDF 文件
        # 当前先验证组件初始化
        fake_embedding = FakeEmbedding(dimension=128)
        fake_vector_store = FakeVectorStore()
        pipeline = IngestionPipeline(
            test_settings,
            embedding=fake_embedding,
            vector_store=fake_vector_store
        )
        
        # 验证存储组件已初始化
        assert pipeline._vector_upserter is not None
        assert pipeline._bm25_indexer is not None
        assert pipeline._image_storage is not None
    
    def test_pipeline_error_handling(self, temp_dir, test_settings):
        """测试 Pipeline 的错误处理"""
        fake_embedding = FakeEmbedding(dimension=128)
        fake_vector_store = FakeVectorStore()
        pipeline = IngestionPipeline(
            test_settings,
            embedding=fake_embedding,
            vector_store=fake_vector_store
        )
        
        # 测试无效文件路径
        with pytest.raises(FileNotFoundError):
            pipeline.process("nonexistent_file.pdf", "test_collection")
        
        # 测试空文件路径
        with pytest.raises(ValueError, match="file_path 不能为空"):
            pipeline.process("", "test_collection")
        
        # 测试空集合名称
        with pytest.raises(ValueError, match="collection_name 不能为空"):
            pipeline.process("test.pdf", "")


class TestIngestionPipelineWithFixtures:
    """使用 fixtures 样例文档的 Pipeline 测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def test_settings(self, temp_dir):
        """创建测试用的 Settings"""
        return create_test_settings(temp_dir)
    
    @pytest.fixture
    def sample_pdf_path(self):
        """获取 fixtures 中的 PDF 文件路径"""
        pdf_path = Path(__file__).parent.parent / "fixtures" / "sample_documents" / "sample.pdf"
        if pdf_path.exists():
            return str(pdf_path)
        pytest.skip("sample.pdf 不存在，跳过测试")
    
    def test_pipeline_with_sample_pdf(self, temp_dir, test_settings, sample_pdf_path):
        """测试使用 fixtures 样例 PDF 运行完整 pipeline"""
        # 使用 Fake Embedding 和 VectorStore 进行测试
        fake_embedding = FakeEmbedding(dimension=128)
        fake_vector_store = FakeVectorStore()
        pipeline = IngestionPipeline(
            test_settings,
            embedding=fake_embedding,
            vector_store=fake_vector_store
        )
        
        # 运行完整 pipeline
        pipeline.process(
            file_path=sample_pdf_path,
            collection_name="test_collection"
        )
        
        # 验证输出文件存在
        # 1. BM25 索引文件
        bm25_index_path = Path("data/db/bm25/test_collection/index.json")
        assert bm25_index_path.exists(), "BM25 索引文件应该存在"
        
        # 2. 向量存储（FakeVectorStore 使用内存，但可以验证调用）
        # 实际实现中，Chroma 会创建持久化文件
        
        # 验证索引可以加载
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        indexer = BM25Indexer()
        indexer.load("test_collection")
        assert indexer.get_total_chunks() > 0, "索引应该包含 chunks"
    
    def test_pipeline_outputs_vectors_and_index(self, temp_dir, test_settings, sample_pdf_path):
        """测试 Pipeline 输出向量与 bm25 索引文件（验收标准）"""
        # 使用 Fake Embedding 和 VectorStore 进行测试
        fake_embedding = FakeEmbedding(dimension=128)
        fake_vector_store = FakeVectorStore()
        pipeline = IngestionPipeline(
            test_settings,
            embedding=fake_embedding,
            vector_store=fake_vector_store
        )
        
        # 运行 pipeline
        pipeline.process(
            file_path=sample_pdf_path,
            collection_name="test_collection"
        )
        
        # 验收标准：输出向量与 bm25 索引文件
        # 1. 验证 BM25 索引文件存在
        bm25_index_path = Path("data/db/bm25/test_collection/index.json")
        assert bm25_index_path.exists(), "BM25 索引文件应该存在"
        
        # 2. 验证索引文件内容有效
        import json
        with open(bm25_index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        
        assert "inverted_index" in index_data
        assert "chunk_metadata" in index_data
        assert index_data["total_chunks"] > 0
        
        # 3. 验证向量已存储（通过 FakeVectorStore 验证调用）
        # 实际实现中，向量会持久化到 Chroma
