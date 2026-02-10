"""
图片 Pipeline 单元测试

测试 PDF → Document → Chunks → 图片保存的完整流程，
验证图片数据传递和图片路径获取。
"""
import os
import tempfile
import shutil
from pathlib import Path

import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.models import Document, Chunk

MARKITDOWN_AVAILABLE = True  # Pipeline 不直接依赖 MarkItDown
from src.core.settings import (
    Settings, IngestionConfig, LLMConfig, VisionLLMConfig,
    EmbeddingConfig, VectorStoreConfig, RetrievalConfig,
    RerankConfig, EvaluationConfig, ObservabilityConfig,
    LoggingConfig, DashboardConfig,
)
from src.libs.embedding.fake_embedding import FakeEmbedding
from src.libs.vector_store.fake_vector_store import FakeVectorStore
from src.ingestion.storage.image_storage import ImageStorage


def _create_test_settings(temp_dir: str) -> Settings:
    """创建测试用 Settings"""
    return Settings(
        llm=LLMConfig(provider="fake", model="fake-model"),
        vision_llm=VisionLLMConfig(provider="fake", model="fake-vision"),
        embedding=EmbeddingConfig(
            provider="fake", model="fake-embedding",
            openai_api_key="", local_model_path="", device="cpu"
        ),
        vector_store=VectorStoreConfig(
            backend="fake",
            persist_path=str(Path(temp_dir) / "chroma"),
            collection_name="test_collection"
        ),
        retrieval=RetrievalConfig(
            sparse_backend="bm25", fusion_algorithm="rrf",
            top_k_dense=20, top_k_sparse=20, top_k_final=10
        ),
        rerank=RerankConfig(backend="none", model="", top_m=30, timeout_seconds=5),
        evaluation=EvaluationConfig(backends=["ragas"], golden_test_set=""),
        observability=ObservabilityConfig(
            enabled=True,
            logging=LoggingConfig(
                log_file=str(Path(temp_dir) / "logs" / "traces.jsonl"),
                log_level="INFO"
            ),
            detail_level="standard",
            dashboard=DashboardConfig(enabled=False, port=8501)
        ),
        ingestion=IngestionConfig(
            chunk_size=512,
            chunk_overlap=50,
            enable_llm_refinement=False,
            enable_metadata_enrichment=True,
            enable_image_captioning=False,
            batch_size=2
        ),
    )


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestImageDataPassing:
    """图片数据传递测试"""
    
    def test_split_document_passes_image_data_to_chunks(self):
        """测试 split_document 将图片数据正确传递到 chunk metadata"""
        settings = _create_test_settings(tempfile.mkdtemp())
        pipeline = IngestionPipeline(
            settings,
            embedding=FakeEmbedding(dimension=128),
            vector_store=FakeVectorStore()
        )
        
        doc_id = "doc_test123"
        image_id = f"{doc_id}_page_0_img_0"
        image_bytes = b"\x89PNG\r\n\x1a\n"  # 最小 PNG 头
        
        document = Document(
            id=doc_id,
            text=f"Some text.\n[IMAGE: {image_id}]\nMore text.",
            metadata={
                "image_data": {image_id: image_bytes},
                "images": [{"image_id": image_id, "page": 0, "mime_type": "image/png"}]
            }
        )
        
        chunks = pipeline.split_document(document)
        
        chunks_with_images = [c for c in chunks if "image_refs" in c.metadata]
        assert len(chunks_with_images) >= 1
        for chunk in chunks_with_images:
            assert image_id in chunk.metadata.get("image_refs", [])
            assert "image_data" in chunk.metadata
            assert image_id in chunk.metadata["image_data"]
            assert chunk.metadata["image_data"][image_id] == image_bytes
    
    def test_extract_image_refs(self):
        """测试 _extract_image_refs 正确提取占位符"""
        settings = _create_test_settings(tempfile.mkdtemp())
        pipeline = IngestionPipeline(
            settings,
            embedding=FakeEmbedding(dimension=128),
            vector_store=FakeVectorStore()
        )
        
        text = "Before [IMAGE: id_1] middle [IMAGE: id_2] after"
        refs = pipeline._extract_image_refs(text)
        assert refs == ["id_1", "id_2"]


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestImageSaveE2E:
    """端到端图片保存测试"""
    
    @pytest.fixture
    def temp_dir(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)
    
    def test_save_images_persists_to_storage(self, temp_dir):
        """测试 _save_images 将 chunk 中的图片保存到 ImageStorage"""
        settings = _create_test_settings(temp_dir)
        pipeline = IngestionPipeline(
            settings,
            embedding=FakeEmbedding(dimension=128),
            vector_store=FakeVectorStore()
        )
        
        image_id = "doc_test_page_0_img_0"
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50  # 模拟图片数据
        
        chunks = [
            Chunk(
                id="chunk_0",
                text=f"Text with [IMAGE: {image_id}]",
                metadata={
                    "image_refs": [image_id],
                    "image_data": {image_id: image_bytes},
                    "image_metadata": [{"image_id": image_id, "mime_type": "image/png"}],
                },
                start_offset=0,
                end_offset=100,
            )
        ]
        
        collection_name = "test_image_save_unit"
        pipeline._save_images(chunks, collection_name)
        pipeline._image_storage.save_index(collection_name)
        
        # 验证 ImageStorage 索引
        path = pipeline._image_storage.get_image_path(image_id)
        assert path is not None
        assert Path(path).exists()
        
        # 验证索引文件
        index_file = Path("data/images") / collection_name / "index.json"
        assert index_file.exists()
        import json
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        assert image_id in index_data.get("images", {})


@pytest.mark.skipif(not MARKITDOWN_AVAILABLE, reason="MarkItDown 未安装")
class TestImagePathRetrieval:
    """图片路径获取测试"""
    
    def test_image_storage_get_path_after_save(self, tmp_path):
        """测试保存后可通过 image_id 获取路径"""
        storage = ImageStorage(base_path=str(tmp_path))
        image_id = "doc_test_page_0_img_0"
        image_data = b"\x89PNG\r\n\x1a\n"
        
        path = storage.save_image(
            image_id=image_id,
            image_data=image_data,
            collection_name="test",
            metadata={"mime_type": "image/png"}
        )
        
        assert path is not None
        assert Path(path).exists()
        retrieved = storage.get_image_path(image_id)
        assert retrieved == path
