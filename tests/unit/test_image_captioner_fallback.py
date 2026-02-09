"""
ImageCaptioner 契约测试

验证图片描述生成、降级机制和失败处理功能。
"""
import pytest
from unittest.mock import MagicMock

from src.ingestion.models import Chunk
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.core.settings import (
    Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig,
    VectorStoreConfig, RetrievalConfig, RerankConfig,
    EvaluationConfig, ObservabilityConfig, IngestionConfig,
    LoggingConfig, DashboardConfig
)


def _create_test_settings(
    enable_image_captioning: bool = True,
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
        enable_metadata_enrichment=False,
        enable_image_captioning=enable_image_captioning,
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


class TestImageCaptionerBasic:
    """ImageCaptioner 基础功能测试"""
    
    def test_image_captioner_initialization(self):
        """测试 ImageCaptioner 可以初始化"""
        settings = _create_test_settings()
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        assert captioner is not None
        assert captioner.get_transform_name() == "image_captioner"
    
    def test_no_image_refs_returns_unchanged(self):
        """测试没有 image_refs 时直接返回原 Chunk"""
        settings = _create_test_settings()
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        chunk = Chunk(
            id="test_chunk_1",
            text="这是没有图片的文本内容。",
            metadata={}
        )
        
        result = captioner.transform(chunk)
        
        # 应该返回原始 Chunk（或副本）
        assert result.id == chunk.id
        assert result.text == chunk.text
        assert "image_refs" not in result.metadata or not result.metadata.get("image_refs")
    
    def test_empty_image_refs_returns_unchanged(self):
        """测试空的 image_refs 列表时直接返回原 Chunk"""
        settings = _create_test_settings()
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        chunk = Chunk(
            id="test_chunk_2",
            text="这是文本内容。",
            metadata={"image_refs": []}
        )
        
        result = captioner.transform(chunk)
        
        assert result.id == chunk.id
        assert result.text == chunk.text


class TestImageCaptionerEnabled:
    """ImageCaptioner 启用模式测试"""
    
    def test_vision_llm_captioning_with_mock(self):
        """测试使用 Mock Vision LLM 生成描述（核心验收标准）"""
        settings = _create_test_settings(enable_image_captioning=True)
        
        # 创建 Mock Vision LLM
        mock_vision_llm = MagicMock()
        mock_vision_llm.chat.return_value = "这是一张系统架构图，展示了三层架构设计。"
        
        captioner = ImageCaptioner(
            settings.ingestion,
            settings.vision_llm,
            vision_llm=mock_vision_llm
        )
        
        chunk = Chunk(
            id="test_chunk_3",
            text="这是包含图片的文本。[IMAGE: img_001]",
            metadata={"image_refs": ["img_001"]}
        )
        
        result = captioner.transform(chunk)
        
        # 验证 Vision LLM 被调用
        assert mock_vision_llm.chat.called
        
        # 验证 metadata 包含图片描述
        assert "image_captions" in result.metadata
        assert isinstance(result.metadata["image_captions"], dict)
        assert "img_001" in result.metadata["image_captions"]
        
        # 验证 captioning_method
        assert result.metadata["captioning_method"] == "vision_llm"
        
        # 验证描述被注入到文本中（可选）
        assert "图片描述" in result.text or "image_captions" in result.metadata
    
    def test_multiple_image_refs(self):
        """测试多个图片引用"""
        settings = _create_test_settings(enable_image_captioning=True)
        
        mock_vision_llm = MagicMock()
        mock_vision_llm.chat.side_effect = [
            "第一张图片的描述",
            "第二张图片的描述"
        ]
        
        captioner = ImageCaptioner(
            settings.ingestion,
            settings.vision_llm,
            vision_llm=mock_vision_llm
        )
        
        chunk = Chunk(
            id="test_chunk_4",
            text="文本内容 [IMAGE: img_001] 更多文本 [IMAGE: img_002]",
            metadata={"image_refs": ["img_001", "img_002"]}
        )
        
        result = captioner.transform(chunk)
        
        # 验证所有图片都有描述
        assert "image_captions" in result.metadata
        captions = result.metadata["image_captions"]
        assert "img_001" in captions
        assert "img_002" in captions
        assert len(captions) == 2


class TestImageCaptionerFallback:
    """ImageCaptioner 降级模式测试"""
    
    def test_fallback_when_disabled(self):
        """测试配置禁用时降级（核心验收标准）"""
        settings = _create_test_settings(enable_image_captioning=False)
        
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        chunk = Chunk(
            id="test_chunk_5",
            text="这是包含图片的文本。[IMAGE: img_001]",
            metadata={"image_refs": ["img_001"]}
        )
        
        result = captioner.transform(chunk)
        
        # 验证降级标记
        assert result.metadata["has_unprocessed_images"] is True
        assert result.metadata["captioning_method"] == "fallback"
        assert "captioning_fallback_reason" in result.metadata
        assert "image_captioning_disabled" in result.metadata["captioning_fallback_reason"]
        
        # 验证 image_refs 被保留
        assert "image_refs" in result.metadata
        assert result.metadata["image_refs"] == ["img_001"]
        
        # 验证没有生成描述
        assert "image_captions" not in result.metadata
    
    def test_fallback_when_vision_llm_not_available(self):
        """测试 Vision LLM 不可用时降级（核心验收标准）"""
        settings = _create_test_settings(enable_image_captioning=True)
        
        # 不传入 vision_llm
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        chunk = Chunk(
            id="test_chunk_6",
            text="这是包含图片的文本。[IMAGE: img_001]",
            metadata={"image_refs": ["img_001"]}
        )
        
        result = captioner.transform(chunk)
        
        # 验证降级标记
        assert result.metadata["has_unprocessed_images"] is True
        assert result.metadata["captioning_method"] == "fallback"
        assert "vision_llm_not_available" in result.metadata["captioning_fallback_reason"]
        
        # 验证 image_refs 被保留
        assert "image_refs" in result.metadata
        assert result.metadata["image_refs"] == ["img_001"]
    
    def test_fallback_on_vision_llm_error(self):
        """测试 Vision LLM 调用失败时降级（核心验收标准）"""
        settings = _create_test_settings(enable_image_captioning=True)
        
        # 创建会失败的 Mock Vision LLM
        mock_vision_llm = MagicMock()
        mock_vision_llm.chat.side_effect = RuntimeError("Vision LLM 调用失败")
        
        captioner = ImageCaptioner(
            settings.ingestion,
            settings.vision_llm,
            vision_llm=mock_vision_llm
        )
        
        chunk = Chunk(
            id="test_chunk_7",
            text="这是包含图片的文本。[IMAGE: img_001]",
            metadata={"image_refs": ["img_001"]}
        )
        
        # 不应该抛出异常，应该降级
        result = captioner.transform(chunk)
        
        # 验证降级标记
        assert result.metadata["has_unprocessed_images"] is True
        assert result.metadata["captioning_method"] == "fallback"
        assert "vision_llm_error" in result.metadata["captioning_fallback_reason"]
        
        # 验证 image_refs 被保留
        assert "image_refs" in result.metadata
        assert result.metadata["image_refs"] == ["img_001"]
        
        # 验证文本未被修改（降级时保留原文本）
        assert result.text == chunk.text


class TestImageCaptionerEdgeCases:
    """ImageCaptioner 边界情况测试"""
    
    def test_preserves_chunk_text_on_fallback(self):
        """测试降级时保留原始文本"""
        settings = _create_test_settings(enable_image_captioning=False)
        
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        original_text = "这是原始文本内容。[IMAGE: img_001]"
        chunk = Chunk(
            id="test_chunk_8",
            text=original_text,
            metadata={"image_refs": ["img_001"]}
        )
        
        result = captioner.transform(chunk)
        
        # 验证文本被保留
        assert result.text == original_text
    
    def test_preserves_position_info(self):
        """测试保留位置信息"""
        settings = _create_test_settings()
        
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        chunk = Chunk(
            id="test_chunk_9",
            text="测试内容",
            metadata={"image_refs": ["img_001"]},
            start_offset=100,
            end_offset=200
        )
        
        result = captioner.transform(chunk)
        
        # 验证位置信息被保留
        assert result.start_offset == 100
        assert result.end_offset == 200
    
    def test_preserves_existing_metadata(self):
        """测试保留现有元数据"""
        settings = _create_test_settings(enable_image_captioning=False)
        
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        chunk = Chunk(
            id="test_chunk_10",
            text="测试内容",
            metadata={
                "source_path": "/path/to/file.pdf",
                "chunk_index": 0,
                "image_refs": ["img_001"],
                "custom_field": "custom_value"
            }
        )
        
        result = captioner.transform(chunk)
        
        # 验证现有元数据被保留
        assert result.metadata["source_path"] == "/path/to/file.pdf"
        assert result.metadata["chunk_index"] == 0
        assert result.metadata["custom_field"] == "custom_value"
        # 验证新增了降级标记
        assert "has_unprocessed_images" in result.metadata
    
    def test_transform_empty_chunk(self):
        """测试空 Chunk 抛出错误"""
        settings = _create_test_settings()
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        # Chunk 模型在初始化时就会验证 text 不能为空
        with pytest.raises(ValueError, match="不能为空"):
            chunk = Chunk(
                id="test_chunk_11",
                text="",
                metadata={}
            )
    
    def test_transform_none_chunk(self):
        """测试 None Chunk 抛出错误"""
        settings = _create_test_settings()
        captioner = ImageCaptioner(settings.ingestion, settings.vision_llm)
        
        with pytest.raises(ValueError, match="不能为空"):
            captioner.transform(None)
    
    def test_partial_caption_failure(self):
        """测试部分图片描述生成失败时的处理"""
        settings = _create_test_settings(enable_image_captioning=True)
        
        mock_vision_llm = MagicMock()
        mock_vision_llm.chat.side_effect = [
            "第一张图片的描述",  # 成功
            RuntimeError("第二张图片失败"),  # 失败
            "第三张图片的描述"  # 成功
        ]
        
        captioner = ImageCaptioner(
            settings.ingestion,
            settings.vision_llm,
            vision_llm=mock_vision_llm
        )
        
        chunk = Chunk(
            id="test_chunk_12",
            text="文本 [IMAGE: img_001] [IMAGE: img_002] [IMAGE: img_003]",
            metadata={"image_refs": ["img_001", "img_002", "img_003"]}
        )
        
        # 不应该抛出异常，应该继续处理
        result = captioner.transform(chunk)
        
        # 验证部分成功
        assert "image_captions" in result.metadata
        captions = result.metadata["image_captions"]
        assert "img_001" in captions
        assert "img_003" in captions
        # img_002 应该也有记录（可能是错误信息）
        assert "img_002" in captions
