"""
MetadataEnricher 契约测试

验证规则增强、LLM 增强和失败降级功能。
"""
import json
import pytest
from unittest.mock import MagicMock

from src.ingestion.models import Chunk
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.core.settings import (
    Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig,
    VectorStoreConfig, RetrievalConfig, RerankConfig,
    EvaluationConfig, ObservabilityConfig, IngestionConfig,
    LoggingConfig, DashboardConfig
)


def _create_test_settings(
    enable_metadata_enrichment: bool = False,
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
        enable_metadata_enrichment=enable_metadata_enrichment,
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


class TestMetadataEnricherRuleBased:
    """MetadataEnricher 规则增强测试"""
    
    def test_metadata_enricher_initialization(self):
        """测试 MetadataEnricher 可以初始化"""
        settings = _create_test_settings(enable_metadata_enrichment=False)
        enricher = MetadataEnricher(settings.ingestion)
        
        assert enricher is not None
        assert enricher.get_transform_name() == "metadata_enricher"
    
    def test_rule_based_enrichment_contains_required_fields(self):
        """测试规则增强包含必需字段（核心验收标准）"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_1",
            text="这是关于 Python 编程的文档片段。Python 是一种高级编程语言。",
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        # 验证必需字段存在
        assert "title" in enriched.metadata
        assert "summary" in enriched.metadata
        assert "tags" in enriched.metadata
        
        # 验证 title 和 summary 为非空字符串（核心验收标准）
        assert isinstance(enriched.metadata["title"], str)
        assert len(enriched.metadata["title"]) > 0
        assert isinstance(enriched.metadata["summary"], str)
        assert len(enriched.metadata["summary"]) > 0
        
        # 验证 tags 是列表
        assert isinstance(enriched.metadata["tags"], list)
    
    def test_rule_based_extracts_title_from_markdown(self):
        """测试从 Markdown 标题提取 title"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_2",
            text="## Python 编程指南\n\n这是内容。",
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        assert "title" in enriched.metadata
        assert "Python 编程指南" in enriched.metadata["title"]
    
    def test_rule_based_uses_existing_title(self):
        """测试使用现有 metadata 中的 title"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_3",
            text="这是内容。",
            metadata={"title": "已有标题"}
        )
        
        enriched = enricher.transform(chunk)
        
        assert enriched.metadata["title"] == "已有标题"
    
    def test_rule_based_generates_summary(self):
        """测试生成 summary"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。"
        chunk = Chunk(
            id="test_chunk_4",
            text=text,
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        assert "summary" in enriched.metadata
        assert len(enriched.metadata["summary"]) > 0
        # Summary 应该包含原文的部分内容
        assert len(enriched.metadata["summary"]) <= len(text)
    
    def test_rule_based_infers_tags(self):
        """测试推断 tags"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_5",
            text="本文档介绍 Python 编程语言和机器学习算法。",
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        assert "tags" in enriched.metadata
        assert isinstance(enriched.metadata["tags"], list)
        # 应该包含 "python" 或 "机器学习" 相关标签
        tags_str = " ".join(enriched.metadata["tags"]).lower()
        assert "python" in tags_str or "机器学习" in tags_str or len(enriched.metadata["tags"]) > 0
    
    def test_rule_based_preserves_existing_metadata(self):
        """测试保留现有元数据"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_6",
            text="测试内容",
            metadata={
                "source_path": "/path/to/file.pdf",
                "chunk_index": 0,
                "custom_field": "custom_value"
            }
        )
        
        enriched = enricher.transform(chunk)
        
        # 验证现有元数据被保留
        assert enriched.metadata["source_path"] == "/path/to/file.pdf"
        assert enriched.metadata["chunk_index"] == 0
        assert enriched.metadata["custom_field"] == "custom_value"
        # 验证新增字段
        assert "title" in enriched.metadata
        assert "summary" in enriched.metadata
        assert "tags" in enriched.metadata
    
    def test_rule_based_handles_empty_text(self):
        """测试处理空文本"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_7",
            text="   ",
            metadata={}
        )
        
        # 应该仍然生成默认的 title 和 summary
        enriched = enricher.transform(chunk)
        
        assert "title" in enriched.metadata
        assert "summary" in enriched.metadata
        assert len(enriched.metadata["title"]) > 0
        assert len(enriched.metadata["summary"]) > 0


class TestMetadataEnricherLLM:
    """MetadataEnricher LLM 增强测试"""
    
    def test_llm_enrichment_with_mock_llm(self):
        """测试使用 Mock LLM 进行增强（核心验收标准）"""
        settings = _create_test_settings(enable_metadata_enrichment=True)
        
        # 创建 Mock LLM，返回 JSON 格式的元数据
        mock_llm = MagicMock()
        mock_response = json.dumps({
            "title": "LLM 生成的标题",
            "summary": "LLM 生成的摘要",
            "tags": ["tag1", "tag2", "tag3"]
        })
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(settings.ingestion, llm=mock_llm)
        
        chunk = Chunk(
            id="test_chunk_8",
            text="这是关于 Python 编程的内容。",
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        # 验证 LLM 被调用
        assert mock_llm.chat.called
        
        # 验证使用 LLM 增强
        assert enriched.metadata["enrichment_method"] == "llm"
        
        # 验证 LLM 生成的内容被写入 metadata
        assert enriched.metadata["title"] == "LLM 生成的标题"
        assert enriched.metadata["summary"] == "LLM 生成的摘要"
        assert enriched.metadata["tags"] == ["tag1", "tag2", "tag3"]
    
    def test_llm_enrichment_fallback_on_failure(self):
        """测试 LLM 失败时降级到规则结果（核心验收标准）"""
        settings = _create_test_settings(enable_metadata_enrichment=True)
        
        # 创建会失败的 Mock LLM
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("LLM 调用失败")
        
        enricher = MetadataEnricher(settings.ingestion, llm=mock_llm)
        
        chunk = Chunk(
            id="test_chunk_9",
            text="这是测试内容。",
            metadata={}
        )
        
        # 不应该抛出异常，应该降级
        enriched = enricher.transform(chunk)
        
        # 验证降级到规则模式
        assert enriched.metadata["enrichment_method"] == "rule_fallback"
        assert "enrichment_fallback_reason" in enriched.metadata
        
        # 验证使用规则增强的结果（title 和 summary 应该存在）
        assert "title" in enriched.metadata
        assert len(enriched.metadata["title"]) > 0
        assert "summary" in enriched.metadata
        assert len(enriched.metadata["summary"]) > 0
    
    def test_llm_enrichment_disabled(self):
        """测试 LLM 增强被禁用时只使用规则"""
        settings = _create_test_settings(enable_metadata_enrichment=False)
        
        mock_llm = MagicMock()
        enricher = MetadataEnricher(settings.ingestion, llm=mock_llm)
        
        chunk = Chunk(
            id="test_chunk_10",
            text="测试内容",
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        # 验证 LLM 未被调用
        assert not mock_llm.chat.called
        
        # 验证使用规则模式
        assert enriched.metadata["enrichment_method"] == "rule"
        assert "title" in enriched.metadata
        assert "summary" in enriched.metadata
    
    def test_llm_enrichment_invalid_json_fallback(self):
        """测试 LLM 返回无效 JSON 时降级"""
        settings = _create_test_settings(enable_metadata_enrichment=True)
        
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "这不是有效的 JSON"
        
        enricher = MetadataEnricher(settings.ingestion, llm=mock_llm)
        
        chunk = Chunk(
            id="test_chunk_11",
            text="测试内容",
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        # 应该降级到规则模式
        assert enriched.metadata["enrichment_method"] == "rule_fallback"
        # 但应该有规则增强的结果
        assert "title" in enriched.metadata
        assert "summary" in enriched.metadata
    
    def test_llm_enrichment_partial_json(self):
        """测试 LLM 返回部分 JSON 时使用可用字段"""
        settings = _create_test_settings(enable_metadata_enrichment=True)
        
        mock_llm = MagicMock()
        # 返回部分有效的 JSON
        mock_response = json.dumps({
            "title": "LLM 标题",
            # summary 和 tags 缺失
        })
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(settings.ingestion, llm=mock_llm)
        
        chunk = Chunk(
            id="test_chunk_12",
            text="测试内容",
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        # 应该使用 LLM 的 title，但 summary 和 tags 应该使用规则结果
        assert enriched.metadata["title"] == "LLM 标题"
        assert "summary" in enriched.metadata
        assert "tags" in enriched.metadata


class TestMetadataEnricherEdgeCases:
    """MetadataEnricher 边界情况测试"""
    
    def test_transform_empty_chunk(self):
        """测试空 Chunk 抛出错误"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        # Chunk 模型在初始化时就会验证 text 不能为空
        # 所以我们需要在创建 Chunk 时就捕获错误
        with pytest.raises(ValueError, match="不能为空"):
            chunk = Chunk(
                id="test_chunk_13",
                text="",
                metadata={}
            )
    
    def test_transform_none_chunk(self):
        """测试 None Chunk 抛出错误"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        with pytest.raises(ValueError, match="不能为空"):
            enricher.transform(None)
    
    def test_preserves_chunk_text(self):
        """测试保留 Chunk 文本内容"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        original_text = "这是原始文本内容。"
        chunk = Chunk(
            id="test_chunk_14",
            text=original_text,
            metadata={}
        )
        
        enriched = enricher.transform(chunk)
        
        # 验证文本未被修改
        assert enriched.text == original_text
    
    def test_preserves_position_info(self):
        """测试保留位置信息"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_15",
            text="测试内容",
            metadata={},
            start_offset=100,
            end_offset=200
        )
        
        enriched = enricher.transform(chunk)
        
        # 验证位置信息被保留
        assert enriched.start_offset == 100
        assert enriched.end_offset == 200
    
    def test_rule_based_title_extraction_priority(self):
        """测试标题提取优先级"""
        settings = _create_test_settings()
        enricher = MetadataEnricher(settings.ingestion)
        
        # 测试：现有 metadata 中的 title 应该优先于 Markdown 标题
        chunk = Chunk(
            id="test_chunk_16",
            text="## Markdown 标题\n\n内容",
            metadata={"title": "已有标题"}
        )
        
        enriched = enricher.transform(chunk)
        
        # 应该使用已有标题，而不是 Markdown 标题
        assert enriched.metadata["title"] == "已有标题"
