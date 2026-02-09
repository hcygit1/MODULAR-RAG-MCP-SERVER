"""
ChunkRefiner 单元测试

验证规则去噪、LLM 重写和失败降级功能。
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ingestion.models import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.core.settings import (
    Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig,
    VectorStoreConfig, RetrievalConfig, RerankConfig,
    EvaluationConfig, ObservabilityConfig, IngestionConfig,
    LoggingConfig, DashboardConfig
)


def _create_test_settings(
    enable_llm_refinement: bool = False,
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
        enable_llm_refinement=enable_llm_refinement,
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


class TestBaseTransform:
    """BaseTransform 抽象基类测试"""
    
    def test_base_transform_is_abstract(self):
        """测试 BaseTransform 是抽象类，不能直接实例化"""
        with pytest.raises(TypeError):
            BaseTransform()
    
    def test_base_transform_has_transform_method(self):
        """测试 BaseTransform 定义了 transform 方法"""
        assert hasattr(BaseTransform, 'transform')
        assert callable(getattr(BaseTransform, 'transform'))


class TestChunkRefinerRuleBased:
    """ChunkRefiner 规则去噪测试"""
    
    def test_chunk_refiner_initialization(self):
        """测试 ChunkRefiner 可以初始化"""
        settings = _create_test_settings(enable_llm_refinement=False)
        refiner = ChunkRefiner(settings.ingestion)
        
        assert isinstance(refiner, ChunkRefiner)
        assert isinstance(refiner, BaseTransform)
        assert refiner.get_transform_name() == "chunk_refiner"
    
    def test_rule_based_removes_page_numbers(self):
        """测试去除页码"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_1",
            text="第 1 页\n\n这是内容。\n\n第 2 页\n\n更多内容。",
            metadata={}
        )
        
        refined = refiner.transform(chunk)
        
        # 验证页码被去除
        assert "第 1 页" not in refined.text
        assert "第 2 页" not in refined.text
        # 验证内容保留
        assert "这是内容" in refined.text
        assert "更多内容" in refined.text
        assert refined.metadata["refinement_method"] == "rule"
    
    def test_rule_based_removes_excessive_whitespace(self):
        """测试去除多余空白"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_2",
            text="这是    内容。\n\n\n\n更多内容。",
            metadata={}
        )
        
        refined = refiner.transform(chunk)
        
        # 验证多余空白被去除
        assert "     " not in refined.text  # 连续空格
        assert "\n\n\n\n" not in refined.text  # 连续空行
        assert refined.metadata["refinement_method"] == "rule"
    
    def test_rule_based_removes_header_footer_patterns(self):
        """测试去除页眉页脚模式"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_3",
            text="- 1 -\n\n这是内容。\n\n- 2 -",
            metadata={}
        )
        
        refined = refiner.transform(chunk)
        
        # 验证页眉页脚被去除
        assert "- 1 -" not in refined.text
        assert "- 2 -" not in refined.text
        assert "这是内容" in refined.text
    
    def test_rule_based_preserves_content(self):
        """测试保留重要内容"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
        original_text = "这是重要内容。包含技术术语：Python、RAG、LLM。"
        chunk = Chunk(
            id="test_chunk_4",
            text=original_text,
            metadata={}
        )
        
        refined = refiner.transform(chunk)
        
        # 验证重要内容被保留
        assert "重要内容" in refined.text
        assert "Python" in refined.text
        assert "RAG" in refined.text
        assert "LLM" in refined.text
    
    def test_rule_based_handles_empty_text(self):
        """测试处理空文本"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_5",
            text="   \n\n   ",
            metadata={}
        )
        
        # 清理后为空字符串会导致 Chunk 创建失败
        # 因为 Chunk 模型不允许空文本
        # 所以应该抛出错误
        with pytest.raises(ValueError, match="不能为空"):
            refined = refiner.transform(chunk)
    
    def test_rule_based_preserves_metadata(self):
        """测试保留元数据"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_6",
            text="测试内容",
            metadata={
                "source_path": "/path/to/file.pdf",
                "chunk_index": 0,
                "custom_field": "custom_value"
            }
        )
        
        refined = refiner.transform(chunk)
        
        # 验证元数据被保留
        assert refined.metadata["source_path"] == "/path/to/file.pdf"
        assert refined.metadata["chunk_index"] == 0
        assert refined.metadata["custom_field"] == "custom_value"
        assert refined.metadata["refinement_method"] == "rule"
    
    def test_rule_based_preserves_position_info(self):
        """测试保留位置信息"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
        chunk = Chunk(
            id="test_chunk_7",
            text="测试内容",
            metadata={},
            start_offset=100,
            end_offset=200
        )
        
        refined = refiner.transform(chunk)
        
        # 验证位置信息被保留
        assert refined.start_offset == 100
        assert refined.end_offset == 200


class TestChunkRefinerLLM:
    """ChunkRefiner LLM 重写测试"""
    
    def test_llm_refinement_with_mock_llm(self):
        """测试使用 Mock LLM 进行重写"""
        settings = _create_test_settings(enable_llm_refinement=True)
        
        # 创建 Mock LLM
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "这是 LLM 重写后的内容。"
        
        refiner = ChunkRefiner(settings.ingestion, llm=mock_llm)
        
        chunk = Chunk(
            id="test_chunk_8",
            text="原始内容。",
            metadata={}
        )
        
        refined = refiner.transform(chunk)
        
        # 验证 LLM 被调用
        assert mock_llm.chat.called
        # 验证使用 LLM 重写
        assert refined.metadata["refinement_method"] == "llm"
        # 验证文本被重写
        assert refined.text == "这是 LLM 重写后的内容。"
    
    def test_llm_refinement_uses_prompt_template(self):
        """测试 LLM 重写使用 Prompt 模板"""
        settings = _create_test_settings(enable_llm_refinement=True)
        
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "重写后的内容"
        
        custom_prompt = "Custom prompt: {chunk_text}"
        refiner = ChunkRefiner(settings.ingestion, llm=mock_llm, prompt_template=custom_prompt)
        
        chunk = Chunk(
            id="test_chunk_9",
            text="原始内容",
            metadata={}
        )
        
        refined = refiner.transform(chunk)
        
        # 验证使用了自定义 Prompt
        call_args = mock_llm.chat.call_args[0][0]
        assert "Custom prompt" in call_args[0]["content"]
        assert "原始内容" in call_args[0]["content"]
    
    def test_llm_refinement_fallback_on_failure(self):
        """测试 LLM 失败时降级到规则结果"""
        settings = _create_test_settings(enable_llm_refinement=True)
        
        # 创建会失败的 Mock LLM
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("LLM 调用失败")
        
        refiner = ChunkRefiner(settings.ingestion, llm=mock_llm)
        
        original_text = "这是原始内容。"
        chunk = Chunk(
            id="test_chunk_10",
            text=original_text,
            metadata={}
        )
        
        # 不应该抛出异常，应该降级
        refined = refiner.transform(chunk)
        
        # 验证降级到规则模式
        assert refined.metadata["refinement_method"] == "rule_fallback"
        assert "refinement_fallback_reason" in refined.metadata
        # 验证使用规则清理的结果（文本应该被清理但保留内容）
        assert "原始内容" in refined.text or original_text.strip() in refined.text
    
    def test_llm_refinement_disabled(self):
        """测试 LLM 重写被禁用时只使用规则"""
        settings = _create_test_settings(enable_llm_refinement=False)
        
        mock_llm = MagicMock()
        refiner = ChunkRefiner(settings.ingestion, llm=mock_llm)
        
        chunk = Chunk(
            id="test_chunk_11",
            text="测试内容",
            metadata={}
        )
        
        refined = refiner.transform(chunk)
        
        # 验证 LLM 未被调用
        assert not mock_llm.chat.called
        # 验证使用规则模式
        assert refined.metadata["refinement_method"] == "rule"
    
    def test_llm_refinement_empty_response_fallback(self):
        """测试 LLM 返回空结果时降级"""
        settings = _create_test_settings(enable_llm_refinement=True)
        
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ""  # 返回空结果
        
        refiner = ChunkRefiner(settings.ingestion, llm=mock_llm)
        
        original_text = "原始内容"
        chunk = Chunk(
            id="test_chunk_12",
            text=original_text,
            metadata={}
        )
        
        refined = refiner.transform(chunk)
        
        # 验证降级
        assert refined.metadata["refinement_method"] == "rule_fallback"
        assert "refinement_fallback_reason" in refined.metadata


class TestChunkRefinerEdgeCases:
    """ChunkRefiner 边界情况测试"""
    
    def test_transform_empty_chunk(self):
        """测试空 Chunk 抛出错误"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
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
        refiner = ChunkRefiner(settings.ingestion)
        
        with pytest.raises(ValueError, match="不能为空"):
            refiner.transform(None)
    
    def test_prompt_template_loading(self):
        """测试 Prompt 模板加载"""
        settings = _create_test_settings()
        refiner = ChunkRefiner(settings.ingestion)
        
        # 验证 Prompt 模板已加载
        assert refiner._prompt_template is not None
        assert len(refiner._prompt_template) > 0
        assert "{chunk_text}" in refiner._prompt_template
    
    def test_custom_prompt_template(self):
        """测试自定义 Prompt 模板"""
        settings = _create_test_settings()
        custom_prompt = "Custom: {chunk_text}"
        refiner = ChunkRefiner(settings.ingestion, prompt_template=custom_prompt)
        
        assert refiner._prompt_template == custom_prompt
