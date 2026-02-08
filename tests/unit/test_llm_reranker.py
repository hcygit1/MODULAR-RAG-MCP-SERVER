"""
LLM Reranker 单元测试

使用 mock LLM 测试 LLM Reranker 的实现。
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.core.settings import (
    Settings, LLMConfig, VisionLLMConfig, EmbeddingConfig,
    VectorStoreConfig, RetrievalConfig, RerankConfig,
    EvaluationConfig, ObservabilityConfig, IngestionConfig,
    LoggingConfig, DashboardConfig
)
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.reranker.reranker_factory import RerankerFactory
from src.libs.vector_store.base_vector_store import QueryResult


def _create_test_settings(
    llm_config: LLMConfig,
    rerank_config: RerankConfig = None
) -> Settings:
    """创建测试用的 Settings 对象"""
    if rerank_config is None:
        rerank_config = RerankConfig(
            backend="llm",
            model="gpt-4o",
            top_m=30,
            timeout_seconds=5
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
        chunk_size=512,
        chunk_overlap=50,
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


def _create_test_prompt_file(content: str) -> str:
    """创建临时 prompt 文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(content)
        return f.name


def test_factory_creates_llm_reranker():
    """测试工厂可以创建 LLM Reranker"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    # Mock LLM 的 chat 方法
    with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '["id1", "id2", "id3"]'
        mock_llm_factory.return_value = mock_llm
        
        reranker = RerankerFactory.create(settings)
        
        assert isinstance(reranker, LLMReranker)
        assert reranker.get_backend() == "llm"


def test_llm_reranker_initialization():
    """测试 LLM Reranker 初始化"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    # 创建临时 prompt 文件
    prompt_content = "Query: {query}\n\nCandidates:\n{candidates}\n\nRanked chunk IDs:"
    prompt_path = _create_test_prompt_file(prompt_content)
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            
            assert reranker.get_backend() == "llm"
            assert reranker._prompt_template == prompt_content
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_prompt_file_not_found():
    """测试 prompt 文件不存在时抛出错误"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    with pytest.raises(ValueError, match="Prompt 模板文件不存在"):
        LLMReranker(settings, prompt_path="/nonexistent/path/rerank.txt")


def test_llm_reranker_empty_query():
    """测试查询为空时抛出错误"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2")
            ]
            
            with pytest.raises(ValueError, match="查询文本不能为空"):
                reranker.rerank("", candidates)
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_empty_candidates():
    """测试候选列表为空时抛出错误"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            
            with pytest.raises(ValueError, match="候选列表不能为空"):
                reranker.rerank("test query", [])
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_single_candidate():
    """测试单个候选时直接返回"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [QueryResult(id="id1", score=0.9, text="text1")]
            
            result = reranker.rerank("test query", candidates)
            
            assert len(result) == 1
            assert result[0].id == "id1"
            # LLM 不应该被调用
            mock_llm.chat.assert_not_called()
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_successful_rerank():
    """测试成功重排序"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}\nRanked chunk IDs:")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            # LLM 返回排序后的 ID 列表（JSON 格式）
            mock_llm.chat.return_value = '["id2", "id3", "id1"]'
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2"),
                QueryResult(id="id3", score=0.7, text="text3")
            ]
            
            result = reranker.rerank("test query", candidates)
            
            assert len(result) == 3
            assert result[0].id == "id2"  # 第一个应该是 id2（LLM 返回的第一个）
            assert result[1].id == "id3"
            assert result[2].id == "id1"
            
            # 验证 LLM 被调用了一次
            mock_llm.chat.assert_called_once()
            call_args = mock_llm.chat.call_args[0][0]
            assert call_args[0]["role"] == "user"
            assert "test query" in call_args[0]["content"]
            assert "id1" in call_args[0]["content"]
            assert "id2" in call_args[0]["content"]
            assert "id3" in call_args[0]["content"]
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_json_with_extra_text():
    """测试 LLM 返回包含额外文本的 JSON"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            # LLM 返回包含额外文本的 JSON
            mock_llm.chat.return_value = 'Here are the ranked IDs:\n["id2", "id1"]\nThese are the most relevant.'
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2")
            ]
            
            result = reranker.rerank("test query", candidates)
            
            assert len(result) == 2
            assert result[0].id == "id2"
            assert result[1].id == "id1"
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_invalid_json():
    """测试 LLM 返回无效 JSON 时抛出错误"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = "This is not JSON"
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2")
            ]
            
            with pytest.raises(RuntimeError, match="LLM Reranker 解析失败"):
                reranker.rerank("test query", candidates)
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_invalid_id():
    """测试 LLM 返回无效 ID 时抛出错误"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            # LLM 返回不存在的 ID
            mock_llm.chat.return_value = '["id999", "id1"]'
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2")
            ]
            
            with pytest.raises(RuntimeError, match="LLM 返回的 ID 不在候选列表中"):
                reranker.rerank("test query", candidates)
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_non_list_response():
    """测试 LLM 返回非列表格式时抛出错误"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            # LLM 返回字典而不是列表
            mock_llm.chat.return_value = '{"ids": ["id1", "id2"]}'
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2")
            ]
            
            with pytest.raises(RuntimeError, match="LLM 返回的不是数组格式"):
                reranker.rerank("test query", candidates)
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_partial_ranking():
    """测试 LLM 只返回部分候选的排序"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            # LLM 只返回部分 ID
            mock_llm.chat.return_value = '["id2", "id1"]'
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2"),
                QueryResult(id="id3", score=0.7, text="text3")
            ]
            
            result = reranker.rerank("test query", candidates)
            
            # 应该包含所有候选，但 id2 和 id1 在前面
            assert len(result) == 3
            assert result[0].id == "id2"
            assert result[1].id == "id1"
            assert result[2].id == "id3"  # id3 没有被排序，放在最后
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_llm_failure():
    """测试 LLM 调用失败时抛出错误"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_path = _create_test_prompt_file("Query: {query}\nCandidates:\n{candidates}")
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = RuntimeError("LLM API 调用失败")
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2")
            ]
            
            with pytest.raises(RuntimeError, match="LLM Reranker 调用失败"):
                reranker.rerank("test query", candidates)
    finally:
        Path(prompt_path).unlink()


def test_llm_reranker_prompt_building():
    """测试 prompt 构建逻辑"""
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        openai_api_key="test-key"
    )
    settings = _create_test_settings(llm_config)
    
    prompt_template = "Query: {query}\n\nCandidates:\n{candidates}\n\nRanked chunk IDs:"
    prompt_path = _create_test_prompt_file(prompt_template)
    
    try:
        with patch('src.libs.llm.llm_factory.LLMFactory.create') as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.chat.return_value = '["id1", "id2"]'
            mock_llm_factory.return_value = mock_llm
            
            reranker = LLMReranker(settings, prompt_path=prompt_path)
            candidates = [
                QueryResult(id="id1", score=0.9, text="text1"),
                QueryResult(id="id2", score=0.8, text="text2")
            ]
            
            reranker.rerank("test query", candidates)
            
            # 验证 prompt 内容
            call_args = mock_llm.chat.call_args[0][0]
            prompt_content = call_args[0]["content"]
            
            assert "test query" in prompt_content
            assert "id1" in prompt_content
            assert "id2" in prompt_content
            assert "text1" in prompt_content or "text2" in prompt_content
    finally:
        Path(prompt_path).unlink()
