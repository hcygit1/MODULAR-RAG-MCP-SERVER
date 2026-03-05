"""
配置加载测试

测试配置文件的加载、解析和校验功能。
"""
import pytest
import tempfile
import yaml
from pathlib import Path

from src.core.settings import (
    load_settings,
    validate_settings,
    Settings,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
)


def test_load_valid_settings():
    """测试加载有效的配置文件"""
    settings = load_settings("config/settings.yaml")
    assert isinstance(settings, Settings)
    assert hasattr(settings, "llm")
    assert hasattr(settings, "embedding")
    assert hasattr(settings, "vector_store")


def test_load_settings_file_not_found():
    """测试配置文件不存在的情况"""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_settings("config/nonexistent.yaml")
    
    assert "配置文件不存在" in str(exc_info.value)


def test_load_settings_missing_required_field():
    """测试缺失必填字段的情况"""
    # 创建临时配置文件，缺少 embedding.provider
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "llm": {
                "provider": "azure",
                "model": "gpt-4o",
            },
            "vision_llm": {
                "provider": "azure",
                "model": "gpt-4o",
            },
            "embedding": {
                # 缺少 provider
                "model": "text-embedding-3-small",
            },
            "vector_store": {
                "backend": "chroma",
                "persist_path": "./data/db/chroma",
                "collection_name": "knowledge_base",
            },
            "retrieval": {
                "sparse_backend": "bm25",
                "fusion_algorithm": "rrf",
                "top_k_dense": 20,
                "top_k_sparse": 20,
                "top_k_final": 10,
            },
            "rerank": {
                "backend": "none",
                "model": "",
                "top_m": 30,
                "timeout_seconds": 5,
            },
            "evaluation": {
                "backends": ["ragas"],
                "golden_test_set": "./tests/fixtures/golden_test_set.json",
            },
            "observability": {
                "enabled": True,
                "logging": {
                    "log_file": "./logs/traces.jsonl",
                    "log_level": "INFO",
                },
                "detail_level": "standard",
                "dashboard": {
                    "enabled": True,
                    "port": 8501,
                },
            },
            "ingestion": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "enable_llm_refinement": False,
                "enable_metadata_enrichment": True,
                "enable_image_captioning": True,
                "batch_size": 32,
            },
        }
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            load_settings(temp_path)
        
        assert "缺失必填字段" in str(exc_info.value)
        assert "embedding.provider" in str(exc_info.value)
    finally:
        Path(temp_path).unlink()


def test_load_settings_missing_multiple_fields():
    """测试缺失多个必填字段的情况"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "llm": {
                # 缺少 provider
                "model": "gpt-4o",
            },
            "vision_llm": {
                "provider": "azure",
                "model": "gpt-4o",
            },
            "embedding": {
                # 缺少 provider
                "model": "text-embedding-3-small",
            },
            "vector_store": {
                "backend": "chroma",
                "persist_path": "./data/db/chroma",
                "collection_name": "knowledge_base",
            },
            "retrieval": {
                "sparse_backend": "bm25",
                "fusion_algorithm": "rrf",
                "top_k_dense": 20,
                "top_k_sparse": 20,
                "top_k_final": 10,
            },
            "rerank": {
                "backend": "none",
                "model": "",
                "top_m": 30,
                "timeout_seconds": 5,
            },
            "evaluation": {
                "backends": ["ragas"],
                "golden_test_set": "./tests/fixtures/golden_test_set.json",
            },
            "observability": {
                "enabled": True,
                "logging": {
                    "log_file": "./logs/traces.jsonl",
                    "log_level": "INFO",
                },
                "detail_level": "standard",
                "dashboard": {
                    "enabled": True,
                    "port": 8501,
                },
            },
            "ingestion": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "enable_llm_refinement": False,
                "enable_metadata_enrichment": True,
                "enable_image_captioning": True,
                "batch_size": 32,
            },
        }
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError) as exc_info:
            load_settings(temp_path)
        
        error_msg = str(exc_info.value)
        assert "缺失必填字段" in error_msg
        assert "llm.provider" in error_msg
        assert "embedding.provider" in error_msg
    finally:
        Path(temp_path).unlink()


def test_validate_settings_valid():
    """测试校验有效配置"""
    settings = load_settings("config/settings.yaml")
    # 不应该抛出异常
    validate_settings(settings)


def test_validate_settings_sqlite_requires_fts5():
    """backend=sqlite 时必须 sparse_backend=fts5（统一存储约束）"""
    settings = load_settings("config/settings.yaml")
    # 模拟 sqlite + bm25 的非法组合
    settings.vector_store.backend = "sqlite"
    settings.vector_store.sqlite_path = "./data/db/rag.sqlite"
    settings.vector_store.embedding_dim = 1536
    settings.retrieval.sparse_backend = "bm25"
    with pytest.raises(ValueError) as exc_info:
        validate_settings(settings)
    assert "sparse_backend" in str(exc_info.value)
    assert "fts5" in str(exc_info.value)


def test_load_settings_local_override():
    """测试 settings.local.yaml 覆盖敏感项"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "settings.yaml"
        local_path = Path(tmpdir) / "settings.local.yaml"
        # 最小有效 base 配置
        base = {
            "llm": {"provider": "qwen", "model": "qwen-turbo", "dashscope_api_key": ""},
            "vision_llm": {"provider": "qwen", "model": "qwen-vl-max", "dashscope_api_key": ""},
            "embedding": {
                "provider": "qwen",
                "model": "text-embedding-v1",
                "dashscope_api_key": "",
            },
            "vector_store": {
                "backend": "qdrant",
                "persist_path": "./data/db/qdrant",
                "collection_name": "kb",
            },
            "retrieval": {"sparse_backend": "bm25", "fusion_algorithm": "rrf", "top_k_dense": 20, "top_k_sparse": 20, "top_k_final": 10},
            "rerank": {"backend": "none", "model": "", "top_m": 30, "timeout_seconds": 5},
            "evaluation": {"backends": ["ragas"], "golden_test_set": "./tests/fixtures/golden_test_set.json"},
            "observability": {
                "enabled": True,
                "logging": {"log_file": "./logs/traces.jsonl", "log_level": "INFO"},
                "detail_level": "standard",
                "dashboard": {"enabled": True, "port": 8501},
            },
            "ingestion": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "enable_llm_refinement": False,
                "enable_metadata_enrichment": True,
                "enable_image_captioning": True,
                "batch_size": 32,
            },
        }
        with open(base_path, "w") as f:
            yaml.dump(base, f, allow_unicode=True)
        with open(local_path, "w") as f:
            yaml.dump({"llm": {"dashscope_api_key": "sk-local-override"}}, f, allow_unicode=True)

        settings = load_settings(str(base_path))
        assert settings.llm.dashscope_api_key == "sk-local-override"


def test_settings_structure():
    """测试 Settings 对象的结构"""
    settings = load_settings("config/settings.yaml")

    # 验证所有主要配置都存在
    assert hasattr(settings, "llm")
    assert hasattr(settings, "vision_llm")
    assert hasattr(settings, "embedding")
    assert hasattr(settings, "vector_store")
    assert hasattr(settings, "retrieval")
    assert hasattr(settings, "rerank")
    assert hasattr(settings, "evaluation")
    assert hasattr(settings, "observability")
    assert hasattr(settings, "ingestion")
    
    # 验证嵌套配置
    assert hasattr(settings.observability, "logging")
    assert hasattr(settings.observability, "dashboard")
    assert isinstance(settings.observability.logging.log_file, str)
    assert isinstance(settings.observability.dashboard.port, int)
